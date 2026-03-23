"""Evaluation loop for the tuned lens model."""
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Literal, Optional

import torch as th
from simple_parsing import field
from tqdm.auto import tqdm
from transformers import PreTrainedModel

from tuned_lens.nn.lenses import Lens, LogitLens, TunedLens
from tuned_lens.scripts.ingredients import (
    Data,
    Distributed,
    Model,
)
from tuned_lens.stats import LogitStats
from tuned_lens.utils import (
    mask_input_ids_for_mlm,
    maybe_all_reduce,
    pytree_map,
    pytree_stack,
    shift_labels,
    shift_preds,
)

LensType = Literal["logit", "tuned"]


logger = logging.getLogger(__name__)


def _nested_dict():
    return defaultdict(_nested_dict)


@dataclass
class Eval:
    """Type hinting for CLI args."""

    data: Data

    model: Model

    dist: Distributed

    output: Path = field(alias=["-o"])
    """Folder to save the eval results to."""

    lens_name: Optional[str] = field(alias=["-l"], default=None)
    """Path to the tuned lens model to evaluate. Defaults to None."""

    logit: bool = True
    """Whether to evaluate the logit lens"""

    seed: int = 42
    """Random seed used for data shuffling."""

    tokens: Optional[int] = None
    """Number of tokens to evaluate on. If None, will use the entire dataset."""

    token_shift: Optional[int] = field(default=None)
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    per_gpu_batch_size: int = 1
    """Number of samples to try to fit on a GPU at once."""

    layer_transfer: bool = field(action="store_true")
    """Evaluate the transfer of the lens to different layers of the transformer."""

    record_logit_stats: bool = field(action="store_true")
    """Record the statistics of the marginal token distribution at each layer."""

    mlm_probability: float = 0.15
    """Masking probability for masked-LM models."""

    def _prepare_batch_for_encoder_decoder_model(
        self, batch: dict, model: PreTrainedModel
    ) -> dict:
        """Populate required decoder inputs for encoder-decoder models."""
        if self.model.model_type != "encoder-decoder":
            raise ValueError(
                "Encoder-decoder batch preparation called for non-encoder-decoder model."
            )
        if "decoder_input_ids" not in batch:
            if getattr(model.config, "decoder_start_token_id", None) is None:
                raise ValueError(
                    "Encoder-decoder models must set `decoder_start_token_id` in config."
                )
            shift_right = getattr(model, "_shift_right", None)
            if shift_right is None:
                raise ValueError("Encoder-decoder model is missing `_shift_right`.")
            batch["decoder_input_ids"] = shift_right(batch["input_ids"])

        if "decoder_attention_mask" not in batch:
            pad_token_id = getattr(model.config, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError(
                    "Encoder-decoder models must set `pad_token_id` in config."
                )
            batch["decoder_attention_mask"] = (
                batch["decoder_input_ids"] != pad_token_id
            ).long()

        return batch

    @staticmethod
    def _masked_mean(x: th.Tensor, valid_mask: Optional[th.Tensor]) -> th.Tensor:
        """Average tensor values, optionally over a boolean mask."""
        if valid_mask is None:
            return x.mean()
        if valid_mask.any():
            return x[valid_mask].mean()
        return x.new_tensor(0.0)

    def load_lens(self, model: PreTrainedModel) -> dict[str, Lens]:
        """Load the tuned lens model."""
        lenses = {}
        if self.logit:
            lenses["logit"] = LogitLens.from_model(model)
        if self.lens_name is not None:
            lenses["tuned"] = TunedLens.from_model_and_pretrained(model, self.lens_name)
        return lenses

    def calculate_batch_limit(self, tokens_per_sample: int):
        """Calculate the total number of batches to evaluate on."""
        assert self.tokens is not None
        global_batch_size = self.dist.world_size * self.per_gpu_batch_size
        tokens_per_batch = global_batch_size * tokens_per_sample
        return self.tokens // tokens_per_batch

    def _initialize_logit_stats_recorders(
        self, lenses: dict[str, Lens], total_layers: int
    ):
        if self.record_logit_stats:
            self.logit_stats_recorders = {
                lens_type: {f"layer_{i}": LogitStats() for i in range(total_layers)}
                for lens_type in lenses.keys()
            }
            self.logit_stats_recorder_final = LogitStats()
        else:
            self.logit_stats_recorders = None
            self.logit_stats_recorder_final = None

    def _record_logit_stats(self, logp: th.Tensor, layer: int, lens_type: str):
        if self.logit_stats_recorders is not None:
            self.logit_stats_recorders[lens_type][f"layer_{layer}"].update(
                logp, assume_normalized=True
            )

    def _record_logit_stats_final(self, logp: th.Tensor):
        if self.logit_stats_recorder_final is not None:
            self.logit_stats_recorder_final.update(logp, assume_normalized=True)

    def _save_logit_stats(self) -> defaultdict:
        logit_stats = _nested_dict()
        if self.logit_stats_recorders is not None:
            for lens_type, recorders in self.logit_stats_recorders.items():
                for layer, recorder in recorders.items():
                    recorder.all_reduce_()
                    logit_stats[lens_type]["logit_stats"][layer] = (
                        recorder.marginal_probs.cpu().numpy().tolist()
                    )

        if self.logit_stats_recorder_final is not None:
            self.logit_stats_recorder_final.all_reduce_()
            logit_stats["baseline"]["logit_stats"]["final"] = (
                self.logit_stats_recorder_final.marginal_probs.cpu().numpy().tolist()
            )

        return logit_stats

    def _evaluate_lenses_on_hidden(
        self,
        lenses: dict[str, Lens],
        hidden: th.Tensor,
        layer: int,
        final_probs: th.Tensor,
        final_lps: th.Tensor,
        labels: th.Tensor,
        shift: int,
        valid_mask: Optional[th.Tensor],
        batch_output: defaultdict,
        total_layers: int,
    ):
        """Evaluate a lens at a given layer. Batch output is modified in place.

        Args:
            lenses: The dictionary of lenses to evaluate on this hidden state.
            hidden: (batch x seq x d_model) The hidden states of the transformer.
            layer: The layer this hidden state is from.
            final_probs: (batch x seq x vocab) The final probabilities of
                the transformer.
            final_lps: (batch x seq x vocab) The final log probabilities
                of the transformer.
            labels: (batch x seq) The labels for the transformer.
            batch_output: Where to store the logging results.
            total_layers: The total number of layers in the transformer.
            logp_stats: where to record the logging results.
        """
        for lens_type, lens in lenses.items():
            layer_name = f"layer_{layer}"
            lens_lps = lens(hidden, idx=layer).log_softmax(dim=-1)
            lens_probs = lens_lps.exp()
            shifted_lens_lps = shift_preds(lens_lps, shift)

            self._record_logit_stats(lens_lps, layer, lens_type)

            ce = th.nn.functional.cross_entropy(
                shifted_lens_lps.flatten(0, 1),
                labels.flatten(),
                reduction="none",
                ignore_index=-100,
            )
            batch_output[lens_type]["ce"][layer_name] = self._masked_mean(
                ce.view_as(labels), valid_mask
            )

            entropy = th.sum(
                -lens_probs * lens_lps, dim=-1
            )
            batch_output[lens_type]["entropy"][layer_name] = self._masked_mean(
                entropy, valid_mask
            )

            kl = th.sum(
                final_probs * (final_lps - lens_lps), dim=-1
            )
            batch_output[lens_type]["kl"][layer_name] = self._masked_mean(kl, valid_mask)

            if self.layer_transfer:
                for i in range(total_layers):
                    trans_name = f"layer_{i}"
                    transfer_lps = lens(hidden, idx=i).log_softmax(dim=-1)
                    shifted_transfer_lps = shift_preds(transfer_lps, shift)
                    batch_output[lens_type]["layer_transfer"]["ce"][trans_name][
                        layer_name
                    ] = self._masked_mean(
                        th.nn.functional.cross_entropy(
                            shifted_transfer_lps.flatten(0, 1),
                            labels.flatten(),
                            reduction="none",
                            ignore_index=-100,
                        ).view_as(labels),
                        valid_mask,
                    )
                    transfer_kl = th.sum(lens_probs * (lens_lps - transfer_lps), dim=-1)
                    batch_output[lens_type]["layer_transfer"]["kl"][trans_name][
                        layer_name
                    ] = self._masked_mean(transfer_kl, valid_mask)

    @th.autocast("cuda", enabled=th.cuda.is_available())
    @th.no_grad()
    def execute(self):
        """Evaluates a TunedLens model against a transformer on a dataset."""
        # Load model, tokenizer, data, and lens
        self.dist.init()
        model = tokenizer = data = lenses = nats_to_bpb = None

        # See comment in train_loop.py for why we do this
        load_device = self.dist.device if not self.dist.fsdp else None
        if self.dist.primary:
            # Let the primary processes populate the cache
            model, tokenizer = self.model.load(load_device)
            data, nats_to_bpb = self.data.load(
                tokenizer, model_type=self.model.model_type
            )
            lenses = self.load_lens(model)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
            model, tokenizer = self.model.load(load_device, must_use_cache=True)
            data, nats_to_bpb = self.data.load(
                tokenizer, model_type=self.model.model_type
            )
            lenses = self.load_lens(model)

        assert model and tokenizer and data and lenses and nats_to_bpb

        model_type = self.model.model_type
        is_encoder_decoder_model = model_type == "encoder-decoder"
        uses_masked_objective = model_type in {"masked", "encoder-decoder"}

        if uses_masked_objective:
            mask_token_id = tokenizer.mask_token_id
            if mask_token_id is None:
                raise ValueError(
                    f"Tokenizer '{tokenizer.__class__.__name__}' does not define a "
                    "mask token, but this model_type uses masked-token supervision."
                )
            self._mask_token_id = int(mask_token_id)
            self._special_token_ids = list(tokenizer.all_special_ids)

        model = self.dist.shard_model(model)
        # Note since we are not training we can just move the lens to the device.
        # No need to use DDP
        lenses = {name: lens.to(self.dist.device) for name, lens in lenses.items()}
        dl = self.dist.dataloader(data)
        dl.seed(self.seed)

        for lens in lenses.values():
            lens.eval()

        if self.tokens is not None:
            tokens_per_sample = len(data[0]["input_ids"])
            if self.tokens > len(data) * tokens_per_sample:
                raise ValueError(
                    f"Requested to evaluate on {self.tokens} tokens, "
                    f"but dataset only contains {len(data)*tokens_per_sample} tokens."
                )

            batch_limit = self.calculate_batch_limit(tokens_per_sample)
            assert batch_limit > 0, "Batch limit must be positive."
            dl = islice(dl, batch_limit)
            total = batch_limit
        else:
            total = len(data) // self.dist.world_size

        L = model.config.num_hidden_layers

        self._initialize_logit_stats_recorders(lenses, L)

        root_dir = self.output

        root_dir.mkdir(exist_ok=True, parents=True)

        batches = []

        self.dist.barrier()
        logger.info(
            f"All processes initialized. Running evaluation on {total} batches."
        )

        pbar = tqdm(dl, desc="Evaluating", position=self.dist.rank, total=total)
        for batch in pbar:
            batch = self.dist.send_to_device(batch)
            ce_labels = batch["input_ids"]
            if uses_masked_objective:
                masked_input_ids, ce_labels = mask_input_ids_for_mlm(
                    batch["input_ids"],
                    self._mask_token_id,
                    self._special_token_ids,
                    self.mlm_probability,
                )
                batch["input_ids"] = masked_input_ids

            if is_encoder_decoder_model:
                batch = self._prepare_batch_for_encoder_decoder_model(batch, model)
            output = model(**batch, output_hidden_states=True)

            if is_encoder_decoder_model:
                if output.encoder_hidden_states is None:
                    raise ValueError(
                        "Expected `encoder_hidden_states` for encoder-decoder model output."
                    )
                hidden_states = output.encoder_hidden_states[:-1]
            else:
                if output.hidden_states is None:
                    raise ValueError("Model output does not expose `hidden_states`.")
                hidden_states = output.hidden_states[:-1]

            final_lps = output.logits.log_softmax(dim=-1)

            final_probs = final_lps.exp()
            assert not th.isnan(output.logits).any(), "Logits are NaN"

            shift = self.token_shift
            if shift is None:
                shift = 0 if uses_masked_objective else 1 # TODO: potential mismatch between train and eval for KL loss in the case of causal models!

            labels = shift_labels(ce_labels, shift)
            valid_mask = labels != -100 if uses_masked_objective else None

            batch_output = _nested_dict()

            # Compute tuned lens eval and statistics if applicable
            for j, h in zip(range(L), hidden_states):
                self._evaluate_lenses_on_hidden(
                    lenses=lenses,
                    hidden=h,
                    layer=j,
                    final_probs=final_probs,
                    final_lps=final_lps,
                    labels=labels,
                    shift=shift,
                    valid_mask=valid_mask,
                    batch_output=batch_output,
                    total_layers=L,
                )

            baseline_ce = th.nn.functional.cross_entropy(
                shift_preds(final_lps, shift).flatten(0, 1),
                labels.flatten(),
                reduction="none",
                ignore_index=-100,
            )
            batch_output["baseline"]["ce"]["final"] = self._masked_mean(
                baseline_ce.view_as(labels), valid_mask
            )
            baseline_entropy = th.sum(
                -final_probs * final_lps, dim=-1
            )
            batch_output["baseline"]["entropy"]["final"] = self._masked_mean(
                baseline_entropy, valid_mask
            )

            batches.append(pytree_map(th.mean, batch_output))  # type: ignore[arg-type]

            self._record_logit_stats_final(final_lps)

        pbar.close()
        agg = pytree_map(lambda x: nats_to_bpb * x.mean(), pytree_stack(batches))
        agg = pytree_map(lambda x: maybe_all_reduce(x), agg)
        agg = pytree_map(lambda x: x.cpu().numpy().item(), agg)

        assert isinstance(agg, dict)

        batches = pytree_map(lambda x: nats_to_bpb * x, batches)
        batches = pytree_map(lambda x: maybe_all_reduce(x), batches)
        batches = pytree_map(lambda x: x.cpu().item(), batches)
        assert isinstance(batches, list)

        logit_stats = self._save_logit_stats()

        if self.dist.primary:
            with (root_dir / "batches.jsonl").open("w") as f:
                json.dump(batches, f)

            with (root_dir / "aggregate_metrics.json").open("w") as f:
                json.dump(agg, f)

            if self.record_logit_stats:
                with (root_dir / "logit_stats.json").open("w") as f:
                    json.dump(logit_stats, f)

# TODO: review eval script and shapes for maskd vs causal vs encoder-decoder models (and shift = 0 or 1 or 2 or None)! for both ce loss and kl loss!
