"""Training loop for training a TunedLens model against a transformer on a dataset."""
import dataclasses
import enum
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import torch as th
from simple_parsing import field
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torchdata.dataloader2 import DataLoader2
from tqdm.auto import trange
from transformers import PreTrainedModel

import tuned_lens.scripts.ingredients as ing
from tuned_lens import TunedLens
from tuned_lens.utils import (
    mask_input_ids_for_mlm,
    maybe_all_reduce,
    shift_labels,
    shift_preds,
)

logger = logging.getLogger(__name__)


class LossChoice(enum.Enum):
    """Options of what loss to select when training the model."""

    CE = "ce"
    KL = "kl"


@dataclass
class State:
    """All of the stateful information in the training loop."""

    dataloader: DataLoader2
    lens: TunedLens
    opt: Optimizer
    scheduler: LambdaLR
    wandb_id: Optional[str]
    nats_to_bpb: float
    step: int = 0

    def load(self, snapshot_file: Path, device: th.device) -> None:
        """Load a snapshot file."""
        logger.info(f"Loading snapshot from {snapshot_file}...")
        snapshot = th.load(snapshot_file, map_location=device)
        self.step = snapshot["step"]
        self.wandb_id = snapshot["wandb_id"]
        self.lens.load_state_dict(snapshot["lens"])
        self.opt.load_state_dict(snapshot["optim"])
        self.scheduler.load_state_dict(snapshot["scheduler"])
        self.dataloader.load_state_dict(snapshot["dataloader"])

    def save(self, snapshot_file: Path) -> None:
        """Save a snapshot file."""
        logger.info(f"Saving snapshot to {snapshot_file}...")
        if isinstance(self.opt, ZeroRedundancyOptimizer):
            self.opt.consolidate_state_dict()

        th.save(
            {
                "lens": self.lens.state_dict(),
                "optim": self.opt.state_dict(),
                "scheduler": self.scheduler.state_dict(),
                "dataloader": self.dataloader.state_dict(),
                "step": self.step,
                "wandb_id": self.wandb_id,
            },
            snapshot_file,
        )


@dataclass
class Train:
    """Training loop for the tuned lens."""

    model: ing.Model
    """Model configuration."""

    data: ing.Data
    """Data configuration."""

    opt: ing.Optimizer
    """Optimizer configuration."""

    dist: ing.Distributed
    """Configuration for how to distribute the training."""

    output: Path = field(alias=["-o"])
    """Directory to save the lenses to."""

    seed: int = 42
    """Random seed for data shuffling."""

    lens_name_or_path: Optional[str] = field(alias=["-l"], default=None)
    """Name of a pretrained lens to load for fine-tuning."""

    bias_only: Optional[bool] = field(action="store_true")
    """Train only the bias term."""

    num_steps: int = 250
    """Number of training steps."""

    tokens_per_step: int = 2**18
    """Number of tokens per step."""

    wandb: Optional[str] = None
    """Name of run in Weights & Biases."""

    token_shift: Optional[int] = None
    """How to shift the labels wrt the input tokens (1 = next token, 0 = current token,
    -1 = previous token, etc.)"""

    checkpoint_freq: Optional[int] = None
    """Steps between saving a checkpoint. If None, no checkpoints are saved."""

    checkpoint_dir: Optional[Path] = None
    """Directory to save checkpoints to. If None, will use <output>/checkpoints."""

    loss: LossChoice = LossChoice.KL
    """Loss function to use."""

    mlm_probability: float = 0.15
    """Masking probability for masked-LM models."""

    val_freq: int = 0
    """Validate every N optimizer steps using a held-out 10% split.
    Set to 0 to disable validation."""

    def __post_init__(self):
        """Set defaults for some fields."""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = self.output / "checkpoints"

    def _prepare_batch_for_encoder_decoder_model(
        self, batch: dict, model: Union[PreTrainedModel, FSDP]
    ) -> dict:
        """Populate required decoder inputs for encoder-decoder models."""
        if self.model.model_type != "encoder-decoder":
            raise ValueError(
                "Encoder-decoder batch preparation called for non-encoder-decoder model."
            )
        wrapped = getattr(model, "module", model)
        if "decoder_input_ids" not in batch:
            if getattr(wrapped.config, "decoder_start_token_id", None) is None:
                raise ValueError(
                    "Encoder-decoder models must set `decoder_start_token_id` in config."
                )
            shift_right = getattr(wrapped, "_shift_right", None)
            if shift_right is None:
                raise ValueError("Encoder-decoder model is missing `_shift_right`.")
            batch["decoder_input_ids"] = shift_right(batch["input_ids"])

        if "decoder_attention_mask" not in batch:
            pad_token_id = getattr(wrapped.config, "pad_token_id", None)
            if pad_token_id is None:
                raise ValueError(
                    "Encoder-decoder models must set `pad_token_id` in config."
                )
            batch["decoder_attention_mask"] = (
                batch["decoder_input_ids"] != pad_token_id
            ).long()

        return batch

    def get_lens(self, model: PreTrainedModel) -> TunedLens:
        """Load or create a TunedLens model."""
        if self.lens_name_or_path is None:
            logger.info("Randomly initializing lens...")
            lens = TunedLens.from_model(model)
        else:
            logger.info("Loading pretrained lens...")
            lens = TunedLens.from_model_and_pretrained(model, self.lens_name_or_path)

        dtypes = {p.dtype for p in lens.parameters()}
        assert (
            len(dtypes) == 1
        ), f"Expected all parameters to have the same dtype, got {dtypes}"

        lens_dtype = next(iter(dtypes))
        lens_size = sum(p.numel() * p.element_size() for p in lens.parameters())

        # Include the optimizer state in the memory usage
        num_bytes = lens_size * (self.opt.per_parameter_optim_state_size() + 1)
        logger.info(
            f"Tuned lens memory usage: {num_bytes / 2 ** 20:.2f} MB in {lens_dtype}"
        )

        if self.bias_only:
            logger.info("Freezing the matrix weights to train only the bias terms.")
            for probe in lens:
                probe.weight.requires_grad_(False)

        return lens

    def _get_wandb_id(self) -> Optional[str]:
        if not self.dist.primary or not self.wandb:
            return None

        from wandb.sdk.lib import runid

        return runid.generate_id()

    def _init_logging(self, model_name: str, lens: TunedLens, wandb_id: Optional[str]):
        """Initialize logging to weights and biases."""
        if not self.dist.primary or not self.wandb:
            return

        logger.debug("Initializing Weights & Biases ...")
        import wandb

        wandb.init(
            config=dataclasses.asdict(self),
            group=model_name,
            name=self.wandb,
            id=wandb_id,
            resume="allow",
        )
        wandb.watch(lens)

    def _log(
        self,
        opt: th.optim.Optimizer,
        step: int,
        losses: dict[str, list[float]],
        tuned_lens: TunedLens,
        nats_to_bpb: float,
    ):
        """Log statistics about the training process to weights and biases."""
        if not self.dist.primary or not self.wandb:
            return

        import wandb

        log_dict = {}
        log_dict.update(
            {f"loss/{k}": th.tensor(v).mean() * nats_to_bpb for k, v in losses.items()}
        )

        # Log statistics about optimizer & probes
        for i, probe in enumerate(tuned_lens):
            name = "input" if i == 0 else f"{i - 1}.ffn"
            states = [opt.state[p] for p in probe.parameters()]

            # Approximate the true grad norm using the optimizer's moving
            # avg
            corr = 1 - self.opt.momentum**step
            if self.opt.optimizer == "sgd" and not self.opt.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [
                        # Undo PyTorch's scaling of the gradient by
                        # 1 / (1 - β)
                        (1 - self.opt.momentum) * s["momentum_buffer"].flatten() / corr
                        for s in states
                    ]
                ).norm()
            elif self.opt.optimizer == "adam" and not self.opt.zero:
                log_dict["grad_norm/" + name] = th.cat(
                    [s["exp_avg"].flatten() / corr for s in states]
                ).norm()

            if isinstance(probe, th.nn.Linear):
                log_dict["bias_norm/" + name] = probe.bias.data.norm()
                log_dict["weight_norm/" + name] = probe.weight.data.norm()

        wandb.log(log_dict)

    def snapshot(self, state: State):
        """Save a snapshot of the training process to disk."""
        if self.dist.primary:
            assert self.checkpoint_dir is not None
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
            state.save(self.checkpoint_dir / f"snapshot_{state.step}.pth")

    def load_recent_snapshot(self, state: State) -> None:
        """Load the most recent snapshot of the training process from disk."""
        assert self.checkpoint_dir is not None

        if not self.checkpoint_dir.exists():
            logger.warning("No checkpoint directory found. Snapshotting is disabled.")
            return None

        # Find the folder containing the most recent snapshot
        def sort_key_from_path(p: Path):
            if match := re.match(r".*snapshot_(\d+)\.pth", str(p)):
                return int(match.group(1))
            else:
                return -1

        snapshot_location = max(
            self.checkpoint_dir.glob("snapshot_*.pth"),
            key=sort_key_from_path,
            default=None,
        )

        if snapshot_location is None:
            return None

        state.load(snapshot_location, self.dist.device)

    def calculate_gradient_accumulation_steps(
        self, tokens_per_sample: int, total_samples: int
    ) -> int:
        """Calculate the number of batches of data to process before taking a step."""
        # chunk_and_tokenize ensures the samples are all the same length
        # TODO: tokens_per_sample = max_seq_len for per-sequence tokenization,
        # but a fraction of those positions are <pad> (excluded from CE loss via
        # ignore_index=-100). So `tokens_per_step` over-counts real tokens for any
        # model going through _tokenize_perseq_fn (all MLMs + ProGen3).
        samples_per_step, rem = divmod(self.tokens_per_step, tokens_per_sample)
        if rem:
            raise ValueError(
                f"Number of tokens per step ({self.tokens_per_step:_}) must be "
                f"divisible by the number of tokens per sample ({tokens_per_sample})."
            )

        if total_samples / samples_per_step < self.num_steps:
            raise ValueError(
                f"Can only take {total_samples / samples_per_step:.2f} steps on "
                f"dataset with --tokens_per_step={self.tokens_per_step}."
                f"Requested {self.num_steps} steps."
            )

        global_batch_size = self.dist.per_gpu_batch_size * self.dist.world_size
        grad_acc_steps, rem = divmod(samples_per_step, global_batch_size)
        if rem:
            # If the number of samples per step isn't divisible by the global batch
            # size, use ceil division and let the user know about it.
            grad_acc_steps += 1
            adjusted_count = grad_acc_steps * global_batch_size * tokens_per_sample
            logger.warning(
                f"Note: Increasing grad acc steps from {grad_acc_steps - 1} to "
                f"{grad_acc_steps} to maintain load balance across "
                f"{self.dist.world_size} GPUs."
            )
            logger.warning(
                f"Using {adjusted_count:_} tokens per training step "
                f"({self.tokens_per_step:_} requested)."
            )
        else:
            logger.info(f"Gradient accumulation steps: {grad_acc_steps}")
            logger.info(f"Using {self.tokens_per_step:_} tokens per training step.")
        return grad_acc_steps

    def setup(self) -> tuple[State, Union[PreTrainedModel, FSDP], int]:
        """Initialize the training process."""
        self.dist.init()
        model = tokenizer = data = lens = nats_to_bpb = None

        # Annoyingly, FSDP is incompatible with the `device_map` parameter on
        # `from_pretrained`, because it adds forward hooks to the submodules that move
        # things around to different devices. But `bitsandbytes` requires `device_map`
        # to work at all. So we use `device_map` iff we're using FSDP.
        load_device = self.dist.device if not self.dist.fsdp else None

        if self.dist.primary:
            logger.debug("Primary rank populating cache...")
            model, tokenizer = self.model.load(load_device)
            data, nats_to_bpb = self.data.load(
                tokenizer, model_type=self.model.model_type
            )
            lens = self.get_lens(model)

        self.dist.barrier()  # Wait for primary to finish filling the cache

        if not self.dist.primary:
            # Let the non-primary processes load from the cache
            logger.debug("Non-primary rank loading from cache...")
            model, tokenizer = self.model.load(load_device, must_use_cache=True)
            data, nats_to_bpb = self.data.load(
                tokenizer, model_type=self.model.model_type
            )
            lens = self.get_lens(model)

        assert model and tokenizer and data and lens and nats_to_bpb

        if self.model.model_type in {"masked", "encoder-decoder"}:
            mask_token_id = tokenizer.mask_token_id
            if mask_token_id is None:
                raise ValueError(
                    f"Tokenizer '{tokenizer.__class__.__name__}' does not define a "
                    "mask token, but this model_type uses masked-token supervision."
                )
            self._mask_token_id = int(mask_token_id)
            self._special_token_ids = list(tokenizer.all_special_ids)

        self._pad_token_id = (
            int(tokenizer.pad_token_id)
            if tokenizer.pad_token_id is not None
            else None
        )

        self._val_dataset = None
        if self.val_freq > 0:
            split = data.train_test_split(test_size=0.1, seed=self.seed)
            data = split["train"]
            self._val_dataset = split["test"]
            logger.info(
                f"Data split: {len(data)} train / {len(self._val_dataset)} val samples"
            )

        logger.debug(f"Creating data loader and setting seed to {self.seed} ...")
        dl = self.dist.dataloader(data)
        dl.seed(self.seed)
        logger.debug("Creating optimizer and scheduler ...")
        params = [p for p in lens.parameters() if p.requires_grad]
        opt = self.opt.create_optim(params)
        scheduler = self.opt.create_scheduler(opt, self.num_steps)

        ddp_lens = self.dist.distribute_lens(lens)

        state = State(
            step=0,
            wandb_id=self._get_wandb_id(),
            lens=ddp_lens,  # type: ignore
            opt=opt,
            scheduler=scheduler,
            dataloader=dl,
            nats_to_bpb=nats_to_bpb,
        )

        self.load_recent_snapshot(state)

        # Shard the model using fully shared data parallel
        model = self.dist.shard_model(model)

        self._init_logging(
            model_name=self.model.name, lens=state.lens, wandb_id=state.wandb_id
        )

        tokens_per_sample = len(data[0]["input_ids"])
        grad_acc_steps = self.calculate_gradient_accumulation_steps(
            tokens_per_sample, len(data)
        )

        self.dist.barrier()  # Wait for all processes to finish setup
        logger.info("All processes have completed setup.")
        return state, model, grad_acc_steps

    def execute(self):
        """Trains a TunedLens model against a transformer on a dataset."""
        # Load model, tokenizer, data, and lens
        state, model, grad_acc_steps = self.setup()

        losses = defaultdict(list)
        init_batches = state.step * grad_acc_steps
        total_batches = self.num_steps * grad_acc_steps

        # Wait for all processes to finish setup
        self.dist.barrier()
        logger.info("All processes have completed setup. Starting training.")

        # Main training loop
        t = trange(
            init_batches,
            total_batches,
            desc="Training",
            initial=init_batches,
            total=total_batches,
        )
        # TODO this currently silently fails if the dataloader is exhausted
        for batch_idx, batch in zip(t, state.dataloader):
            assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
            uses_masked_objective = self.model.model_type in {
                "masked",
                "encoder-decoder",
            }
            is_encoder_decoder_model = self.model.model_type == "encoder-decoder"

            batch = self.dist.send_to_device(batch)
            target_labels = batch["input_ids"]
            if uses_masked_objective:
                masked_input_ids, target_labels = mask_input_ids_for_mlm(
                    batch["input_ids"],
                    self._mask_token_id,
                    self._special_token_ids,
                    self.mlm_probability,
                )
                batch["input_ids"] = masked_input_ids

            if is_encoder_decoder_model:
                batch = self._prepare_batch_for_encoder_decoder_model(batch, model)
            with th.no_grad():
                output = model(**batch, output_hidden_states=True)

            final_logits = output.logits
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

            shift = self.token_shift
            if self.loss == LossChoice.CE:
                labels = target_labels

                # Predict the *current* token by default for masked-LMs and the
                # *next* token by default for causal LMs. Encoder-decoder models
                # also use current-token alignment in this training setup.
                if shift is None:
                    shift = 0 if uses_masked_objective else 1
            elif self.loss == LossChoice.KL:
                labels = final_logits.float().log_softmax(dim=-1)

                # Match the *current* token distribution by default
                if shift is None:
                    shift = 0  # unlike CE labels derived from input, shift not needed here!
                if uses_masked_objective:
                    shifted_target_labels = shift_labels(target_labels, shift)
                    valid_kl_mask = shifted_target_labels != -100
                elif self._pad_token_id is not None:
                    valid_kl_mask = batch["input_ids"] != self._pad_token_id
                else:
                    valid_kl_mask = None
            else:
                raise NotImplementedError(f"Unknown loss {self.loss}")

            labels = shift_labels(labels, shift)
            if self.loss == LossChoice.CE and self._pad_token_id is not None:
                # Mask out positions where the shifted label is a pad token so they
                # are excluded from CE loss (cross_entropy ignores index -100).
                # For masked models this is a no-op: pad labels are already -100.
                labels = labels.masked_fill(labels == self._pad_token_id, -100)

            # We do this sequentially to save VRAM
            for i, h in enumerate(hidden_states):
                # We use bfloat16 because it has a larger dynamic range than float16
                # and it seems to remove the need for doing grad scaling, which is very
                # annoying to set up in the context of multiple backward passes.
                with th.autocast(self.dist.device.type, dtype=th.bfloat16):
                    logits = shift_preds(state.lens(h, idx=i), shift)

                    if self.loss == LossChoice.CE:
                        if uses_masked_objective and not (labels != -100).any().item():
                            # Keep the graph valid while contributing no gradient.
                            loss = logits.sum() * 0.0
                        else:
                            loss = th.nn.functional.cross_entropy(
                                logits.flatten(0, -2),
                                labels.flatten(),
                                ignore_index=-100,
                            )
                    elif self.loss == LossChoice.KL:
                        token_kl = th.sum(
                            labels.exp() * (labels - logits.log_softmax(-1)), dim=-1
                        )
                        if valid_kl_mask is None:
                            loss = token_kl.mean()
                        else:
                            valid_count = valid_kl_mask.sum()
                            if valid_count.item() == 0:
                                # Keep the graph valid while contributing no gradient.
                                loss = token_kl.sum() * 0.0
                            else:
                                loss = (
                                    token_kl * valid_kl_mask.to(token_kl.dtype)
                                ).sum() / valid_count.to(token_kl.dtype)
                    else:
                        raise NotImplementedError

                    logging_loss = loss.detach()
                    # TODO: should the reduce across ranks be weighted  by number of masked tokens?!
                    logging_loss = maybe_all_reduce(logging_loss).item()
                    if self.dist.primary:
                        losses[f"translator_{i}"].append(logging_loss)

                    scaled_loss = loss / grad_acc_steps

                scaled_loss.backward()

            step, rem = divmod(batch_idx, grad_acc_steps)
            if rem == grad_acc_steps - 1:
                th.nn.utils.clip_grad_norm_(state.lens.parameters(), 1.0)
                state.opt.step()
                state.opt.zero_grad(set_to_none=False)
                state.scheduler.step()

                # Unwrap the lens from DDP if needed
                lens = getattr(state.lens, "module", state.lens)
                self._log(state.opt, step, losses, lens, state.nats_to_bpb)
                losses.clear()
                state.step = step + 1
                if self.val_freq > 0 and state.step % self.val_freq == 0:
                    self._validate(model, state)
                if (
                    self.checkpoint_freq
                    and step % self.checkpoint_freq == self.checkpoint_freq - 1
                ):
                    self.snapshot(state)

        if self.dist.primary:
            logger.info(f"Saving lens to {self.output}")

            # Unwrap the lens from DDP if needed
            lens = getattr(state.lens, "module", state.lens)
            lens.save(self.output)

    ######### additions for validation loss #######
    # same computations as train step, repeated for validation

    @th.no_grad()
    def _compute_batch_losses_for_validation(
        self, batch: dict, model: Union[PreTrainedModel, FSDP], lens: TunedLens
    ) -> dict[str, float]:
        """Validation-only forward pass returning per-translator scalar losses."""
        uses_masked_objective = self.model.model_type in {
            "masked",
            "encoder-decoder",
        }
        is_encoder_decoder_model = self.model.model_type == "encoder-decoder"
        target_labels = batch["input_ids"]
        if uses_masked_objective:
            masked_input_ids, target_labels = mask_input_ids_for_mlm(
                batch["input_ids"],
                self._mask_token_id,
                self._special_token_ids,
                self.mlm_probability,
            )
            batch["input_ids"] = masked_input_ids

        if is_encoder_decoder_model:
            batch = self._prepare_batch_for_encoder_decoder_model(batch, model)
        output = model(**batch, output_hidden_states=True)
        final_logits = output.logits
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

        shift = self.token_shift
        if self.loss == LossChoice.CE:
            labels = target_labels
            if shift is None:
                shift = 0 if uses_masked_objective else 1
            valid_kl_mask = None
        elif self.loss == LossChoice.KL:
            labels = final_logits.float().log_softmax(dim=-1)
            if shift is None:
                shift = 0
            if uses_masked_objective:
                shifted_target_labels = shift_labels(target_labels, shift)
                valid_kl_mask = shifted_target_labels != -100
            elif self._pad_token_id is not None:
                valid_kl_mask = batch["input_ids"] != self._pad_token_id
            else:
                valid_kl_mask = None
        else:
            raise NotImplementedError(f"Unknown loss {self.loss}")

        labels = shift_labels(labels, shift)
        if self.loss == LossChoice.CE and self._pad_token_id is not None:
            labels = labels.masked_fill(labels == self._pad_token_id, -100)

        result = {}
        for i, h in enumerate(hidden_states):
            with th.autocast(self.dist.device.type, dtype=th.bfloat16):
                logits = shift_preds(lens(h, idx=i), shift)

                if self.loss == LossChoice.CE:
                    if uses_masked_objective and not (labels != -100).any().item():
                        # Keep the graph valid while contributing no gradient.
                        loss = logits.sum() * 0.0
                    else:
                        loss = th.nn.functional.cross_entropy(
                            logits.flatten(0, -2),
                            labels.flatten(),
                            ignore_index=-100,
                        )
                elif self.loss == LossChoice.KL:
                    token_kl = th.sum(
                        labels.exp() * (labels - logits.log_softmax(-1)), dim=-1
                    )
                    if valid_kl_mask is None:
                        loss = token_kl.mean()
                    else:
                        valid_count = valid_kl_mask.sum()
                        if valid_count.item() == 0:
                            # Keep the graph valid while contributing no gradient.
                            loss = token_kl.sum() * 0.0
                        else:
                            loss = (
                                token_kl * valid_kl_mask.to(token_kl.dtype)
                            ).sum() / valid_count.to(token_kl.dtype)
                else:
                    raise NotImplementedError

                logging_loss = maybe_all_reduce(loss.detach()).item()
                if self.dist.primary:
                    result[f"translator_{i}"] = logging_loss

        return result

    def _validate(self, model: Union[PreTrainedModel, FSDP], state: State) -> None:
        """Run validation on the held-out split and log to Weights & Biases."""
        if self._val_dataset is None:
            return

        lens = getattr(state.lens, "module", state.lens)
        was_training = lens.training
        lens.eval()

        val_loader = self.dist.dataloader(self._val_dataset)
        val_loader.seed(self.seed)
        val_losses: dict[str, list[float]] = defaultdict(list)

        try:
            for batch in val_loader:
                assert isinstance(batch, dict), f"Expected dict, got {type(batch)}"
                batch = self.dist.send_to_device(batch)
                for k, v in self._compute_batch_losses_for_validation(
                    batch, model, lens
                ).items():
                    if self.dist.primary:
                        val_losses[k].append(v)
        finally:
            lens.train(was_training)

        if self.dist.primary and val_losses:
            metrics = {
                f"val_loss/{k}": th.tensor(v).mean() * state.nats_to_bpb
                for k, v in val_losses.items()
            }
            if self.wandb:
                import wandb

                wandb.log(metrics, step=state.step)
            else:
                metrics_str = ", ".join(
                    f"{k}={float(v):.6f}" for k, v in metrics.items()
                )
                logger.info(f"Validation step {state.step}: {metrics_str}")
