"""Tokenizer and model wrappers for non-HuggingFace models.

As more non-HF protein language models are integrated (E1, ProGen3, ESM3, ...),
add their wrappers here so ingredients.py stays clean.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PretrainedConfig, PreTrainedModel, PreTrainedTokenizer
from transformers.modeling_outputs import MaskedLMOutput


class E1TokenizerWrapper(PreTrainedTokenizer):
    """HF-style tokenizer wrapper for Profluent E1, adapted for tuned-lens.

    Subclasses PreTrainedTokenizer so it passes isinstance checks and HF
    special-token properties (mask_token_id, pad_token_id, etc.) work
    correctly via the standard HF property system.

    Delegates actual tokenization to E1BatchPreparer. The __call__ method
    handles the max_length / truncation / padding contract that
    chunk_and_tokenize expects, and returns all E1-specific input fields
    (sequence_ids, within_seq_position_ids, global_position_ids) alongside
    the standard input_ids / attention_mask.

    Design note: adapted from E1NNSightTokenizer in E1/src/E1/e1_nnsight_adapter.py,
    tailored for the tuned-lens training/eval pipeline instead of NNsight.
    """

    def __init__(self, **kwargs):
        # Lazy import so the base tunedlens venv (without E1) can still import
        # this class without error — E1 is only required when instantiated.
        from E1.batch_preparer import E1BatchPreparer

        self.batch_preparer = E1BatchPreparer(device=torch.device("cpu"))
        vocab = self.batch_preparer.tokenizer.get_vocab()
        self._vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}

        # Wire HF special-token slots; IDs are resolved via _convert_token_to_id.
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("mask_token", "?")
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("model_max_length", 8192)

        super().__init__(**kwargs)

    # --- Vocab ---

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    # --- Special token IDs ---

    @property
    def all_special_ids(self) -> list[int]:
        """IDs that should never be masked during MLM training.

        Includes BOS, EOS, pad, and E1 frame-boundary markers '1' and '2'.
        The mask token '?' is intentionally excluded — it does not appear in
        input protein sequences, so there is no need to protect it.
        """
        return list(
            {
                self.pad_token_id,
                self.bos_token_id,
                self.eos_token_id,
                self._vocab.get("1"),
                self._vocab.get("2"),
            }
            - {None}
        )

    # --- Main call: used by chunk_and_tokenize ---

    def __call__(
        self,
        texts,
        max_length: Optional[int] = None,
        padding=None,
        truncation: bool = False,
        return_attention_mask: bool = False,
        **kwargs,
    ) -> dict:
        if isinstance(texts, str):
            texts = [texts]

        # Each sequence gains 4 boundary tokens: BOS, "1", "2", EOS.
        # Truncate the amino-acid sequence so the total stays within max_length.
        if truncation and max_length is not None:
            max_aa = max_length - 4
            texts = [seq[:max_aa] for seq in texts]

        batch = self.batch_preparer.get_batch_kwargs(texts, device=torch.device("cpu"))
        # Drop fields not needed for training
        batch.pop("context", None)
        batch.pop("context_len", None)
        batch.pop("labels", None)

        # get_batch_kwargs pads to the longest sequence in the current call.
        # If padding="max_length", extend all fields to exactly max_length so
        # that all samples in the dataset have the same shape (required for
        # the dataloader's collate step).
        if padding == "max_length" and max_length is not None:
            current_len = batch["input_ids"].shape[1]
            if current_len < max_length:
                pad_len = max_length - current_len
                pad_values = {
                    "input_ids": self.pad_token_id,
                    "sequence_ids": -1,
                    "within_seq_position_ids": -1,
                    "global_position_ids": -1,
                }
                for key, pad_val in pad_values.items():
                    batch[key] = F.pad(batch[key], (0, pad_len), value=pad_val)

        result = {
            k: v.tolist()
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        if return_attention_mask:
            pad_id = self.pad_token_id
            result["attention_mask"] = [
                [0 if tok == pad_id else 1 for tok in row]
                for row in result["input_ids"]
            ]

        return result

    # --- Required abstract methods (unused: __call__ bypasses the HF pipeline) ---

    def _tokenize(self, text: str) -> list[str]:
        return list(text)  # each amino acid is one token in E1

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, "<unk>")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


class ProGen3TokenizerWrapper(PreTrainedTokenizer):
    """HF-style tokenizer wrapper for Profluent ProGen3, adapted for tuned-lens.

    Subclasses PreTrainedTokenizer so it passes isinstance checks and HF
    special-token properties (pad_token_id, bos_token_id, etc.) work
    correctly via the standard HF property system.

    Delegates actual tokenization to ProGen3BatchPreparer. The __call__ method
    handles the max_length / truncation / padding contract that
    chunk_and_tokenize expects, and returns all ProGen3-specific input fields
    (position_ids, sequence_ids) alongside the standard input_ids.

    ProGen3 uses a BPE tokenizer with empty merges list, so each amino acid
    maps to exactly one token. Boundary tokens are <bos>, '1', '2', <eos>
    (4 total), so max_aa = max_length - 4.
    """

    def __init__(self, **kwargs):
        # Lazy import so the base tunedlens venv (without ProGen3) can still
        # import this class without error — progen3 is only required when
        # instantiated.
        from progen3.batch_preparer import ProGen3BatchPreparer

        self.batch_preparer = ProGen3BatchPreparer()
        vocab = self.batch_preparer.tokenizer.get_vocab()
        self._vocab = vocab
        self._id_to_token = {v: k for k, v in vocab.items()}

        # Wire HF special-token slots; IDs are resolved via _convert_token_to_id.
        kwargs.setdefault("pad_token", "<pad>")
        kwargs.setdefault("bos_token", "<bos>")
        kwargs.setdefault("eos_token", "<eos>")
        kwargs.setdefault("model_max_length", 65536)

        super().__init__(**kwargs)

    # --- Vocab ---

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def get_vocab(self) -> dict:
        return dict(self._vocab)

    # --- Main call: used by chunk_and_tokenize ---

    def __call__(
        self,
        texts,
        max_length: Optional[int] = None,
        padding=None,
        truncation: bool = False,
        return_attention_mask: bool = False,
        **kwargs,
    ) -> dict:
        if isinstance(texts, str):
            texts = [texts]

        # Each sequence gains 4 boundary tokens: <bos>, "1", "2", <eos>.
        # Tokenizer has empty BPE merges so 1 AA = 1 token exactly.
        # Truncate the amino-acid sequence so the total stays within max_length.
        if truncation and max_length is not None:
            max_aa = max_length - 4
            texts = [seq[:max_aa] for seq in texts]

        batch = self.batch_preparer.get_batch_kwargs(texts, device=torch.device("cpu"))
        # Drop labels — not needed for tuned-lens training
        batch.pop("labels", None)

        # get_batch_kwargs pads to the longest sequence in the current call.
        # If padding="max_length", extend all fields to exactly max_length so
        # that all samples in the dataset have the same shape (required for
        # the dataloader's collate step).
        if padding == "max_length" and max_length is not None:
            current_len = batch["input_ids"].shape[1]
            if current_len < max_length:
                pad_len = max_length - current_len
                pad_values = {
                    "input_ids": self.batch_preparer.pad_token_id,
                    "position_ids": 0,
                    "sequence_ids": 0,
                }
                for key, pad_val in pad_values.items():
                    batch[key] = F.pad(batch[key], (0, pad_len), value=pad_val)

        result = {
            k: v.tolist()
            for k, v in batch.items()
            if isinstance(v, torch.Tensor)
        }

        if return_attention_mask:
            pad_id = self.pad_token_id
            result["attention_mask"] = [
                [0 if tok == pad_id else 1 for tok in row]
                for row in result["input_ids"]
            ]

        return result

    # --- Required abstract methods (unused: __call__ bypasses the HF pipeline) ---

    def _tokenize(self, text: str) -> list[str]:
        return list(text)  # each amino acid is one token in ProGen3

    def _convert_token_to_id(self, token: str) -> int:
        return self._vocab.get(token, 0)

    def _convert_id_to_token(self, index: int) -> str:
        return self._id_to_token.get(index, "<unk>")

    def save_vocabulary(self, save_directory, filename_prefix=None):
        return ()


class ESM3Config(PretrainedConfig):
    """Minimal HF-style config for the wrapped ESM3 model (esm3_1.4b).

    Values pinned to the EvolutionaryScale ESM3 `esm.pretrained.ESM3_sm_open_v0`, weights file `esm3_sm_open_v1.pth`, esm3_1.4b aka esm3-open':
        d_model = 1536, 
        n_heads = 24, 
        v_heads = 256, 
        n_layers = 48,
        sequence vocab = 64.
    Review: `tie_word_embeddings=False`: without it, PreTrainedModel's tie_weights() machinery 
    would try to wire input/output embeddings, which doesn't make sense for ESM3.
    """

    model_type = "esm3"

    def __init__(
        self,
        hidden_size: int = 1536,
        num_hidden_layers: int = 48,
        pad_token_id: int = 1,
        tie_word_embeddings: bool = False,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers


class ESM3Wrapper(PreTrainedModel):
    """Adapter so EvolutionaryScale's ESM3 fits tuned-lens's expected interface.

    The upstream `esm.models.esm3.ESM3` is a plain `nn.Module` (not a
    PreTrainedModel) whose `forward()` only accepts `sequence_tokens=...`,
    discards per-layer hidden states inside `TransformerStack.forward`, and
    returns an `ESMOutput` dataclass with `sequence_logits` (not `logits`).

    This wrapper:
      - Subclasses PreTrainedModel so isinstance checks pass and `.base_model`
        resolves to the inner ESM3 via `base_model_prefix`.
      - Returns a `MaskedLMOutput` with HF-style `logits` and `hidden_states`
        attributes (matches what tuned-lens's train_loop / eval_loop read).
      - Captures per-block hidden states via a forward hook on
        `inner.transformer`, prepending the encoder output so the tuple has
        length `num_hidden_layers + 1` (HF convention; tuned-lens does
        `output.hidden_states[:-1]` and iterates against `num_hidden_layers`
        translators).
      - `attention_mask` is silently ignored: ESM3 derives padding internally
        from <pad> ids in `sequence_tokens`.

    Review: NaN handling for unused tracks (structure/ss8/sasa/function/coords) 
    is delegated to upstream ESM3.forward, which fills missing tracks with safe
    defaults; `build_affine3d_from_coordinates` zeros NaN coords before any
    matmul, so passing nothing for the structure side is correct and safe?
    """

    config_class = ESM3Config
    base_model_prefix = "_model"
    supports_gradient_checkpointing = False

    def __init__(self, config: ESM3Config, esm3_model: nn.Module):
        super().__init__(config)
        self._model = esm3_model
        # DO NOT call self.post_init(): the inner ESM3 weights are already loaded; post_init -> init_weights would re-initialise them.

        # Mirror our config onto the inner ESM3 instance so that model_surgery
        # type checks (`_is_esm3_model(base_model)`) work uniformly with how
        # other models are checked. (Upstream ESM3 currently has no `.config` of its own.)
        esm3_model.config = config

    def _init_weights(self, module):
        # Required-ish abstract on PreTrainedModel; never called for our use.
        pass

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_hidden_states: bool = False,
        **_,
    ) -> MaskedLMOutput:
        captured: dict = {}

        def _hook(_module, inputs, output):
            x_in = inputs[0]                      # encoder output = input to block 0
            _, _, block_hiddens = output          # length N (one per block)
            captured["hiddens"] = (x_in, *block_hiddens)   # length N+1, HF-style

        h = self._model.transformer.register_forward_hook(_hook)
        try:
            esm_out = self._model(sequence_tokens=input_ids)
        finally:
            h.remove()

        return MaskedLMOutput(
            logits=esm_out.sequence_logits,
            hidden_states=captured["hiddens"] if output_hidden_states else None,
        )
