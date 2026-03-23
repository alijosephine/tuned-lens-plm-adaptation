"""Tools for finding and modifying components in a transformer model."""

from contextlib import contextmanager
from typing import Any, Generator, TypeVar, Union

try:
    import transformer_lens as tl

    _transformer_lens_available = True
except ImportError:
    _transformer_lens_available = False

import torch as th
import transformers as tr
from torch import nn
from transformers import models


def get_value_for_key(obj: Any, key: str) -> Any:
    """Get a value using `__getitem__` if `key` is numeric and `getattr` otherwise."""
    return obj[int(key)] if key.isdigit() else getattr(obj, key)


def set_value_for_key_(obj: Any, key: str, value: Any) -> None:
    """Set value in-place if `key` is numeric and `getattr` otherwise."""
    if key.isdigit():
        obj[int(key)] = value
    else:
        setattr(obj, key, value)


def get_key_path(model: th.nn.Module, key_path: str) -> Any:
    """Get a value by key path, e.g. `layers.0.attention.query.weight`."""
    for key in key_path.split("."):
        model = get_value_for_key(model, key)

    return model


def set_key_path_(
    model: th.nn.Module, key_path: str, value: Union[th.nn.Module, th.Tensor]
) -> None:
    """Set a value by key path in-place, e.g. `layers.0.attention.query.weight`."""
    keys = key_path.split(".")
    for key in keys[:-1]:
        model = get_value_for_key(model, key)

    setattr(model, keys[-1], value)


T = TypeVar("T", bound=th.nn.Module)


@contextmanager
def assign_key_path(model: T, key_path: str, value: Any) -> Generator[T, None, None]:
    """Temporarily set a value by key path while in the context."""
    old_value = get_key_path(model, key_path)
    set_key_path_(model, key_path, value)
    try:
        yield model
    finally:
        set_key_path_(model, key_path, old_value)


Model = Union[tr.PreTrainedModel, "tl.HookedTransformer"]
Norm = Union[
    th.nn.LayerNorm,
    models.llama.modeling_llama.LlamaRMSNorm,
    models.gemma.modeling_gemma.GemmaRMSNorm,
    nn.Module,
]


def get_model_hidden_size_and_num_layers(model: tr.PreTrainedModel) -> tuple[int, int]:
    """Infer model hidden size and layer count from config fields."""
    d_model = getattr(model.config, "hidden_size", None)
    if d_model is None:
        d_model = getattr(model.config, "n_embd", None)
    if d_model is None:
        d_model = getattr(model.config, "d_model", None)
    if d_model is None:
        raise ValueError(
            "Could not infer hidden size from model config. Expected one of "
            "'hidden_size', 'n_embd', or 'd_model'."
        )

    num_hidden_layers = getattr(model.config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        num_hidden_layers = getattr(model.config, "n_layer", None)
    if num_hidden_layers is None:
        num_hidden_layers = getattr(model.config, "num_layers", None)
    if num_hidden_layers is None:
        raise ValueError(
            "Could not infer number of layers from model config. Expected one of "
            "'num_hidden_layers', 'n_layer', or 'num_layers'."
        )

    return int(d_model), int(num_hidden_layers)


# refer to run/example_configs_and_architectures folder for model types!!
def _model_type(obj: Any) -> Union[str, None]:
    """Return `config.model_type` when available."""
    config = getattr(obj, "config", None)
    model_type = getattr(config, "model_type", None)
    return model_type if isinstance(model_type, str) else None


def _is_progen_model(obj: Any) -> bool:
    """Return True for ProGen2-family models."""
    return _model_type(obj) == "progen"


def _is_esm_model(obj: Any) -> bool:
    """Return True for ESM/DPLM-family models."""
    return _model_type(obj) == "esm"


def _is_t5_model(obj: Any) -> bool:
    """Return True for T5/ProtT5-family models."""
    return _model_type(obj) == "t5"


def _is_profluent_e1_model(obj: Any) -> bool:
    """Return True for Profluent E1-family models."""
    return _model_type(obj) == "E1"


def get_unembedding_matrix(model: Model) -> nn.Module:
    """The final transformation from the model hidden state to the output logits.

    For models with a simple linear head this returns an ``nn.Linear``.  For
    models whose head is a composite module (e.g. ESM2's ``EsmLMHead`` which
    contains ``dense → layer_norm → decoder``), the full module is returned so
    that the complete unembedding path is used when computing distributions.
    """
    if isinstance(model, tr.PreTrainedModel):
        # ESM/DPLM: lm_head is EsmLMHead (dense + layer_norm + decoder).
        # get_output_embeddings() only returns the inner decoder Linear, losing
        # the dense projection and layer norm.  Return the full lm_head instead.
        if _is_esm_model(model):
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and isinstance(lm_head, nn.Module):
                return lm_head
            raise ValueError("error in extracting lm_head from ESM model")

        # T5/ProtT5: lm_head is a plain Linear(d_model, vocab, bias=False).
        if _is_t5_model(model):
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and isinstance(lm_head, nn.Linear):
                return lm_head
            raise ValueError(
                f"error in extracting lm_head from T5 model"
            )

        # E1: mlm_head is Sequential(Linear → GELU → LayerNorm → Linear).
        if _is_profluent_e1_model(model):
            mlm_head = getattr(model, "mlm_head", None)
            if mlm_head is not None and isinstance(mlm_head, nn.Sequential):
                return mlm_head
            raise ValueError(
                f"error in extracting lm_head from E1 model"
            )

        # ProGen2: lm_head is a plain Linear(d_model, vocab, bias=True).
        if _is_progen_model(model):
            lm_head = getattr(model, "lm_head", None)
            if lm_head is not None and isinstance(lm_head, nn.Linear):
                return lm_head
            raise ValueError(
                f"error in extracting lm_head from ProGen2 model"
            )

        # Default path for all standard HuggingFace causal/masked LMs
        # (GPT-2, LLaMA, Mistral, Gemma, OPT, Bloom, GPT-NeoX, …):
        # get_output_embeddings() returns the lm_head directly as nn.Linear.
        unembed = None
        try:
            unembed = model.get_output_embeddings()
        except Exception:
            pass
        if isinstance(unembed, nn.Linear):
            return unembed

        raise ValueError(
            f"Could not find a supported unembedding module for model type "
            f"'{_model_type(model)}' (got {type(unembed)} from get_output_embeddings)."
        )
    elif _transformer_lens_available and isinstance(model, tl.HookedTransformer):
        linear = nn.Linear(
            in_features=model.cfg.d_model,
            out_features=model.cfg.d_vocab_out,
        )
        linear.bias.data = model.unembed.b_U
        linear.weight.data = model.unembed.W_U.transpose(0, 1)
        return linear
    else:
        raise ValueError(f"Model class {type(model)} not recognized!")


def get_final_norm(model: Model) -> Norm:
    """Get the final norm from a model.

    This isn't standardized across models, so this will need to be updated as
    we add new models.
    """
    if _transformer_lens_available and isinstance(model, tl.HookedTransformer):
        return model.ln_final

    if not hasattr(model, "base_model"):
        raise ValueError("Model does not have a `base_model` attribute.")

    base_model = model.base_model
    if _is_t5_model(base_model):
        final_layer_norm = base_model.encoder.final_layer_norm
    elif _is_profluent_e1_model(base_model):
        final_layer_norm = base_model.norm
    elif _is_esm_model(base_model): # should cover dplm as well!
        final_layer_norm = base_model.encoder.emb_layer_norm_after
    elif _is_progen_model(base_model):
        final_layer_norm = base_model.ln_f
    elif isinstance(base_model, models.opt.modeling_opt.OPTModel):
        final_layer_norm = base_model.decoder.final_layer_norm
    elif isinstance(base_model, models.gpt_neox.modeling_gpt_neox.GPTNeoXModel):
        final_layer_norm = base_model.final_layer_norm
    elif isinstance(
        base_model,
        (
            models.bloom.modeling_bloom.BloomModel,
            models.gpt2.modeling_gpt2.GPT2Model,
            models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
            models.gptj.modeling_gptj.GPTJModel,
        ),
    ):
        final_layer_norm = base_model.ln_f
    elif isinstance(base_model, models.llama.modeling_llama.LlamaModel):
        final_layer_norm = base_model.norm
    elif isinstance(base_model, models.mistral.modeling_mistral.MistralModel):
        final_layer_norm = base_model.norm
    elif isinstance(base_model, models.gemma.modeling_gemma.GemmaModel):
        final_layer_norm = base_model.norm
    else:
        raise NotImplementedError(f"Unknown model type {type(base_model)}")

    if final_layer_norm is None:
        raise ValueError("Model does not have a final layer norm.")

    assert isinstance(final_layer_norm, Norm.__args__)  # type: ignore

    return final_layer_norm


def get_transformer_layers(model: Model) -> tuple[str, th.nn.ModuleList]:
    """Get the decoder layers from a model.

    Args:
        model: The model to search.

    Returns:
        A tuple containing the key path to the layer list and the list itself.

    Raises:
        ValueError: If no such list exists.
    """
    if not hasattr(model, "base_model"):
        raise ValueError("Model does not have a `base_model` attribute.")

    base_model = model.base_model
    if _is_t5_model(base_model):
        path_to_layers = "base_model.encoder.block"
        layer_list = base_model.encoder.block
    elif _is_profluent_e1_model(base_model):
        path_to_layers = "base_model.layers"
        layer_list = base_model.layers
    elif _is_esm_model(base_model):
        path_to_layers = "base_model.encoder.layer"
        layer_list = base_model.encoder.layer
    elif _is_progen_model(base_model):
        path_to_layers = "base_model.h"
        layer_list = base_model.h
    elif isinstance(base_model, models.opt.modeling_opt.OPTModel):
        path_to_layers = "base_model.decoder.layers"
        layer_list = base_model.decoder.layers
    elif isinstance(base_model, models.gpt_neox.modeling_gpt_neox.GPTNeoXModel):
        path_to_layers = "base_model.layers"
        layer_list = base_model.layers
    elif isinstance(
        base_model,
        (
            models.bloom.modeling_bloom.BloomModel,
            models.gpt2.modeling_gpt2.GPT2Model,
            models.gpt_neo.modeling_gpt_neo.GPTNeoModel,
            models.gptj.modeling_gptj.GPTJModel,
        ),
    ):
        path_to_layers = "base_model.h"
        layer_list = base_model.h
    elif isinstance(base_model, models.llama.modeling_llama.LlamaModel):
        path_to_layers = "base_model.layers"
        layer_list = base_model.layers
    elif isinstance(base_model, models.mistral.modeling_mistral.MistralModel):
        path_to_layers = "base_model.layers"
        layer_list = base_model.layers
    elif isinstance(base_model, models.gemma.modeling_gemma.GemmaModel):
        path_to_layers = "base_model.layers"
        layer_list = base_model.layers
    else:
        raise NotImplementedError(f"Unknown model type {type(base_model)}")

    if not isinstance(layer_list, th.nn.ModuleList): # TODO: is this check required? revisit!
        raise ValueError(f"Expected ModuleList at '{path_to_layers}', got {type(layer_list)}")
    return path_to_layers, layer_list


@contextmanager
def delete_layers(model: T, indices: list[int]) -> Generator[T, None, None]:
    """Temporarily delete the layers at `indices` from `model` while in the context."""
    list_path, layer_list = get_transformer_layers(model)
    modified_list = th.nn.ModuleList(layer_list)
    for i in sorted(indices, reverse=True):
        del modified_list[i]

    set_key_path_(model, list_path, modified_list)
    try:
        yield model
    finally:
        set_key_path_(model, list_path, layer_list)


@contextmanager
def permute_layers(model: T, indices: list[int]) -> Generator[T, None, None]:
    """Temporarily permute the layers of `model` by `indices` while in the context.

    The number of indices provided may be not be equal to the number of
    layers in the model. Layers will be dropped or duplicated accordingly.
    """
    list_path, layer_list = get_transformer_layers(model)
    permuted_list = th.nn.ModuleList([layer_list[i] for i in indices])
    set_key_path_(model, list_path, permuted_list)

    try:
        yield model
    finally:
        set_key_path_(model, list_path, layer_list)


def permute_layers_(model: th.nn.Module, indices: list[int]):
    """Permute the layers of `model` by `indices` in-place.

    The number of indices provided may be not be equal to the number of
    layers in the model. Layers will be dropped or duplicated accordingly.
    """
    list_path, layer_list = get_transformer_layers(model)
    permuted_list = th.nn.ModuleList([layer_list[i] for i in indices])
    set_key_path_(model, list_path, permuted_list)


@contextmanager
def replace_layers(
    model: T, indices: list[int], replacements: list[th.nn.Module]
) -> Generator[T, None, None]:
    """Replace the layers at `indices` with `replacements` while in the context."""
    list_path, layer_list = get_transformer_layers(model)
    modified_list = th.nn.ModuleList(layer_list)
    for i, replacement in zip(indices, replacements):
        modified_list[i] = replacement

    set_key_path_(model, list_path, modified_list)
    try:
        yield model
    finally:
        set_key_path_(model, list_path, layer_list)
