# script to test integration of esm3 into tunedlens
# for now, considering only the sequence stream, i.e., structure tokens ignored!
#
# Standalone smoke test (NOT a pytest case): downloads / loads real ESM3 weights,
# so run it manually inside the esm3 conda env, e.g.:
#     python tests/test_esm3.py
#
# Exercises the full chain that tuned-lens relies on:
#   1. Model.load() returns a PreTrainedModel + tokenizer.
#   2. model_surgery can locate unembedding, final norm, and transformer layers.
#   3. forward(input_ids, output_hidden_states=True) returns HF-style outputs
#      with N+1 hidden states.
#   4. TunedLens.from_model() builds without error and the lens forward shapes
#      are consistent end-to-end.

from __future__ import annotations

import torch as th
import transformers as tr

from tuned_lens import TunedLens
from tuned_lens import model_surgery as ms
from tuned_lens.model_wrappers import ESM3Wrapper
from tuned_lens.scripts.ingredients import Model


EXPECTED_HIDDEN = 1536
EXPECTED_LAYERS = 48
EXPECTED_VOCAB = 64


def main() -> None:
    device = th.device("cuda" if th.cuda.is_available() else "cpu")
    print(f"[setup] device = {device}")

    # 1. Load via the same code path the training script will use.
    model, tokenizer = Model(
        name="esm3_sm_open_v0",
        model_loader="esm3",
        model_type="masked",
    ).load(device=device)

    assert isinstance(model, tr.PreTrainedModel), type(model)
    assert isinstance(model, ESM3Wrapper), type(model)
    assert isinstance(tokenizer, tr.PreTrainedTokenizerBase), type(tokenizer)

    cfg = model.config
    assert cfg.model_type == "esm3"
    assert cfg.hidden_size == EXPECTED_HIDDEN
    assert cfg.num_hidden_layers == EXPECTED_LAYERS
    assert cfg.name_or_path == "esm3_sm_open_v0"
    # Mirror onto inner instance, used by model_surgery base_model checks.
    assert model.base_model is model._model
    assert getattr(model.base_model, "config", None) is cfg
    print("[1/4] load + config mirror OK")

    # 2. model_surgery resolution.
    unembed = ms.get_unembedding_matrix(model)
    final_norm = ms.get_final_norm(model)
    layer_path, layers = ms.get_transformer_layers(model)

    assert isinstance(unembed, th.nn.Sequential), type(unembed)
    # RegressionHead = Linear -> GELU -> LayerNorm -> Linear; final Linear gives vocab.
    last_linear = [m for m in unembed if isinstance(m, th.nn.Linear)][-1]
    assert last_linear.out_features == EXPECTED_VOCAB, last_linear.out_features

    assert layer_path == "base_model.transformer.blocks", layer_path
    assert len(layers) == EXPECTED_LAYERS, len(layers)
    assert final_norm is not None
    print(
        f"[2/4] surgery OK  unembed={type(unembed).__name__} "
        f"norm={type(final_norm).__name__} #layers={len(layers)}"
    )

    # 3. Forward with a tiny batch.
    seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQ"
    enc = tokenizer(seq, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]

    with th.no_grad():
        out = model(input_ids=input_ids, output_hidden_states=True)

    assert hasattr(out, "logits") and hasattr(out, "hidden_states")
    L = input_ids.shape[1]
    assert out.logits.shape == (1, L, EXPECTED_VOCAB), out.logits.shape
    assert isinstance(out.hidden_states, tuple)
    assert len(out.hidden_states) == EXPECTED_LAYERS + 1, len(out.hidden_states)
    for i, h in enumerate(out.hidden_states):
        assert h.shape == (1, L, EXPECTED_HIDDEN), (i, h.shape)
    # Sanity: hidden_states should be float32 (we cast inner to f32 in ingredients).
    assert out.hidden_states[0].dtype == th.float32, out.hidden_states[0].dtype
    print(
        f"[3/4] forward OK  logits={tuple(out.logits.shape)} "
        f"hiddens={len(out.hidden_states)}x{tuple(out.hidden_states[0].shape)}"
    )

    # 4. TunedLens construction + a forward through one translator.
    lens = TunedLens.from_model(model).to(device)
    assert len(lens.layer_translators) == EXPECTED_LAYERS, len(lens.layer_translators)

    h_mid = out.hidden_states[EXPECTED_LAYERS // 2]
    with th.no_grad():
        lens_logits = lens(h_mid, idx=EXPECTED_LAYERS // 2)
    assert lens_logits.shape == (1, L, EXPECTED_VOCAB), lens_logits.shape
    print(
        f"[4/4] TunedLens.from_model OK  translators={len(lens.layer_translators)} "
        f"lens_logits={tuple(lens_logits.shape)}"
    )

    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()