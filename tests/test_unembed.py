from typing import Optional

import torch as th
import transformers as tr

from tuned_lens.model_surgery import get_final_norm
from tuned_lens.nn import Unembed


def back_translate(unembed: Unembed, h: th.Tensor, tol: float = 1e-4) -> th.Tensor:
    """Project hidden states into logits and then back into hidden states."""
    scale = h.norm(dim=-1, keepdim=True) / h.shape[-1] ** 0.5
    logits = unembed(h)
    return unembed.invert(logits, h0=th.randn_like(h), tol=tol).preimage * scale


def test_correctness(random_small_model: tr.PreTrainedModel):
    # One problem: we want to check that we handle GPT-J's unembedding bias
    # correctly, but it's zero-initialized. Give it a random Gaussian bias.
    U = random_small_model.get_output_embeddings()
    if U.bias is not None:
        U.bias.data.normal_()

    unembed = Unembed(random_small_model)
    ln_f = get_final_norm(random_small_model)

    x = th.randn(1, 1, random_small_model.config.hidden_size)
    y = U(ln_f(x)).log_softmax(-1)  # type: ignore[attr-defined]

    th.testing.assert_close(y, unembed(x).log_softmax(-1))

    x_hat = back_translate(unembed, x, tol=1e-5)
    th.testing.assert_close(y.exp(), unembed(x_hat).softmax(-1), atol=5e-4, rtol=0.01)


def test_output_embeddings_for_selected_protein_models():
    """Manual probe: load selected protein models and print output embedding modules."""
    model_specs = [
        ("esm2", "facebook/esm2_t6_8M_UR50D", tr.AutoModelForMaskedLM, {}),
        ("prott5", "Rostlab/prot_t5_xl_uniref50", tr.AutoModelForSeq2SeqLM, {}),
        (
            "progen2",
            "hugohrban/progen2-small",
            tr.AutoModelForCausalLM,
            {"trust_remote_code": True},
        ),
        (
            "e1",
            "Profluent-Bio/E1-150m",
            tr.AutoModelForMaskedLM,
            {"trust_remote_code": True},
        ),
        (
            "dplm",
            "airkingbd/dplm_150m",
            tr.AutoModelForMaskedLM,
            {"trust_remote_code": True},
        ),
    ]

    failures: list[str] = []
    for family, model_id, model_cls, extra_kwargs in model_specs:
        model: Optional[tr.PreTrainedModel] = None
        try:
            model = model_cls.from_pretrained(
                model_id,
                low_cpu_mem_usage=True,
                **extra_kwargs,
            )
            out_embed = model.get_output_embeddings()
            print(
                f"[{family}] {model_id} -> output_embeddings={type(out_embed).__name__} "
                f"linear={isinstance(out_embed, th.nn.Linear)}"
            )
        except Exception as exc:
            msg = f"[{family}] {model_id} -> ERROR: {type(exc).__name__}: {exc}"
            print(msg)
            failures.append(msg)
        finally:
            del model

    assert not failures, "Some model probes failed:\n" + "\n".join(failures)
