import pickle
#from tuned_lens.model_wrappers import E1TokenizerWrapper
from tuned_lens.model_wrappers import ProGen3TokenizerWrapper

#for cls in [E1TokenizerWrapper]:
for cls in [ProGen3TokenizerWrapper]:
    tok = cls()
    blob = pickle.dumps(tok)
    tok2 = pickle.loads(blob)
    out = tok2(["MKTAYIAKQRQISFVKSHFSRQLEERLG"], max_length=64,
               padding="max_length", truncation=True)
    assert "input_ids" in out
    print(f"{cls.__name__}: pickle OK ({len(blob)} bytes), forward OK")


# NOTE: the Profluent models (E1, ProGen3) deadlock issue.
#
# Symptom: rank 0 hangs at `Map (num_proc=8): 0%` inside `Dataset.map`; ranks
# 1-3 trip `dist.barrier()`'s 1200 s timeout in train_loop.setup() and the job
# fails with `torch.distributed.DistBackendError: wait timeout`. ESM2/ESM3 on
# the same code path tokenize cleanly in ~30 s, so the issue is specific to
# the E1/ProGen3 wrappers' import chain (Rust `tokenizers.Tokenizer` + heavy
# local packages → fork-after-CUDA / fork-after-rust-thread-pool deadlock).
#
# Tried and didn't work:
#   - num_proc=1 in Dataset.map: avoids fork but rank 0 takes >20 min on the
#     real dataset and still trips the barrier timeout.
#   - multiprocessing_context="spawn" kwarg on Dataset.map: not a public API
#     on any released `datasets` version (hf/transformers#34793) → TypeError.
#   - Relying on `datasets`'s built-in TOKENIZERS_PARALLELISM=false auto-set:
#     mitigates only half of the deadlock; rank 0 still hangs.
#
# Current fix (tuned_lens/__main__.py:_maybe_force_spawn_for_profluent_loaders):
# at process start, if `--model_loader` is `e1` or `progen3`, flip the start
# method to "spawn" on BOTH stdlib `multiprocessing` AND the `multiprocess`
# library (datasets uses the latter, they have separate global state). Must
# run before any torch import. Other model families keep fork.
#
# Status: shim landed but not yet validated end-to-end on E1/ProGen3 warmups
# (pending sbatch rerun). Watch the new .err for either:
#   - `Map (num_proc=8): X%` with steady progress  → fix worked;
#   - `PicklingError` from dill within seconds      → tokenize_fn closure not
#     spawn-pickleable; refactor needed (see model_wrappers / tokenizer);
#   - silent hang at 0% for >2 min                  → root cause isn't fork,
#     fall back to rank-0-only tokenization then barrier (option D).
