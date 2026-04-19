"""Script to train or evaluate a set of tuned lenses for a language model."""

import os
import sys

# TODO: hacky fix to solve the Profluent models (E1, ProGen3) deadlock issue in data.py.
def _maybe_force_spawn_for_profluent_loaders() -> None:
    """Force the multiprocessing start method to "spawn" before any heavy import.

    Profluent's E1 and ProGen3 wrappers pull in local packages and a Rust
    `tokenizers.Tokenizer`. When `datasets.Dataset.map(num_proc=8)` later forks
    workers (the library's default), those forked children deadlock silently
    on rank 0 — every other rank then trips `dist.barrier()`'s 1200 s timeout.
    `datasets`'s built-in `TOKENIZERS_PARALLELISM=false` only handles half of
    the deadlock; switching to "spawn" gives each worker a fresh interpreter
    with no inherited CUDA / Rust state, which empirically clears the hang.

    Has to run BEFORE we import torch (and therefore before the rest of this
    module's imports), otherwise CUDA / threading state has already been
    initialised in the parent and `set_start_method` is too late.

    Gated on `--model_loader {e1, progen3}` so the other model families keep
    using fork (spawn would force every worker to re-import the world).
    """
    try:
        i = sys.argv.index("--model_loader")
    except ValueError:
        return
    if i + 1 >= len(sys.argv) or sys.argv[i + 1] not in {"e1", "progen3"}:
        return
    # IMPORTANT: HuggingFace `datasets` uses `multiprocess` (the dill-pickling
    # Pathos fork), NOT stdlib `multiprocessing`. They have separate global
    # state, so we have to flip the start method on BOTH for `Dataset.map`'s
    # internal `mp.Pool(num_proc)` to actually use spawn. Stdlib `multiprocessing`
    # is also flipped for safety (e.g. torch's DataLoader workers).
    import multiprocessing
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    try:
        import multiprocess
        multiprocess.set_start_method("spawn", force=True)
    except (ImportError, RuntimeError):
        pass


_maybe_force_spawn_for_profluent_loaders()


import logging  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from typing import Literal, Optional, Union  # noqa: E402

from simple_parsing import ArgumentParser, ConflictResolution  # noqa: E402
from torch.distributed.elastic.multiprocessing.errors import record  # noqa: E402

from .scripts.eval_loop import Eval  # noqa: E402
from .scripts.train_loop import Train  # noqa: E402


@dataclass
class Main:
    """Routes to the subcommands."""

    command: Union[Train, Eval]

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    """The log level to use."""

    def execute(self):
        """Run the script."""
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            FORMAT = f"[%(levelname)s] rank={local_rank} %(message)s"
        else:
            FORMAT = "[%(levelname)s] %(message)s"

        logging.basicConfig(level=self.log_level, format=FORMAT)
        self.command.execute()


@record
def main(args: Optional[list[str]] = None):
    """Entry point for the CLI."""
    parser = ArgumentParser(conflict_resolution=ConflictResolution.EXPLICIT)
    parser.add_arguments(Main, dest="prog")
    args = parser.parse_args(args=args)
    prog: Main = args.prog
    prog.execute()


if __name__ == "__main__":
    main()
