"""Shared configuration for the scripts."""
import enum
import logging
import os
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from typing import Optional, Union

import torch as th
import torch.distributed as dist
from datasets import Dataset, DatasetDict, load_dataset
from simple_parsing import field
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.optim import ZeroRedundancyOptimizer
from torch.nn.parallel import DistributedDataParallel as DDP
from torchdata import dataloader2, datapipes
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    T5Tokenizer,
    get_linear_schedule_with_warmup,
)
from typing_extensions import Literal

from tuned_lens.data import (
    chunk_and_tokenize,
    dataset_from_fasta,
)
from tuned_lens.model_surgery import get_transformer_layers
from tuned_lens.model_wrappers import (
    E1TokenizerWrapper,
    ESM3Config,
    ESM3Wrapper,
    ProGen3TokenizerWrapper,
)
from tuned_lens.nn.lenses import Lens
from tuned_lens.utils import (
    TreeType,
    handle_name_conflicts,
    send_to_device,
)

logger = logging.getLogger(__name__)


@dataclass
class Data:
    """Configuration for the dataset."""

    name: list[str] = field(default_factory=lambda: ["the_pile", "all"], nargs="*")
    """Name of dataset to use. Can either be a local .jsonl file or a name
    suitable to be passed to the HuggingFace load_dataset function."""

    split: str = "validation"
    """Split of the dataset to use."""

    text_column: str = "text"
    """Column of the dataset containing text to run the model on."""

    revision: Optional[str] = None
    """The revision of the dataset to use"""

    max_seq_len: int = 2048
    """The maximum length of the input sequences."""

    dataset_shuffle: bool = False
    """Whether to shuffle the dataset prior to tokenization."""

    dataset_shuffle_seed: int = 42
    """Seed to use for shuffling the dataset"""

    def load(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model_type: Literal["causal", "masked", "encoder-decoder"] = "causal",
    ) -> tuple[Dataset, float]:
        """Load the dataset, tokenize it and compute nats_to_bpb."""
        logger.info(f"Loading dataset '{' '.join(self.name)}'")
        logger.debug(f"Using split '{self.split}', revision '{self.revision}'")

        if len(self.name) == 1 and self.name[0].lower().endswith(".fasta"):
            logger.info(f"Loading local FASTA file '{self.name[0]}'")
            dataset = dataset_from_fasta(self.name[0], text_key=self.text_column)
        elif len(self.name) == 1 and self.name[0].endswith(".jsonl"):
            dataset = Dataset.from_json(self.name[0])
            assert isinstance(dataset, Dataset)
        else:
            dataset = load_dataset(*self.name, split=self.split, revision=self.revision)
            if not isinstance(dataset, (Dataset, DatasetDict)):
                raise ValueError(
                    "Only Dataset and DatasetDict instances are supported."
                )

        logger.debug(f"Dataset has {len(dataset)} samples.")
        logger.debug(f"Dataset columns: {dataset.column_names}")

        if self.dataset_shuffle:
            logger.debug(f"Shuffling dataset with seed: {self.dataset_shuffle_seed}")
            dataset = dataset.shuffle(self.dataset_shuffle_seed)

        logger.debug("Beginning tokenization...")
        processed, nats_to_bpb = chunk_and_tokenize(
            dataset,
            tokenizer,
            model_type=model_type,
            text_key=self.text_column,
            max_seq_len=self.max_seq_len,
        )

        logger.info(f"Using nats per token to bits per byte ratio: {nats_to_bpb}")

        assert isinstance(processed, Dataset)

        return processed, nats_to_bpb

@dataclass
class Model:
    """Configuration for the model and tokenizer."""

    name: str
    """Name of model to use in the Huggingface Hub."""

    precision: Literal["auto", "bfloat16", "float16", "float32", "int8"] = "auto"
    """Precision in which to load the model weights."""

    revision: str = "main"
    """Git revision to use for pretrained models."""

    slow_tokenizer: bool = field(action="store_true")
    """Use a slow tokenizer."""

    tokenizer: Optional[str] = None
    """Name of pretrained tokenizer to use from the Huggingface Hub. If None, will use
    AutoTokenizer.from_pretrained('<model name>')."""

    tokenizer_type: Optional[str] = None
    """Name of tokenizer class to use. If None, will use AutoTokenizer."""

    trust_remote_code: bool = field(action="store_true")
    """Allow loading model/tokenizer repos that define custom Python code."""

    model_type: Literal["causal", "masked", "encoder-decoder"] = "causal"
    """Model head type to load.

    - "causal": use AutoModelForCausalLM.
    - "masked": use AutoModelForMaskedLM.
    - "encoder-decoder": use AutoModelForSeq2SeqLM (e.g. ProtT5).
    """

    model_loader: Literal["huggingface", "e1", "progen3", "esm3"] = "huggingface"
    """Which loading backend to use.

    - "huggingface": standard HuggingFace Auto* classes (default).
    - "e1": Profluent E1 local installation.
    - "progen3": Profluent ProGen3 local installation.
    - "esm3": EvolutionaryScale ESM3 (sequence stream only).
    """

    def load_tokenizer(self, must_use_cache: bool = False):
        """Load the tokenizer.

        Returns a PreTrainedTokenizerBase for HuggingFace models, or an
        E1TokenizerWrapper for E1 (see tuned_lens/model_wrappers.py).
        """
        if self.model_loader == "e1":
            return E1TokenizerWrapper()

        if self.model_loader == "progen3":
            return ProGen3TokenizerWrapper()

        if self.model_loader == "esm3":
            # EsmSequenceTokenizer is already a PreTrainedTokenizerFast — no wrapper needed.
            from esm.tokenization import EsmSequenceTokenizer
            return EsmSequenceTokenizer()

        with handle_name_conflicts():
            # T5TokenizerFast in newer transformers incorrectly tries to parse spiece.model
            # as a tiktoken file. Use T5Tokenizer (slow/SentencePiece) directly for T5 models.
            if self.model_type == "encoder-decoder":
                tokenizer = T5Tokenizer.from_pretrained(
                    self.tokenizer or self.name,
                    revision=self.revision,
                    local_files_only=must_use_cache,
                )
            else:
                tokenizer = AutoTokenizer.from_pretrained(
                    self.tokenizer or self.name,
                    revision=self.revision,
                    use_fast=not self.slow_tokenizer,
                    tokenizer_type=self.tokenizer_type,
                    local_files_only=must_use_cache,
                    trust_remote_code=self.trust_remote_code,
                )

        assert isinstance(tokenizer, PreTrainedTokenizerBase)

        # ProtT5 tokenizers do not define `mask_token` by default, but expose
        # sentinel extra ids. Reuse <extra_id_0> for MLM masking.
        if (
            self.model_type == "encoder-decoder"
            and tokenizer.mask_token_id is None
        ):
            sentinel = "<extra_id_0>"
            sentinel_id = tokenizer.convert_tokens_to_ids(sentinel)
            unk_id = getattr(tokenizer, "unk_token_id", None)
            if isinstance(sentinel_id, int) and (unk_id is None or sentinel_id != unk_id):
                tokenizer.mask_token = sentinel

        return tokenizer

    def load(
        self, device: Optional[th.device], must_use_cache: bool = False
    ) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
        """Load the model and tokenizer.

        Args:
            device: The device to load the model on. Implemented with the `device_map`
                argument of `from_pretrained`.
            must_use_cache: If True, will raise an error if the model is not cached.
        """
        logger.info(f"Loading pretrained weights for '{self.name}'...")
        logger.debug(
            (
                "Using revision {revision} dtype {dtype}, and device {device}, "
                "trust_remote_code={trust_remote_code}, model_type={model_type}"
            ).format(
                revision=self.revision,
                dtype=self.precision,
                device=device,
                trust_remote_code=self.trust_remote_code,
                model_type=self.model_type,
            )
        )

        try:
            dtype = {
                "auto": "auto",
                "bfloat16": th.bfloat16,
                "float16": th.float16,
                "float32": th.float32,
                # `bitsandbytes` requires weights to initially be in fp16
                "int8": th.float16,
            }[self.precision]
        except KeyError as e:
            raise ValueError(f"Unknown precision: {self.precision}") from e

        if self.model_loader == "e1":
            from E1.modeling import E1ForMaskedLM
            with handle_name_conflicts():
                model = E1ForMaskedLM.from_pretrained(
                    self.name,
                    device_map={"": device} if device is not None else None,
                    low_cpu_mem_usage=True,
                    torch_dtype=dtype,
                    local_files_only=must_use_cache,
                )
            logger.info("Loaded E1 model via local E1 package.")
        elif self.model_loader == "progen3":
            from progen3.modeling import ProGen3ForCausalLM
            with handle_name_conflicts():
                model = ProGen3ForCausalLM.from_pretrained(
                    self.name,
                    device_map={"": device} if device is not None else None,
                    low_cpu_mem_usage=True,
                    torch_dtype=dtype,
                    local_files_only=must_use_cache,
                )
            logger.info("Loaded ProGen3 model via local progen3 package.")
        elif self.model_loader == "esm3":
            # ESM3 has no HuggingFace integration; load via the upstream
            # `esm.pretrained` factory and wrap in our PreTrainedModel adapter.
            from esm.pretrained import ESM3_sm_open_v0
            inner = ESM3_sm_open_v0(device=device or th.device("cpu"))
            # Cast to float32 unless caller explicitly chose another precision.
            # tuned-lens probes are f32; ESM3 defaults to bfloat16 on GPU which
            # would cause dtype mismatch in the lens forward. Empirically
            # validated by depth_analysis (their bfloat16 cast is commented out).
            inner = inner.to(th.float32 if self.precision == "auto" else dtype)

            # pad_token_id default in ESM3Config (1) matches EsmSequenceTokenizer.
            config = ESM3Config(name_or_path=self.name)
            model = ESM3Wrapper(config, esm3_model=inner)
            logger.info("Loaded ESM3 (sequence-only) via local esm package.")
        else: #default: huggingface
            if self.model_type == "causal":
                auto_model_cls = AutoModelForCausalLM
            elif self.model_type == "masked":
                auto_model_cls = AutoModelForMaskedLM
            elif self.model_type == "encoder-decoder":
                auto_model_cls = AutoModelForSeq2SeqLM
            else:
                raise ValueError(f"Unknown model_type: {self.model_type}")

            with handle_name_conflicts():
                model = auto_model_cls.from_pretrained(  # type: ignore
                    self.name,
                    device_map={"": device} if device is not None else None,
                    load_in_8bit=self.precision == "int8",
                    low_cpu_mem_usage=True,
                    revision=self.revision,
                    torch_dtype=dtype,
                    local_files_only=must_use_cache,
                    trust_remote_code=self.trust_remote_code,
                )
            logger.info(f"Loaded model using '{self.model_type}' head.")

        assert isinstance(model, PreTrainedModel)
        model.eval()
        model.requires_grad_(False)

        return model, self.load_tokenizer(must_use_cache=must_use_cache)


class OptimizerOption(enum.Enum):
    """Options for the optimizer to use when training the model."""

    ADAM = "adam"
    SGD = "sgd"


@dataclass
class Optimizer:
    """Configuration for the optimizer."""

    weight_decay: float = 1e-3
    """Weight decay coefficient."""

    lr_scale: float = 1.0
    """The default LR (1e-3 for Adam, 1.0 for SGD) is scaled by this factor."""

    momentum: float = 0.9
    """Momentum coefficient for SGD, or beta1 for Adam."""

    zero: Optional[bool] = field(action="store_true")
    """Use ZeroRedundancyOptimizer."""

    optimizer: OptimizerOption = OptimizerOption.SGD
    """The type of optimizer to use."""

    warmup_steps: Optional[int] = None
    """Number of warmup steps. Defaults to min(0.2 * num_steps, 1000) for Adam and 0
    for SGD."""

    def create_scheduler(
        self, opt: th.optim.Optimizer, num_steps: int
    ) -> th.optim.lr_scheduler.LambdaLR:
        """Create the LR scheduler."""
        if self.warmup_steps is None:
            # Adam generally performs poorly without an LR warmup
            if self.optimizer == "adam":
                self.warmup_steps = min(1000, num_steps // 5)
                logger.info(f"Using {self.warmup_steps} LR warmup steps for Adam")
            else:
                self.warmup_steps = 0

        scheduler = get_linear_schedule_with_warmup(
            opt, self.warmup_steps, num_steps - self.warmup_steps
        )

        return scheduler

    def create_optim(self, params: list[th.nn.Parameter]) -> th.optim.Optimizer:
        """Create the optimizer."""
        # Don't train things that don't need gradients
        β = self.momentum
        if self.optimizer == OptimizerOption.SGD:
            config = dict(
                # PyTorch's implementation effectively scales the LR by 1 / (1 - β),
                # so we undo that here. See https://www.youtube.com/watch?v=k8fTYJPd3_I
                # for discussion. Once we do this, the optimal LR seems to be unity.
                lr=self.lr_scale * (1 - β),
                momentum=β,
                # Empirically Nesterov momentum seems to improve convergence speed.
                nesterov=True,
                weight_decay=self.weight_decay,
            )
            opt_class = th.optim.SGD
        elif self.optimizer == OptimizerOption.ADAM:
            config = dict(
                # Helps convergence slightly by ensuring that the LR actually decays
                amsgrad=True,
                betas=(β, 0.999),
                lr=self.lr_scale * 1e-3,
                weight_decay=self.weight_decay,
            )
            opt_class = th.optim.Adam
        else:
            raise ValueError(f"Unknown optimizer '{self.optimizer}'")

        if self.zero:
            opt = ZeroRedundancyOptimizer(params, optimizer_class=opt_class, **config)
        else:
            opt = opt_class(params, **config)  # type: ignore[call-arg]

        return opt

    def per_parameter_optim_state_size(self) -> int:
        """The number of elements in the optimizer state per parameter."""
        return 2 if self.optimizer == OptimizerOption.ADAM else 1


@dataclass
class Distributed:
    """Configuration and utilities for distributing the model."""

    fsdp: bool = field(action="store_true")
    """Run the model with Fully Sharded Data Parallelism."""

    cpu_offload: bool = field(action="store_true")
    """Use CPU offloading. Must be combined with fsdp"""

    nccl_timeout: int = 1200  # 20 minutes
    """Timeout for NCCL operations in seconds."""

    per_gpu_batch_size: int = 1
    """The batch size per GPU."""

    dataloader_shuffle: bool = True
    """Whether to shuffle the batches of tokenized data as they are loaded."""

    @property
    def rank(self) -> int:
        """The rank of this process.

        Note that in general this is not the same as the local rank.
        However, for single-node training, the local rank is the same as the
        global rank.
        """
        return int(os.environ["RANK"]) if dist.is_initialized() else 0

    @property
    def local_rank(self) -> int:
        """The local rank of this process."""
        return int(os.environ["LOCAL_RANK"]) if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        """Get the world size from torch.distributed."""
        return int(os.environ["WORLD_SIZE"]) if dist.is_initialized() else 1

    @property
    def primary(self) -> bool:
        """Whether this is the rank 0 process."""
        return self.rank == 0

    @property
    def device(self) -> th.device:
        """The device associated with this process."""
        return (
            th.device("cuda", self.local_rank)
            if th.cuda.is_available()
            else th.device("cpu")
        )

    def shard_model(
        self, model: PreTrainedModel
    ) -> Union[FullyShardedDataParallel, PreTrainedModel]:
        """Shard the model using Fully Sharded Data Parallelism if needed."""
        if self.fsdp:
            _, layers = get_transformer_layers(model)
            layer_cls = type(layers[0])
            logger.info(
                f"Using '{layer_cls.__name__}' for transformer_auto_wrap_policy."
            )
            # Some models (e.g. T5-XXL) load with mixed float16/float32 params
            # even when torch_dtype=float16 is requested. FSDP requires uniform
            # dtype across all tensors in a module, so force-cast here.
            model = model.to(th.float16)
            return FullyShardedDataParallel(
                model,
                auto_wrap_policy=partial(
                    transformer_auto_wrap_policy, transformer_layer_cls={layer_cls}
                ),
                cpu_offload=CPUOffload(offload_params=self.cpu_offload),
                device_id=self.rank,
                # This turns out to be important for training speed
                forward_prefetch=True,
                mixed_precision=MixedPrecision(
                    param_dtype=th.float16,
                    reduce_dtype=th.float16,
                    buffer_dtype=th.float16,
                ),
            )
        elif self.cpu_offload:
            raise ValueError("CPU offload requires FSDP.")
        else:
            return model

    def distribute_lens(self, lens: Lens) -> Union[DDP, Lens]:
        """Distribute the lens using DistributedDataParallel and send lens to device."""
        logger.debug(f"Sending Lens to device {self.device}")
        if self.world_size > 1:
            lens.to(self.device)
            logger.debug("Distributing the lens across the GPUS using DDP ...")
            return DDP(lens, device_ids=[self.local_rank], find_unused_parameters=True)
        else:
            return lens.to(self.device)

    def dataloader(
        self,
        dataset: Dataset,
    ) -> dataloader2.DataLoader2:
        """Shard the dataset based on local rank."""
        dp = datapipes.iter.IterableWrapper(dataset)
        if self.world_size > 1:
            rs = dataloader2.DistributedReadingService()
        else:
            rs = None

        if self.dataloader_shuffle:
            dp = dp.shuffle()

        dp = dp.sharding_filter()
        dp = dp.batch(self.per_gpu_batch_size)
        dp = dp.collate()
        return dataloader2.DataLoader2(dp, reading_service=rs)

    def init(self):
        """Initialize distributed process group if started with elastic launch."""
        # Support both distributed and non-distributed training
        local_rank = os.environ.get("LOCAL_RANK")
        if local_rank is not None:
            dist.init_process_group(
                "nccl", timeout=timedelta(seconds=self.nccl_timeout)
            )
            assert (
                th.cuda.is_available()
            ), "CUDA must be available for distributed training"
            th.cuda.set_device(self.local_rank)

    def barrier(self) -> None:
        """Barrier for all processes."""
        if dist.is_initialized():
            dist.barrier()

    def send_to_device(self, pytree: TreeType) -> TreeType:
        """Move pytree to the current device."""
        return send_to_device(pytree, self.device)
