"""Tools for tokenizing and manipulating text datasets."""
import math
from multiprocessing import cpu_count
from pathlib import Path
from typing import Iterator, Literal, TypeVar, Union

from datasets import Dataset, DatasetDict
from transformers import PreTrainedTokenizerBase

T = TypeVar("T", bound=Union[Dataset, DatasetDict])


def _iter_fasta_records(path: str, text_key: str) -> Iterator[dict[str, str]]:
    """Yield FASTA sequences as records with a sequence field.

    FASTA headers are ignored because training only requires the sequence text.
    """
    with open(path, "r") as f:
        seq_lines: list[str] = []
        for line in f:
            if line.startswith(">"):
                if seq_lines:
                    yield {text_key: "".join(seq_lines)}
                    seq_lines = []
                continue

            seq = line.strip()
            if seq:
                seq_lines.append(seq)

        if seq_lines:
            yield {text_key: "".join(seq_lines)}


def dataset_from_fasta(path: str, text_key: str = "sequence") -> Dataset:
    """Load a local FASTA file as a Hugging Face Dataset."""
    fasta_path = Path(path)
    if not fasta_path.exists():
        raise FileNotFoundError(f"FASTA file not found: {path}")

    dataset = Dataset.from_generator(
        _iter_fasta_records,
        gen_kwargs={"path": path, "text_key": text_key},
    )
    assert isinstance(dataset, Dataset)
    return dataset


def chunk_and_tokenize(
    data: T,
    tokenizer: PreTrainedTokenizerBase,
    *,
    model_type: Literal["causal", "masked"] = "causal",
    format: str = "torch",
    num_proc: int = min(cpu_count() // 2, 8),
    text_key: str = "text",
    max_seq_len: int = 2048,
    return_final_batch: bool = False,
    load_from_cache_file: bool = True,
) -> tuple[T, float]:
    """Tokenize a dataset for causal or masked language modeling.

    For `model_type="causal"`, this performs GPT-style concatenation/chunking: short
    sequences are merged with EOS separators and then split into fixed windows.

    For `model_type="masked"`, each sequence is tokenized independently and padded to
    `max_seq_len` so sample boundaries are preserved. (else attention across boundarie smight be an issue!)
    but use max_seq_len ~ 512 so that not many tokens wasted on padding (and that is roughly the length we use for deth analysis anyway!)

    Args:
        data: The dataset to chunk and tokenize.
        tokenizer: The tokenizer to use.
        model_type: Whether to tokenize for a causal or masked LM workflow.
        format: The format to return the dataset in, passed to `Dataset.with_format`.
        num_proc: The number of processes to use for tokenization.
        text_key: The key in the dataset to use as the text to tokenize.
        max_seq_len: The maximum length of a batch of input ids.
        return_final_batch: Whether to return the final batch, which may be smaller
            than the others.
        load_from_cache_file: Whether to load from the cache file.

    Returns:
        * The chunked and tokenized dataset.
        * The ratio of nats to bits per byte see https://arxiv.org/pdf/2101.00027.pdf,
            section 3.1.
    """

    def _tokenize_causal_fn(x: dict[str, list]):
        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        sep = tokenizer.eos_token or "<|endoftext|>"
        joined_text = sep.join([""] + x[text_key])
        output = tokenizer(
            # Concatenate all the samples together, separated by the EOS token.
            joined_text,  # start with an eos token
            max_length=chunk_size,
            return_attention_mask=False,
            return_overflowing_tokens=True,
            truncation=True,
        )

        if overflow := output.pop("overflowing_tokens", None):
            # Slow Tokenizers return unnested lists of ints
            assert isinstance(output["input_ids"][0], int)

            # Chunk the overflow into batches of size `chunk_size`
            chunks = [output["input_ids"]] + [
                overflow[i * chunk_size : (i + 1) * chunk_size]
                for i in range(math.ceil(len(overflow) / chunk_size))
            ]
            output = {"input_ids": chunks}

        total_tokens = sum(len(ids) for ids in output["input_ids"])
        total_bytes = len(joined_text.encode("utf-8"))

        if not return_final_batch:
            # We know that the last sample will almost always be less than the max
            # number of tokens, and we don't want to pad, so we just drop it.
            output = {k: v[:-1] for k, v in output.items()}

        output_batch_size = len(output["input_ids"])

        if output_batch_size == 0:
            raise ValueError(
                "Not enough data to create a single batch complete batch."
                " Either allow the final batch to be returned,"
                " or supply more data."
            )

        # We need to output this in order to compute the number of bits per byte
        div, rem = divmod(total_tokens, output_batch_size)
        output["length"] = [div] * output_batch_size
        output["length"][-1] += rem

        div, rem = divmod(total_bytes, output_batch_size)
        output["bytes"] = [div] * output_batch_size
        output["bytes"][-1] += rem

        return output

    def _tokenize_masked_fn(x: dict[str, list]):
        if tokenizer.pad_token_id is None:
            raise ValueError(
                "Masked tokenization requires a tokenizer with `pad_token_id`."
            )

        chunk_size = min(tokenizer.model_max_length, max_seq_len)
        texts = x[text_key]
        output = tokenizer(
            texts,
            max_length=chunk_size,
            padding="max_length",
            return_attention_mask=True,
            truncation=True,
        )

        input_ids = output["input_ids"]
        attention_mask = output["attention_mask"]
        output_batch_size = len(input_ids)
        if output_batch_size == 0:
            raise ValueError("Tokenizer returned an empty batch.")

        # Track non-padding token counts for metrics conversion.
        lengths = [sum(mask) for mask in attention_mask]
        byte_counts = [len(text.encode("utf-8")) for text in texts]

        output["length"] = lengths
        output["bytes"] = byte_counts

        return output

    if model_type == "causal":
        tokenize_fn = _tokenize_causal_fn
    elif model_type == "masked":
        tokenize_fn = _tokenize_masked_fn
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    data = data.map(
        tokenize_fn,
        # Batching is important for ensuring that we don't waste tokens
        # since we always throw away the last element of the batch we
        # want to keep the batch size as large as possible
        batched=True,
        batch_size=2048,
        num_proc=num_proc,
        remove_columns=get_columns_all_equal(data),
        load_from_cache_file=load_from_cache_file,
    )
    total_bytes: float = sum(data["bytes"])
    total_tokens: float = sum(data["length"])

    columns = ["input_ids"]
    if model_type == "masked": #TODO: is this really required? or do models (e.g. ESM2) build its own attention amsk based on padding tokens?
        columns.append("attention_mask")

    return data.with_format(format, columns=columns), (
        total_tokens / total_bytes
    ) / math.log(2)


def get_columns_all_equal(dataset: Union[Dataset, DatasetDict]) -> list[str]:
    """Get a single list of columns in a `Dataset` or `DatasetDict`.

    We assert the columms are the same across splits if it's a `DatasetDict`.

    Args:
        dataset: The dataset to get the columns from.

    Returns:
        A list of columns.
    """
    if isinstance(dataset, DatasetDict):
        cols_by_split = dataset.column_names.values()
        columns = next(iter(cols_by_split))
        if not all(cols == columns for cols in cols_by_split):
            raise ValueError("All splits must have the same columns")

        return columns

    return dataset.column_names
