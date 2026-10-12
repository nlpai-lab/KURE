from dataclasses import dataclass, field
from typing import Optional

from sentence_transformers.sentence_transformer.training_args import (
    SentenceTransformerTrainingArguments,
)

from evaluators import TASKS


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="nvidia/Nemotron-3-Embed-1B-BF16")
    dtype: str = field(default="bfloat16")
    attn_implementation: str = field(default="flash_attention_2")
    max_seq_length: int = field(default=8192)
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "For a backbone whose modeling code lives in its repo, not transformers."},
    )


@dataclass
class DataArguments:
    train_dataset_path: list[str] = field(metadata={"nargs": "+"})
    eval_subsets: list[str] = field(
        default_factory=lambda: list(TASKS),
        metadata={"nargs": "+", "help": "In-training eval tasks; see evaluators.TASKS. "
                                        "Defaults to all five Korean benchmark tasks."},
    )
    num_negatives: int = field(
        default=10, metadata={"help": "How many of the mined negatives to train on."}
    )
    max_train_samples: Optional[int] = field(
        default=None, metadata={"help": "Subsample the training set. For smoke tests."}
    )


@dataclass
class _DistillArguments:
    """Distillation fields.

    Listed first in the bases so dataclass field ordering puts these after the inherited
    HF/ST fields.
    """

    query_prompt: str = field(default="")
    document_prompt: str = field(default="")
    mix_languages: bool = field(
        default=False,
        metadata={
            "help": "If True, concatenate ko and en into one dataset so batches mix languages. "
            "If False (default), keep them separate and form each batch from a single language. "
            "KL distillation scores each query only against its own 11 candidates -- there are no "
            "in-batch negatives -- so batch composition does not change the loss, and mixing "
            "costs a full re-cast of both datasets to align their label feature types."
        },
    )
    distill_loss: str = field(
        default="kldiv",
        metadata={
            "help": "kldiv | kldiv_minmax"
        },
    )
    kldiv_temperature: float = field(
        default=1.0,
        metadata={
            "help": "Softmax temperature for both teacher and student. kldiv_minmax squeezes "
            "scores into [0, 1] and wants a much lower value (0.05-0.2)."
        },
    )
    normalize_scores: bool = field(
        default=False,
        metadata={"help": "Min-max normalize the teacher labels per row, in the dataset."},
    )
    drop_false_negatives: bool = field(
        default=False,
        metadata={"help": "Drop rows whose hardest negative outscores the positive by the teacher."},
    )
    false_negative_margin: float = field(
        default=0.0,
        metadata={"help": "Required gap: keep a row only if max(negatives) < positive - margin."},
    )


@dataclass
class DenseTrainingArguments(_DistillArguments, SentenceTransformerTrainingArguments):
    matryoshka_dims: list[int] = field(
        default_factory=lambda: [2048, 1024, 512, 256],
        metadata={
            "nargs": "+",
            "help": "Truncated dimensions to train jointly (Matryoshka). Pass the "
            "full embedding size alone (e.g. 2048) to train without truncation -- "
            "HfArgumentParser forces nargs='+' on list fields, so an empty list is not "
            "expressible on the command line.",
        },
    )
    # Global Orthogonal Regularization on top of the KL loss (losses.KLDivGORLoss). 0 = off.
    # Defaults are EmbeddingGemma's: second-moment term only, summed over the input columns.
    # Applied at every Matryoshka dimension. Scale the weight to the KL variant: ~1 for
    # kldiv_minmax, orders of magnitude more for raw kldiv.
    gor_weight: float = field(default=0.0)
    gor_mean_weight: float = field(default=0.0)
    gor_second_moment_weight: float = field(default=1.0)
    gor_aggregation: str = field(default="sum", metadata={"help": "sum | mean"})

    # Extended-length training mode (src/length_grouping.py): length-homogeneous batches in
    # shuffled order, so max_seq_length can be raised to 8192 without every batch paying for it
    # -- a batch costs what its own rows hold (ST packs padding-free under flash attention).
    # A batch over the token budget is row-chunked through the training step. Batches are
    # shuffled in blocks of world_size so every rank sees the same width in the same
    # micro-step, and the longest block runs first as a memory probe.
    extended_length: bool = field(default=False)
    exlen_chunk_tokens: int = field(
        default=160_000,
        metadata={
            "help": "A batch whose text columns exceed this many tokens in total is run a few rows "
            "at a time, each chunk under the budget. Exact: the loss has no in-batch terms."
        },
    )
    exlen_probe_blocks: int = field(
        default=1,
        metadata={
            "help": "Per dataset, how many of the longest blocks (one batch per rank each) to run "
            "first, as the real opening micro-steps, so every rank proves the configuration "
            "does not OOM before forty hours are spent on it."
        },
    )
