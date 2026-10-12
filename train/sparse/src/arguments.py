from dataclasses import dataclass, field
from typing import Optional

from sentence_transformers.sparse_encoder.training_args import SparseEncoderTrainingArguments

from evaluators import TASKS


@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="omai-research/milco-650m")
    dtype: str = field(default="bfloat16")
    attn_implementation: str = field(default="flash_attention_2")
    max_seq_length: int = field(default=8192)
    # Jang et al. 2026, "Rescaling MLM-Head for Neural Sparse Retrieval" (arXiv:2606.18811):
    # W_head <- W_head / k once at initialization. Large-norm MLM heads (ModernBERT family) inflate
    # the unnormalized dot-product scores and collapse contrastive SPLADE training; k=8-16 fixed it
    # there. Measured on skt/A.X-Encoder-base at 8192: zero-shot q.d ~86k at k=1, ~5.7k at k=8,
    # ~1.7k at k=16. 1.0 = standard SPLADE.
    mlm_head_rescale: float = field(default=1.0)
    # SparseUp (Linkup 2026) pooling: log(1+ReLU(logits - shift)) with only the top_k logits kept
    # per position. See src/pooling.py. 0 / 0 = vanilla SpladePooling.
    splade_logit_shift: float = field(default=0.0)
    splade_top_k: int = field(default=0)
    splade_exclude_token_ids: list[int] = field(
        default_factory=list,
        metadata={"nargs": "+", "help": "Prefix tokens ([Q]/[D]) attended but not pooled, and "
                                        "their tied decoder dims muted."},
    )
    # Vocabulary canonicalization (src/pooling.py::canonical_ids_for): 0 = off; n merges
    # tokens whose "##"-stripped NFKC-lowercased form has >= n characters into one output term.
    splade_canonical_min_chars: int = field(default=0)


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
    """Fields shared by the dense and sparse training arguments.

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
            "help": "kldiv | kldiv_minmax | contrastive_kldiv (sparse only) | "
            "contrastive (sparse only; the pre-finetuning stage, which reads anchor/positive "
            "pairs instead of the labeled distillation data)"
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
class SparseTrainingArguments(_DistillArguments, SparseEncoderTrainingArguments):
    # SPLADE FLOPS regularization. Defaults from "From Distillation to Hard Negative Sampling".
    document_regularizer_weight: float = field(default=3e-5)
    query_regularizer_weight: float = field(default=5e-5)
    splade_regularizer: str = field(
        default="flops",
        metadata={"help": "flops | l1, both sides. L1 is on a far larger scale: retune the weights.",
                  "choices": ["flops", "l1"]},
    )
    splade_pooling: str = field(default="max", metadata={"help": "max | sum"})
    # The MLM logits a SPLADE forward has to materialize are [batch, seq_len, vocab], and autograd
    # holds one per column. That is what sets the batch
    # size; keep the effective batch up with gradient_accumulation_steps instead.
    splade_chunk_size: int = field(
        default=128,
        metadata={
            "help": "Tokens per chunk in SpladePooling. Bounds the temporaries the pooling "
            "allocates on top of the logits themselves. 0 disables chunking."
        },
    )
    # GradCache: only worth its second forward pass when a contrastive term makes the batch
    # size part of the objective. Verified to reproduce plain SpladeLoss exactly in float64.
    use_cached_splade: bool = field(default=False)
    mini_batch_size: int = field(
        default=8, metadata={"help": "GradCache chunk; a memory knob that does not change the loss."}
    )
    # contrastive_kldiv only. Separate temperatures let a sharp student distill the teacher's
    # ranking rather than its margins; 0 means "reuse kldiv_temperature for both".
    contrastive_weight: float = field(default=1.0)
    kldiv_weight: float = field(default=1.0)
    contrastive_scale: float = field(
        default=1.0, metadata={"help": "Multiplies the student similarity in the contrastive term."}
    )
    student_temperature: float = field(default=0.0)
    teacher_temperature: float = field(default=0.0)
    # Per-row min-max to [0, 1] before the softmax, as in MinMaxSparseDistillKLDivLoss. It
    # compresses the score spread to 1, so the temperature must drop with it: measured on the
    # Qwen labels, min-max at T=0.15 reproduces the raw teacher's sharpness at T=2.0
    # (top probability 0.743 vs 0.735 over 11 candidates).
    normalize_student: bool = field(default=False)
    normalize_teacher: bool = field(default=False)
    gather_across_devices: bool = field(
        default=True,
        metadata={"help": "Pool in-batch negatives across ranks, so DDP widens them rather than splitting them."},
    )
    # Extended-length training (--extended_length): see length_grouping.py.
    extended_length: bool = field(default=False)
    exlen_chunk_tokens: int = field(
        default=30000,
        metadata={
            "help": "Padded tokens whose graph may be alive at once. A batch whose columns total "
            "more goes through GradCache in chunks of at most this many. ~1.25 MiB per token on "
            "an A6000, so 30000 is ~38 GiB of a 48 GiB card."
        },
    )
    exlen_probe_blocks: int = field(
        default=1,
        metadata={"help": "Longest blocks per dataset to run first as the memory probe. 0 disables."},
    )
    eval_max_active_dims: int = field(
        default=512,
        metadata={
            "help": "Keep only the top-N vocabulary dimensions per vector when evaluating, "
            "which is how a SPLADE index is served anyway. 0 disables the cap -- but an "
            "unregularized model then holds an evaluation corpus as many GB of sparse tensors."
        },
    )
