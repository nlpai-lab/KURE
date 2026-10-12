"""Load the v3 reranker-labeled training data for distillation.

Every SFT dataset has the same columns: `query`, `positive`, `negative_0..9`, `label` (11 floats,
positive first).
Only one teacher's directory should be used per run -- the two label sets sit on
different scales.
"""

from __future__ import annotations

import logging
import os

from datasets import DatasetDict, List, Value, concatenate_datasets, load_from_disk

# sft_ko stores `label` as a fixed-length list of float32, sft_en as a variable-length list of
# float64, so concatenating them fails on feature alignment. Cast both to one type first.
LABEL_FEATURE = List(Value("float32"))


def _minmax(batch: dict) -> dict:
    labels = []
    for scores in batch["label"]:
        lo, hi = min(scores), max(scores)
        labels.append([(s - lo) / (hi - lo) for s in scores] if hi > lo else scores)
    return {"label": labels}


def _keep_true_negatives(margin: float):
    """Keep rows whose hardest negative scores below `positive - margin`.

    A row where a negative outscores the positive is more likely a mislabelled positive than
    a usable negative. It matters for the contrastive term, whose cross-entropy target is the
    positive regardless of what the teacher thinks; the KL term alone would just distill the
    inversion. The threshold is a margin, not a ratio: these are logits and go negative.
    """

    def fn(batch: dict) -> list[bool]:
        return [max(label[1:]) < label[0] - margin for label in batch["label"]]

    return fn


def _truncate(num_negatives: int):
    def fn(batch: dict) -> dict:
        return {"label": [scores[: num_negatives + 1] for scores in batch["label"]]}

    return fn


def load_train_datasets(
    paths: list[str],
    num_negatives: int,
    mix_languages: bool = True,
    normalize_scores: bool = False,
    drop_false_negatives: bool = False,
    false_negative_margin: float = 0.0,
    max_train_samples: int | None = None,
    num_proc: int = 16,
):
    """Return a Dataset (mixed) or DatasetDict (one entry per path).

    Columns are ordered `query, positive, negative_*, label` because the data collator
    and the loss both read them positionally.
    """
    datasets = {}
    for path in paths:
        ds = load_from_disk(path)
        available = sum(1 for c in ds.column_names if c.startswith("negative_"))
        if num_negatives > available:
            raise ValueError(f"{path} has {available} negatives, {num_negatives} requested")

        # Subsample first so the maps below do not rewrite millions of rows for a smoke test.
        if max_train_samples and max_train_samples < ds.num_rows:
            ds = ds.shuffle(seed=42).select(range(max_train_samples))

        # The label vector is positive-first, so it has to be cut to match the kept negatives.
        if num_negatives < available:
            ds = ds.map(_truncate(num_negatives), batched=True, num_proc=num_proc)
        # Before normalize_scores: min-max rescales every row to [0, 1], which would make a
        # margin in teacher-logit units meaningless.
        if drop_false_negatives:
            before = ds.num_rows
            ds = ds.filter(_keep_true_negatives(false_negative_margin), batched=True,
                           num_proc=num_proc, desc=f"{path}: false-negative filter")
            logging.info(
                f"{path}: {before:,} -> {ds.num_rows:,} rows "
                f"(dropped {before - ds.num_rows:,}, {100 * (before - ds.num_rows) / before:.2f}%) "
                f"with margin={false_negative_margin}"
            )
        if normalize_scores:
            ds = ds.map(_minmax, batched=True, num_proc=num_proc)

        columns = ["query", "positive"]
        columns += [f"negative_{i}" for i in range(num_negatives)]
        ds = ds.select_columns(columns + ["label"])
        if mix_languages and ds.features["label"] != LABEL_FEATURE:
            ds = ds.cast_column("label", LABEL_FEATURE)

        datasets[os.path.basename(path.rstrip("/"))] = ds
        logging.info(f"{path}: {ds.num_rows:,} rows, columns {ds.column_names}")

    # No shuffle here: the trainer's sampler already shuffles a single concatenated dataset.
    if mix_languages:
        return concatenate_datasets(list(datasets.values()))
    return DatasetDict(datasets)


def load_pft_datasets(
    paths: list[str],
    mix_languages: bool = False,
    max_train_samples: int | None = None,
):
    """Load the weakly-supervised `(anchor, positive)` pairs for pre-finetuning.

    Separate from `load_train_datasets` because these rows carry neither a teacher label nor
    mined negatives -- the contrastive loss takes its negatives from the batch, which is why
    the stage is run at a batch size that would be pointless for distillation.
    """
    datasets = {}
    for path in paths:
        ds = load_from_disk(path)
        if max_train_samples and max_train_samples < ds.num_rows:
            ds = ds.shuffle(seed=42).select(range(max_train_samples))
        # The collator reads columns by position, so any third column left in place would
        # silently become a second positive.
        ds = ds.select_columns(["anchor", "positive"])
        datasets[os.path.basename(path.rstrip("/"))] = ds
        logging.info(f"{path}: {ds.num_rows:,} rows, columns {ds.column_names}")

    # Keeping the languages apart matters more here than in distillation: the contrastive
    # negatives come from the batch, and a mixed batch makes most of them separable by
    # language alone.
    if mix_languages:
        return concatenate_datasets(list(datasets.values()))
    return DatasetDict(datasets)


def collator_prompts(columns: list[str], query_prompt: str, document_prompt: str) -> dict:
    """Map each text column to the prompt its side of the pair needs.

    Prompts are not cosmetic: a body pre-finetuned with `[Q] ` / `[D] ` (KURE-v2-unsupervised)
    loses accuracy without them.
    """
    prompts = {}
    for col in columns:
        if col in ("query", "anchor") and query_prompt:
            prompts[col] = query_prompt
        elif (col == "positive" or col.startswith("negative")) and document_prompt:
            prompts[col] = document_prompt
    return prompts
