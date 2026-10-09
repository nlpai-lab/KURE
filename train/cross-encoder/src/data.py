"""Load the SFT data for reranker (CrossEncoder) pointwise-MSE distillation.

MSELoss wants exactly two text columns plus a `label` column and regresses the model's raw score
onto that label -- the teacher's yes/no logit difference. The SFT rows are 11-way
(`query, positive, negative_0..9, label[11]`, positive first), the same format as KURE-v2's SFT
data; they are flattened here into (query, doc, label) pointwise rows, 11 per input row.
"""

from __future__ import annotations

import logging

from datasets import concatenate_datasets, load_from_disk

NEG_COLS = [f"negative_{i}" for i in range(10)]
CAND_COLS = ["positive"] + NEG_COLS  # label[j] is the teacher score of CAND_COLS[j]


def _flatten(batch: dict) -> dict:
    """One (query, doc, label) row per candidate: the positive, then the ten negatives."""
    out_q, out_d, out_l = [], [], []
    for i, q in enumerate(batch["query"]):
        label = batch["label"][i]
        for j, c in enumerate(CAND_COLS):
            out_q.append(q)
            out_d.append(batch[c][i])
            out_l.append(float(label[j]))
    return {"query": out_q, "doc": out_d, "label": out_l}


def load_sft(paths: list[str], max_train_samples: int | None = None, num_proc: int = 16):
    """11-way labeled rows -> flattened pointwise rows. `max_train_samples` counts input rows."""
    parts = []
    for p in paths:
        ds = load_from_disk(p).select_columns(["query"] + CAND_COLS + ["label"])
        if max_train_samples and max_train_samples < ds.num_rows:
            ds = ds.shuffle(seed=42).select(range(max_train_samples))
        ds = ds.map(_flatten, batched=True, remove_columns=ds.column_names,
                    num_proc=num_proc, desc=f"{p}: flatten")
        # map() keeps pre-existing column names in their old positions and appends new ones,
        # which yields (query, label, doc); pin the documented (query, doc, label) order.
        ds = ds.select_columns(["query", "doc", "label"])
        parts.append(ds)
        logging.info(f"{p}: {ds.num_rows:,} (query, doc, label) rows")
    return concatenate_datasets(parts)
