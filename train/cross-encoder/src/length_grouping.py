"""Length-grouped batching for the CrossEncoder reranker, so it can train at max_seq_length 8192.

Sort rows by length, cut into batches,
shuffle in blocks of world_size (so every rank sees the same width in the same micro-step, keeping
DDP in step), and run the longest block first as a real memory probe. A batch over a token budget
is row-chunked through the training step so a fixed batch size (hence a fixed global batch) survives
the rare long batches without OOM.

Two things keep this simple:
  * the data is already flattened to one (query, doc, label) row per candidate and concatenated into
    a single dataset, so only get_batch_sampler is overridden -- no multi-dataset block sampler;
  * CrossEncoderDataCollator hands the training step RAW TEXT columns (the loss tokenizes), so a
    row chunk is just a slice of the text lists and the label -- no packed cu_seq_lens surgery.

Pointwise MSE has no in-batch terms (each row's score is regressed onto its own teacher logit), so
the summed chunk gradients equal the whole-batch gradient exactly, and the optimizer still sees the
configured batch size and accumulation, i.e. the same global batch.
"""

from __future__ import annotations

import contextlib
import logging
import os

import numpy as np
import torch
from sentence_transformers.cross_encoder.trainer import CrossEncoderTrainer
from torch.utils.data import BatchSampler
from transformers import TrainerCallback


def row_lengths(dataset, tokenizer, columns, num_proc: int = 16) -> np.ndarray:
    """Summed token length of a row's text columns (query + doc) -- what its packed seq costs.

    Cached by `datasets` across runs, so the one-time measurement over the corpus is paid once.
    """

    def measure(batch):
        total = None
        for column in columns:
            texts = [t if isinstance(t, str) else "" for t in batch[column]]
            lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
            total = lengths if total is None else [a + b for a, b in zip(total, lengths)]
        return {"row_length": total}

    measured = dataset.map(
        measure, batched=True, batch_size=1000, num_proc=num_proc,
        remove_columns=dataset.column_names, desc="measuring row lengths",
    )
    return np.asarray(measured["row_length"], dtype=np.int64)


class LengthGroupedBatchSampler(BatchSampler):
    """Length-homogeneous batches, shuffled in blocks of ``block_size``, longest block first."""

    def __init__(self, lengths, batch_size, drop_last, block_size=1, probe_blocks=1, seed=0):
        super().__init__(range(len(lengths)), batch_size, drop_last)
        order = np.argsort(lengths, kind="stable")
        # Trim the remainder off the SHORT end, not the long end: the longest rows are the ones this
        # path exists to train (and the probe must see them), and short rows are plentiful.
        remainder = len(order) % batch_size
        if drop_last and remainder:
            order = order[remainder:]
        batches = [order[s : s + batch_size].tolist() for s in range(0, len(order), batch_size)]
        batches = batches[len(batches) % block_size :]
        blocks = [batches[i : i + block_size] for i in range(0, len(batches), block_size)]

        # Probe: the longest blocks, worst first, before anything else. Copies -- the same rows are
        # still delivered in the shuffled stream, so nothing is dropped from the epoch.
        probe = (
            [list(b) for block in reversed(blocks[-probe_blocks:]) for b in reversed(block)]
            if probe_blocks else []
        )
        np.random.RandomState(seed).shuffle(blocks)
        self.batches = probe + [b for block in blocks for b in block]
        self.probe_batches = len(probe)
        longest = int(lengths.max()) if len(lengths) else 0
        logging.info(
            f"length grouping: {len(batches)} batches in {len(blocks)} blocks of {block_size}, "
            f"shuffled by block; longest row {longest} tokens; {self.probe_batches} probe batches first"
        )

    def __iter__(self):
        yield from self.batches

    def __len__(self):
        return len(self.batches)


def _chunk_bounds(tokens, budget):
    """Contiguous row ranges whose token totals stay under the budget (a lone row may exceed it)."""
    bounds, start, total = [], 0, 0
    for row, count in enumerate(tokens):
        if total and total + count > budget:
            bounds.append((start, row))
            start, total = row, 0
        total += count
    bounds.append((start, len(tokens)))
    return bounds


class LengthGroupedCrossEncoderTrainer(CrossEncoderTrainer):
    """Length-grouped batches + row-chunking of the wide ones, for the CrossEncoder reranker."""

    def __init__(self, *args, chunk_tokens=16_000, probe_blocks=1, **kwargs):
        self._chunk_tokens = chunk_tokens
        self._probe_blocks = probe_blocks
        self._loss_scale = 1.0
        super().__init__(*args, **kwargs)

    def _text_columns(self, inputs):
        return [k for k, v in inputs.items() if isinstance(v, list) and (not v or isinstance(v[0], str))]

    def get_batch_sampler(self, dataset, batch_size, drop_last, valid_label_columns=None,
                          generator=None, seed=0):
        columns = [c for c in dataset.column_names if c not in ("label", "dataset_name")]
        with self.args.main_process_first(desc="measuring row lengths"):
            lengths = row_lengths(
                dataset, self.model.tokenizer, columns, num_proc=min(16, os.cpu_count() or 1)
            )
        return LengthGroupedBatchSampler(
            lengths, batch_size, drop_last,
            block_size=max(1, self.args.world_size), probe_blocks=self._probe_blocks, seed=self.args.seed,
        )

    def _row_tokens(self, inputs, rows):
        """Per-row token count summed over the text columns (the loss tokenizes the same text)."""
        tokens = [0] * rows
        for col in self._text_columns(inputs):
            ids = self.model.tokenizer(inputs[col], add_special_tokens=False)["input_ids"]
            for i, seq in enumerate(ids):
                tokens[i] += len(seq)
        return tokens

    def _take_rows(self, inputs, i, j):
        """Rows [i, j): slice the text lists and the label; pass scalars (prompt/task) through."""
        out = {}
        text_cols = set(self._text_columns(inputs))
        for k, v in inputs.items():
            if k in text_cols or (isinstance(v, torch.Tensor) and v.ndim >= 1 and v.shape[0] == len(inputs["label"])):
                out[k] = v[i:j]
            else:
                out[k] = v
        return out

    def training_step(self, model, inputs, num_items_in_batch=None):
        rows = inputs["label"].shape[0]
        tokens = self._row_tokens(inputs, rows)
        if sum(tokens) <= self._chunk_tokens:
            return super().training_step(model, inputs, num_items_in_batch)

        # Each chunk carries its share of the row mean; accumulated they equal the whole batch.
        # Only the last chunk syncs under DDP -- the earlier ones run under no_sync so the per-
        # backward all-reduces do not mismatch across ranks that chunked into different counts.
        bounds = _chunk_bounds(tokens, self._chunk_tokens)
        total = None
        for n, (i, j) in enumerate(bounds):
            self._loss_scale = (j - i) / rows
            try:
                ctx = contextlib.nullcontext() if n == len(bounds) - 1 else self.accelerator.no_sync(model)
                with ctx:
                    loss = super().training_step(model, self._take_rows(inputs, i, j), num_items_in_batch)
            finally:
                self._loss_scale = 1.0
            total = loss if total is None else total + loss
        return total

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = super().compute_loss(model, inputs, return_outputs=return_outputs,
                                   num_items_in_batch=num_items_in_batch)
        if self._loss_scale == 1.0:
            return out
        if return_outputs:
            loss, extra = out
            return loss * self._loss_scale, extra
        return out * self._loss_scale


class ProbeMemoryCallback(TrainerCallback):
    """Report peak memory the probe blocks reached, then reset. An OOM on any rank kills the run."""

    def __init__(self, probe_steps: int) -> None:
        self.probe_steps = probe_steps
        self.reported = False

    def on_step_end(self, args, state, control, **kwargs):
        if self.reported or state.global_step < self.probe_steps:
            return
        peak = torch.cuda.max_memory_allocated() / 2**30
        total = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
        logging.info(
            f"extended-length probe survived the longest blocks: peak {peak:.1f} GiB "
            f"of {total:.1f} GiB ({peak / total:.0%})"
        )
        torch.cuda.reset_peak_memory_stats()
        self.reported = True
