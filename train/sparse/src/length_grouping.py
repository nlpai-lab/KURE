"""Length-grouped batching for SPLADE, so it can train past its usual truncation limit.

``--extended_length`` in train.py. Rows are sorted by their longest document, cut into batches, and
the batches shuffled in blocks of ``world_size`` so every rank gets the same width in the same
micro-step (a short->long curriculum would strand the long documents at the tail of the cosine
decay), and the longest block of every dataset runs first as an out-of-memory probe. Every batch
still comes from one dataset, so the per-language batches are kept.

Two things are specific to SPLADE, both because of what a SPLADE batch is:

- The layout is padded. ST only packs padding-free for the feature-extraction task, not for the
  fill-mask head SPLADE reads, so every column arrives as ``[rows, width]`` and memory follows
  rows x width (measured ~1.25 MiB per padded token on an A6000, flat from 8x1024 to 4x8192).
  Length grouping is therefore also what keeps the padding waste down.
- The loss is not a per-row mean. FLOPS squares the *batch mean* of each vocabulary dimension,
  so running the rows of a wide batch through forward/backward a few at a time would regularize
  each chunk against its own mean, which is larger: on our trained SPLADE, 1-row chunks of an
  8-row batch inflate the query term x2.17 and the document term x1.68 (x1.18 / x1.11 at 4 rows),
  and only on the long batches. The wide batches go
  through ``losses.TokenBudgetSpladeLoss`` instead, which chunks with GradCache and is exact;
  nothing is chunked in the trainer here.
"""

from __future__ import annotations

import logging
import os
from itertools import accumulate

import numpy as np
import torch
from sentence_transformers.base.sampler import MultiDatasetDefaultBatchSampler
from sentence_transformers.sparse_encoder import SparseEncoderTrainer
from torch.utils.data import BatchSampler, SubsetRandomSampler
from transformers import TrainerCallback


def row_max_lengths(dataset, tokenizer, columns, num_proc: int = 16) -> np.ndarray:
    """Token length of the longest document in each row. Cached by `datasets` across runs."""

    def measure(batch):
        longest = None
        for column in columns:
            texts = [text if isinstance(text, str) else "" for text in batch[column]]
            lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
            longest = lengths if longest is None else [max(a, b) for a, b in zip(longest, lengths)]
        return {"row_max_length": longest}

    measured = dataset.map(
        measure, batched=True, batch_size=1000, num_proc=num_proc,
        remove_columns=dataset.column_names, desc="measuring row lengths",
    )
    return np.asarray(measured["row_max_length"], dtype=np.int64)


class LengthGroupedBatchSampler(BatchSampler):
    """Length-homogeneous batches, shuffled in blocks of ``block_size``, longest block first.

    Every block is ``block_size`` consecutive batches of the sorted order, so its batches share
    a width; the stream is a concatenation of whole blocks, which is what keeps one block per
    micro-step across the ranks.
    """

    def __init__(
        self,
        lengths: np.ndarray,
        batch_size: int,
        drop_last: bool,
        block_size: int = 1,
        probe_blocks: int = 1,
        seed: int = 0,
    ) -> None:
        super().__init__(range(len(lengths)), batch_size, drop_last)
        self.block_size = block_size

        order = np.argsort(lengths, kind="stable")
        # drop_last would otherwise discard the *final* batch, which in ascending order is the
        # longest rows -- the ones this whole path exists to train on, and the ones the probe
        # has to see. Take the remainder off the short end instead, where rows are plentiful;
        # the same for the batches that do not fill a whole block.
        remainder = len(order) % batch_size
        if drop_last and remainder:
            order = order[remainder:]
        batches = [
            order[start : start + batch_size].tolist()
            for start in range(0, len(order), batch_size)
        ]
        batches = batches[len(batches) % block_size :]
        blocks = [batches[i : i + block_size] for i in range(0, len(batches), block_size)]

        # The probe: the longest blocks, worst first, before anything else. Copies -- the same
        # rows are still delivered in the shuffled stream, so nothing is dropped from the epoch.
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

    def __len__(self) -> int:
        return len(self.batches)


class BlockProportionalBatchSampler(MultiDatasetDefaultBatchSampler):
    """ST's proportional sampler, drawing whole blocks instead of single batches.

    Which dataset comes next is still random in proportion to size, but once chosen it supplies
    ``block_size`` consecutive batches, so the block reaches the ranks intact.
    """

    def __init__(self, dataset, batch_samplers, generator=None, seed=0, block_size=1):
        super().__init__(dataset, batch_samplers, generator=generator, seed=seed)
        self.block_size = block_size

    def __iter__(self):
        if self.generator and self.seed is not None:
            self.generator.manual_seed(self.seed + getattr(self, "epoch", 0))
        offsets = [0] + list(accumulate(len(d) for d in self.dataset.datasets))
        iterators = [iter(s) for s in self.batch_samplers]

        def take_block(dataset_idx):
            for _ in range(self.block_size):
                try:
                    yield [i + offsets[dataset_idx] for i in next(iterators[dataset_idx])]
                except StopIteration:
                    return

        # Every dataset's probe blocks first, in a fixed order, so the opening micro-steps put
        # each dataset's worst batch on every rank -- a random draw could leave one dataset's
        # probe for later. Each sampler already leads with its own probe batches.
        probe_blocks = [getattr(s, "probe_batches", 0) // self.block_size for s in self.batch_samplers]
        for dataset_idx, n_blocks in enumerate(probe_blocks):
            for _ in range(n_blocks):
                yield from take_block(dataset_idx)

        blocks_left = [len(s) // self.block_size - n for s, n in zip(self.batch_samplers, probe_blocks)]
        draws = [i for i, n in enumerate(blocks_left) for _ in range(n)]
        for dataset_idx in SubsetRandomSampler(draws, generator=self.generator):
            yield from take_block(dataset_idx)

    def __len__(self) -> int:
        return sum(len(s) // self.block_size * self.block_size for s in self.batch_samplers)


class LengthGroupedTrainer(SparseEncoderTrainer):
    """Groups each dataset's batches by length; the wide ones are chunked inside the loss.

    Row lengths are measured when ST asks for a dataset's batch sampler, on the dataset object it
    hands over, so nothing has to be matched up by name or size. Rank 0 measures first and the
    other ranks read the `datasets` cache, which also makes a rerun on the same data free.
    """

    def __init__(self, *args, probe_blocks=1, **kwargs):
        self._probe_blocks = probe_blocks
        super().__init__(*args, **kwargs)

    def get_batch_sampler(self, dataset, batch_size, drop_last, valid_label_columns=None,
                          generator=None, seed=0):
        columns = [c for c in dataset.column_names if c not in ("label", "dataset_name")]
        with self.args.main_process_first(desc="measuring row lengths"):
            lengths = row_max_lengths(
                dataset, self.model.tokenizer, columns, num_proc=min(16, os.cpu_count() or 1)
            )
        return LengthGroupedBatchSampler(
            lengths, batch_size, drop_last,
            block_size=self.args.world_size, probe_blocks=self._probe_blocks, seed=self.args.seed,
        )

    def get_multi_dataset_batch_sampler(self, dataset, batch_samplers, generator=None, seed=0):
        return BlockProportionalBatchSampler(
            dataset, batch_samplers, generator=generator, seed=seed,
            block_size=self.args.world_size,
        )


class ProbeMemoryCallback(TrainerCallback):
    """Report the peak memory the probe blocks reached, then reset the counter.

    Every rank meets a probe batch, so every rank's peak is a real reading; only rank 0 logs.
    An OOM on any rank kills the run on its own -- the survival is the check.
    """

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
