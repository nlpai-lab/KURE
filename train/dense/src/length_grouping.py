"""Length-grouped batching, so the student can train past its usual truncation limit.

Exposed as extended-length training: ``--extended_length`` in train.py (on in train.sh). The flag
turns on everything below at once.

6.2% of the corpus's tokens are thrown away at max_seq_length 1024 -- the length distribution
has a very thin, very long tail (p50 198 tokens, p99 913, p99.9 7305, max 46098), so only 3.4%
of rows carry a document over 1024 and 0.3% carry one over 8192. Under flash attention ST
collates padding-free, so a batch costs what its rows actually hold; raising the limit changes
nothing for the 96.6% of short rows, and the tail rows (an 8192 document times 11 columns is
~90k tokens) are handled by the token-budget chunking below. Rows are sorted by length, cut
into batches and shuffled -- BGE-M3's "length grouping with a fixed seed" -- for the two
reasons in the next paragraphs: keeping the ranks in step, and a deterministic probe.

Why grouped-and-shuffled rather than a short->long curriculum. The curriculum literature puts
short-first at the *start* of training (Sequence Length Warmup, Shortformer), to tame gradient
variance during LR warmup; after that, training is i.i.d. at full length. Every recipe that
trains long documents *late* -- Llama 3's 8K->128K stages, the 128K continual-pretraining
recipes, BGE-M3's own long-document stage, WSD's specialised decay phases -- gives that phase
its own learning rate. Nobody strands long documents at the end of a cosine decay: here the last
3.4% of steps runs at 0.35% of peak LR, the last 0.5% at 0.009%. And this student is already a
32K-context model that lost nothing when evaluated at 8192 after training at 1024, so there is
no positional adaptation to schedule. Shuffling spreads the long batches across the run, where
each gets an ordinary LR, which is what the i.i.d. regime of those recipes does.

Blocks, so the ranks stay in step. DDP meets once per optimizer step, at the all-reduce, and
waits for the slowest rank. accelerate deals global batches round-robin (batch ``i`` to rank
``i % world_size``), so if batches were shuffled one at a time a rank holding an 8192-token
batch would keep seven idle ranks waiting -- half of the throughput gain, on the simulation.
Instead the sorted batches are grouped into blocks of ``world_size`` neighbours and the
*blocks* are shuffled: each block lands one batch per rank, so all ranks see the same width
in the same micro-step. This is the megabatch trick behind HF's ``LengthGroupedSampler``. The
multi-dataset sampler below draws whole blocks, so the per-batch language separation and the
proportional mixing are both kept.

Row chunking, by token budget. A 32-row batch is 352 documents; the top block at 8192 tokens
is ~30x the tokens of a typical batch and cannot fit, and dropping the batch size for the whole
run would starve the 96.6% of batches that are short. So a batch whose text columns hold more
than ``chunk_tokens`` tokens in total is run through forward and backward a few rows at a time,
each chunk cut so its tokens stay under the budget and its loss weighted by its share of the
rows. Tokens, not rows or width, because under flash attention ST packs a batch padding-free:
every column arrives as one ``(1, total_tokens)`` sequence with ``cu_seq_lens`` boundaries, so
memory follows the token total and rows cannot be sliced on a batch axis -- ``_take_rows``
cuts the packed tensors at the boundaries instead (and dim 0 of padded ones). The loss is a
per-row mean with no in-batch terms (each query is scored only against its own 11 candidates),
so the summed chunk gradients equal the whole-batch gradient exactly; the optimizer still sees
32-row micro-batches and the same accumulation, i.e. the same global batch. (GOR, when
enabled, does read in-batch statistics and would see the chunk instead.)

The probe. The longest block of every dataset runs *first*, as the opening micro-steps of the
real run, so every rank meets its worst batch through the chunked path it will actually use.
Checking memory before training would lie: Adam's state is two fp32 copies of the parameters
(~8.8 GB for a 1B student) and is not allocated until the first optimizer.step(). Running the
worst batches for real is the only honest check, and it fails in the first minutes rather than
forty hours in. Those rows stay in the shuffled stream as well.
"""

from __future__ import annotations

import contextlib
import logging
import os
from itertools import accumulate

import numpy as np
import torch
from sentence_transformers.base.sampler import MultiDatasetDefaultBatchSampler
from sentence_transformers.sentence_transformer.trainer import SentenceTransformerTrainer
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


def _row_tokens(inputs, rows: int) -> torch.Tensor:
    """Tokens per row summed over the text columns, for the packed or the padded layout."""
    tokens = torch.zeros(rows, dtype=torch.long)
    for key, value in inputs.items():
        if key.endswith("_cu_seq_lens_q"):
            cu = value.to("cpu", torch.long)
            tokens += cu[1:] - cu[:-1]
        elif key.endswith("_attention_mask"):
            tokens += value.sum(1).to("cpu", torch.long)
    return tokens


def _chunk_bounds(tokens: torch.Tensor, budget: int) -> list[tuple[int, int]]:
    """Contiguous row ranges whose token totals stay under the budget (a lone row may exceed it)."""
    bounds, start, total = [], 0, 0
    for row, count in enumerate(tokens.tolist()):
        if total and total + count > budget:
            bounds.append((start, row))
            start, total = row, 0
        total += count
    bounds.append((start, len(tokens)))
    return bounds


def _take_rows(inputs, rows: int, i: int, j: int) -> dict:
    """Rows [i, j) of a collated batch, cutting packed columns at their sequence boundaries."""
    packed = [k[: -len("cu_seq_lens_q")] for k in inputs if k.endswith("_cu_seq_lens_q")]
    out = {}
    for key, value in inputs.items():
        prefix = max((p for p in packed if key.startswith(p)), key=len, default=None)
        if prefix is None:
            per_row = isinstance(value, torch.Tensor) and value.ndim > 0 and value.shape[0] == rows
            out[key] = value[i:j] if per_row else value
            continue
        name = key[len(prefix):]
        cu = inputs[prefix + ("cu_seq_lens_k" if name.endswith("_k") else "cu_seq_lens_q")]
        lo, hi = int(cu[i]), int(cu[j])
        if name in ("input_ids", "position_ids"):
            out[key] = value[:, lo:hi]
        elif name == "seq_idx":
            out[key] = value[:, lo:hi] - i
        elif name in ("cu_seq_lens_q", "cu_seq_lens_k"):
            out[key] = value[i : j + 1] - value[i]
        elif name in ("max_length_q", "max_length_k"):
            out[key] = int((cu[i + 1 : j + 1] - cu[i:j]).max())
        else:
            out[key] = value
    return out


class LengthGroupedTrainer(SentenceTransformerTrainer):
    """Groups each dataset's batches by length, and row-chunks the wide ones through the step.

    Row lengths are measured when ST asks for a dataset's batch sampler, on the dataset object it
    hands over, so nothing has to be matched up by name or size. Rank 0 measures first and the
    other ranks read the `datasets` cache, which also makes a rerun on the same data free.
    """

    def __init__(self, *args, chunk_tokens=160_000, probe_blocks=1, **kwargs):
        self._chunk_tokens = chunk_tokens
        self._probe_blocks = probe_blocks
        self._loss_scale = 1.0
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

    def training_step(self, model, inputs, num_items_in_batch=None):
        rows = inputs["label"].shape[0]
        tokens = _row_tokens(inputs, rows)
        if int(tokens.sum()) <= self._chunk_tokens:
            return super().training_step(model, inputs, num_items_in_batch)

        # Each chunk goes through the parent's step unchanged -- its loss scaling and backward --
        # carrying only the chunk's share of the row mean, so the accumulated gradient is the
        # whole batch's. Only the last chunk may synchronise: on the sync micro-step DDP issues
        # one all-reduce per backward, and ranks chunk different batches into different counts,
        # so the collectives would mismatch. Under no_sync the earlier chunks accumulate locally
        # and the final backward reduces the accumulated .grad -- the standard gradient-
        # accumulation contract. On a non-sync micro-step the caller's no_sync is already in
        # force; nesting is harmless.
        bounds = _chunk_bounds(tokens, self._chunk_tokens)
        total = None
        for n, (i, j) in enumerate(bounds):
            self._loss_scale = (j - i) / rows
            try:
                with contextlib.nullcontext() if n == len(bounds) - 1 else self.accelerator.no_sync(model):
                    loss = super().training_step(model, _take_rows(inputs, rows, i, j), num_items_in_batch)
            finally:
                self._loss_scale = 1.0
            total = loss if total is None else total + loss
        return total

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        out = super().compute_loss(
            model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
        )
        if self._loss_scale == 1.0:
            return out
        if return_outputs:
            loss, extra = out
            return loss * self._loss_scale, extra
        return out * self._loss_scale


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
