"""Losses, collator, and callbacks for the KURE-v2 late-interaction training scripts.

Extracted from lightonai's mdenseon-mlateon training code
(scripts/finetune/multilingual_late_interaction.py), Apache License 2.0.
Three pieces are kept, verbatim:

  CachedContrastiveKLDiv   gradient-cached contrastive + KL-distillation loss (stage 2)
  StopAtStepCallback       stop after N optimizer steps (smoke runs)
  ColBERTCollatorSampleNeg collator that samples a subset of stored negatives per step
"""

from __future__ import annotations

import contextlib
import itertools
import random
from typing import Callable, Iterable, Iterator

import torch
import torch.nn.functional as F
from pylate.losses.cached_contrastive import RandContext
from pylate.losses.contrastive import extract_skiplist_mask
from pylate.utils import all_gather, all_gather_with_gradients, get_rank, get_world_size
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.training_args import TrainingArguments

class CachedContrastiveKLDiv(torch.nn.Module):
    """Gradient-cached contrastive and KL-divergence distillation loss for ColBERT model.

    A single chunked encoder pass feeds two terms. The contrastive term is a cross-entropy over
    the MeanMaxSim scores of every in-batch document of every rank, so the other queries'
    positives and all of the negatives act as negatives. The KL-div term is, per query, the
    KL divergence between the student distribution over its own (positive, negative_0, ...,
    negative_k) and the teacher cross-encoder scores of those same documents, each side
    softmaxed at its own temperature. The loss is
    ``contrastive_weight * contrastive + kldiv_weight * kldiv``.

    The GradCache implementation is optimized for batches mixing short and long texts
    and works faster under limited GPU memory.

    Parameters
    ----------
    model
        ColBERT model.
    contrastive_temperature
        Temperature applied to the student scores of the contrastive term.
    student_temperature
        Softmax temperature applied to the student scores of the KL-div term.
    teacher_temperature
        Softmax temperature applied to the teacher scores of the KL-div term.
    contrastive_weight
        Weight of the contrastive term.
    kldiv_weight
        Weight of the KL-div term, 0 trains on the contrastive term alone.
    mini_batch_size
        GradCache chunk size, a pure memory knob that does not affect the loss.
    defer_grad_sync
        Whether to accumulate the re-forward chunk gradients locally and allreduce once.
    score_bytes_budget
        Cap on the bytes of a single MaxSim score tile.
    chunk_token_budget
        Maximum number of tokens allowed in one encoder chunk, rows x the chunk's max length.
    score_anchor_token_budget
        Cap on the padded query tokens (rows x query max length) of one scoring chunk.
    rand_context
        Whether to capture and replay the RNG state of each chunk, needed only to replay
        dropout: "auto" does it when the model has active dropout, "always" and "never" force it.

    Requirements
    ------------
    1. Columns ordered (query, positive, negative_0, ..., negative_k).
    2. When kldiv_weight is not 0, labels of shape (batch_size, 1 + k) holding the teacher
       scores of those same documents, in the same order.

    Examples
    --------
    >>> from pylate import models

    >>> model = models.ColBERT(
    ...     model_name_or_path="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    ... )

    >>> loss = CachedContrastiveKLDiv(model=model, mini_batch_size=1)

    >>> query = model.tokenize(["fruits are healthy."], is_query=True, pad=False)

    >>> positive = model.tokenize(["fruits are good for health."], is_query=False, pad=False)

    >>> negative = model.tokenize(["fruits are bad for health."], is_query=False, pad=False)

    >>> labels = torch.tensor([[0.7, 0.3]], dtype=torch.float32)

    >>> output = loss(sentence_features=[query, positive, negative], labels=labels)

    >>> assert isinstance(output.item(), float)
    """

    class _FusedPlan:
        """Length-sorted chunking plan over the concatenation of all columns."""

        def __init__(
            self,
            sentence_features: list[dict[str, torch.Tensor]],
            row_cap: int,
            token_budget: int | None,
        ) -> None:
            device = sentence_features[0]["attention_mask"].device
            self.col_lengths = [sf["attention_mask"].sum(dim=1) for sf in sentence_features]
            col_sizes = [int(l.size(0)) for l in self.col_lengths]
            all_lens = torch.cat(self.col_lengths)
            col_ids = torch.cat(
                [
                    torch.full((n,), c, dtype=torch.long, device=device)
                    for c, n in enumerate(col_sizes)
                ]
            )
            row_ids = torch.cat([torch.arange(n, device=device) for n in col_sizes])
            order = torch.argsort(all_lens, descending=True, stable=True)
            # Rows are grouped by column within each chunk so the encoder input is one index_select + cat per column
            self.chunks: list[tuple[int, int, int]] = []
            self.chunk_cols: list[torch.Tensor] = []  # column id per row, chunk order
            self.chunk_rows: list[torch.Tensor] = []  # original row id per row
            sorted_cols = col_ids.index_select(0, order)
            sorted_rows = row_ids.index_select(0, order)
            sorted_lens = [int(x) for x in all_lens.index_select(0, order).tolist()]
            for begin, end, chunk_len in CachedContrastiveKLDiv._greedy_chunks(
                sorted_lens, row_cap, token_budget, waste_cap=1.5
            ):
                cols = sorted_cols[begin:end]
                rows = sorted_rows[begin:end]
                grouped = torch.argsort(cols, stable=True)
                self.chunks.append((begin, end, chunk_len))
                self.chunk_cols.append(cols.index_select(0, grouped))
                self.chunk_rows.append(rows.index_select(0, grouped))
            self.num_columns = len(sentence_features)
            self.col_max = [
                int(l.max().item()) if l.numel() else 0 for l in self.col_lengths
            ]

    @staticmethod
    def _all_gather_padded(tensor: torch.Tensor, with_gradients: bool):
        """Cross-rank all_gather for tensors whose dim-1 (sequence) differs by rank."""
        if get_world_size() == 1:
            return [tensor]
        local_len = torch.tensor([tensor.size(1)], device=tensor.device)
        max_len = int(torch.cat(all_gather(local_len)).max().item())
        if tensor.size(1) < max_len:
            pad = [0] * (2 * (tensor.dim() - 1))
            pad[-1] = max_len - tensor.size(1)
            tensor = F.pad(tensor, pad)
        if with_gradients:
            return all_gather_with_gradients(tensor)
        return all_gather(tensor)

    @staticmethod
    def _greedy_chunks(
        sorted_lengths: list[int],
        row_cap: int,
        token_budget: int | None,
        waste_cap: float | None = None,
        min_tokens: int = 8192,
    ) -> list[tuple[int, int, int]]:
        """(begin, end, chunk_max_len) over a descending-length row order.

        A chunk grows while within row_cap and token_budget; with waste_cap set it
        additionally stops growing once padded/real exceeds the cap - unless the
        chunk is still below min_tokens padded (avoids tiny launch-bound chunks).
        """
        chunks = []
        n = len(sorted_lengths)
        begin = 0
        while begin < n:
            chunk_len = max(1, sorted_lengths[begin])
            real = chunk_len
            end = begin + 1
            while end < n and end - begin < row_cap:
                rows = end - begin + 1
                padded = rows * chunk_len
                if token_budget is not None and padded > token_budget:
                    break
                if (
                    waste_cap is not None
                    and padded >= min_tokens
                    and padded > waste_cap * (real + sorted_lengths[end])
                ):
                    break
                real += sorted_lengths[end]
                end += 1
            chunks.append((begin, end, chunk_len))
            begin = end
        return chunks

    def __init__(
        self,
        model,
        contrastive_temperature: float = 0.02,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        contrastive_weight: float = 1.0,
        kldiv_weight: float = 1.0,
        mini_batch_size: int = 8,
        defer_grad_sync: bool = True,
        score_bytes_budget: int = 512 << 20,
        chunk_token_budget: int | None = 65536,
        score_anchor_token_budget: int = 65536,
        rand_context: str = "auto",  # "auto" | "always" | "never"
    ) -> None:
        super().__init__()
        self.model = model
        self.contrastive_temperature = contrastive_temperature
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.contrastive_weight = contrastive_weight
        self.kldiv_weight = kldiv_weight
        self.mini_batch_size = mini_batch_size
        self.defer_grad_sync = defer_grad_sync
        self.score_bytes_budget = score_bytes_budget
        self.chunk_token_budget = chunk_token_budget
        self.score_anchor_token_budget = score_anchor_token_budget
        self.rand_context = rand_context

        self.cache: list[list[torch.Tensor]] | None = None
        self.random_states: list[list[RandContext | None]] | None = None
        self._needs_rand_context: bool | None = None

    # utils
    def _use_rand_context(self) -> bool:
        if self.rand_context == "always":
            return True
        if self.rand_context == "never":
            return False
        if self._needs_rand_context is None:
            module = self.model.module if hasattr(self.model, "module") else self.model
            has_dropout = any(
                isinstance(m, torch.nn.Dropout) and m.p > 0 for m in module.modules()
            )
            cfg = getattr(getattr(module, "_first_module", lambda: None)(), "auto_model", None)
            cfg = getattr(cfg, "config", None)
            if cfg is not None:
                for k, v in cfg.to_dict().items():
                    if k.endswith("dropout") and isinstance(v, (int, float)) and v > 0:
                        has_dropout = True
            self._needs_rand_context = has_dropout
        return self._needs_rand_context

    def _fused_plan(self, sentence_features: list[dict[str, torch.Tensor]]) -> _FusedPlan:
        # the token budget is the real bound; 4x mini_batch_size rows just keeps kernel sizes sane on short splits
        return self._FusedPlan(
            sentence_features,
            row_cap=max(1, self.mini_batch_size) * 4,
            token_budget=self.chunk_token_budget,
        )

    @staticmethod
    def _chunk_features(
        sentence_features: list[dict[str, torch.Tensor]],
        cols: torch.Tensor,
        rows: torch.Tensor,
        chunk_len: int,
    ) -> dict[str, torch.Tensor]:
        """Build one fused encoder chunk (rows grouped by column)."""
        keys = set(sentence_features[0].keys())
        for sf in sentence_features[1:]:
            keys &= set(sf.keys())
        parts: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
        for c in torch.unique_consecutive(cols).tolist():
            sel = rows[cols == c]
            sf = sentence_features[c]
            for k in keys:
                v = sf[k]
                if not (isinstance(v, torch.Tensor) and v.dim() >= 2):
                    continue
                piece = v.index_select(0, sel)
                if piece.size(1) >= chunk_len:
                    piece = piece[:, :chunk_len]
                else:
                    piece = F.pad(piece, (0, chunk_len - piece.size(1)))
                parts[k].append(piece)
        return {k: torch.cat(v, dim=0) for k, v in parts.items() if v}

    # encoder chunked pass
    def embed_minibatch_iter(
        self,
        sentence_features: list[dict[str, torch.Tensor]],
        with_grad: bool,
        copy_random_state: bool,
        random_states: list[RandContext | None] | None = None,
    ) -> Iterator[tuple[torch.Tensor, RandContext | None]]:
        plan = self._fused_plan(sentence_features)
        use_rand = self._use_rand_context()
        for i, (begin, end, chunk_len) in enumerate(plan.chunks):
            chunk = self._chunk_features(
                sentence_features, plan.chunk_cols[i], plan.chunk_rows[i], chunk_len
            )
            random_state = None if random_states is None else random_states[i]
            grad_context = contextlib.nullcontext if with_grad else torch.no_grad
            random_state_context = (
                contextlib.nullcontext() if random_state is None else random_state
            )
            with random_state_context:
                with grad_context():
                    new_state = (
                        RandContext(*chunk.values())
                        if copy_random_state and use_rand
                        else None
                    )
                    embeddings = F.normalize(
                        self.model(chunk)["token_embeddings"], p=2, dim=-1
                    )
            yield embeddings, new_state

    def _assemble_columns(
        self, reps: list[torch.Tensor], plan: _FusedPlan
    ) -> list[torch.Tensor]:
        """Rebuild per-column (n_rows, col_max_len, h) tensors in original row order."""
        pieces: list[list[torch.Tensor]] = [[] for _ in range(plan.num_columns)]
        piece_rows: list[list[torch.Tensor]] = [[] for _ in range(plan.num_columns)]
        for chunk_emb, cols, rows in zip(reps, plan.chunk_cols, plan.chunk_rows):
            for c in torch.unique_consecutive(cols).tolist():
                sel = (cols == c).nonzero(as_tuple=True)[0]
                target = plan.col_max[c]
                piece = chunk_emb.index_select(0, sel)
                if piece.size(1) < target:
                    piece = F.pad(piece, (0, 0, 0, target - piece.size(1)))
                else:
                    piece = piece[:, :target]
                pieces[c].append(piece)
                piece_rows[c].append(rows.index_select(0, sel))
        out = []
        for c in range(plan.num_columns):
            emb = torch.cat(pieces[c], dim=0)
            rows_c = torch.cat(piece_rows[c])
            inverse = torch.empty_like(rows_c)
            inverse[rows_c] = torch.arange(rows_c.size(0), device=rows_c.device)
            out.append(emb.index_select(0, inverse))
        return out

    # loss
    def calculate_loss_and_cache_gradients(
        self,
        reps: list[list[torch.Tensor]],
        masks: list[torch.Tensor],
        labels: torch.Tensor,
        plan: _FusedPlan,
    ) -> torch.Tensor:
        loss = self.calculate_loss(reps, masks, labels, plan, with_backward=True)
        loss = loss.detach().requires_grad_()
        self.cache = [[r.grad for r in rs] for rs in reps]
        return loss

    def calculate_loss(
        self,
        reps: list[list[torch.Tensor]],
        masks: list[torch.Tensor],
        labels: torch.Tensor,
        plan: _FusedPlan,
        with_backward: bool = False,
    ) -> torch.Tensor:
        device = reps[0][0].device
        do_query_expansion = (
            self.model.do_query_expansion
            if hasattr(self.model, "do_query_expansion")
            else self.model.module.do_query_expansion
        )

        embeddings = self._assemble_columns(reps[0], plan)
        bs_local = embeddings[0].size(0)
        world = get_world_size()
        self_offset = get_rank() * bs_local

        shared_tensors: list[torch.Tensor] = []
        shared_leaves: list[torch.Tensor] = []

        def leaf_boundary(t: torch.Tensor) -> torch.Tensor:
            if not with_backward:
                return t
            shared_tensors.append(t)
            leaf = t.detach().requires_grad_()
            shared_leaves.append(leaf)
            return leaf

        # anchors: fold query mask, detach into a leaf
        anchor_mask = (
            masks[0][:, : plan.col_max[0]] if not do_query_expansion else None
        )
        anchor_emb = embeddings[0]
        if anchor_mask is not None:
            anchor_emb = anchor_emb * anchor_mask.unsqueeze(-1).to(anchor_emb.dtype)
            q_denom_full = anchor_mask.sum(dim=-1).clamp(min=1).unsqueeze(1).to(anchor_emb.dtype)
            anchor_lens = anchor_mask.sum(dim=1)
        else:
            q_denom_full = torch.full(
                (bs_local, 1), anchor_emb.size(1), device=device, dtype=anchor_emb.dtype
            )
            anchor_lens = torch.full((bs_local,), anchor_emb.size(1), device=device)
        anchor_emb = leaf_boundary(anchor_emb)

        # documents: fold masks, gather, length-sort, detach into leaves; docs below the column max clamp at >= 0
        doc_groups = []  # (leaf, lens_sorted, inv_perm, clamp_sorted)
        for emb, mask, col_max in zip(embeddings[1:], masks[1:], plan.col_max[1:]):
            folded = emb * mask[:, :col_max].unsqueeze(-1).to(emb.dtype)
            gathered = torch.cat(self._all_gather_padded(folded, with_gradients=True))
            lens = torch.cat(all_gather(mask[:, :col_max].sum(dim=1)))
            t_max = gathered.size(1)

            perm = torch.argsort(lens, descending=True, stable=True)
            inv_perm = torch.empty_like(perm)
            inv_perm[perm] = torch.arange(perm.size(0), device=device)
            emb_sorted = gathered.index_select(0, perm)
            lens_sorted = [int(x) for x in lens.index_select(0, perm).tolist()]
            clamp_sorted = torch.tensor([l < t_max for l in lens_sorted], device=device)

            doc_groups.append(
                (leaf_boundary(emb_sorted), lens_sorted, inv_perm, clamp_sorted)
            )

        use_kl = labels is not None and self.kldiv_weight > 0
        if use_kl:
            teacher_scores = labels.to(device=device, dtype=anchor_emb.dtype)
            teacher_log_probs = F.log_softmax(
                teacher_scores / self.teacher_temperature, dim=-1
            )

        contrastive_total = torch.zeros((), device=device)
        kl_total = torch.zeros((), device=device)

        # anchor scoring chunks: length-sorted and token-budgeted, so long-query splits get short tiles
        a_order = torch.argsort(anchor_lens, descending=True, stable=True)
        a_lens_sorted = [int(x) for x in anchor_lens.index_select(0, a_order).tolist()]
        for begin, end, s_len in self._greedy_chunks(
            a_lens_sorted,
            row_cap=self.mini_batch_size,
            token_budget=self.score_anchor_token_budget,
        ):
            orig_rows = a_order[begin:end]
            a_emb = anchor_emb.index_select(0, orig_rows)[:, :s_len]
            a_size = a_emb.size(0)
            q_denom = q_denom_full.index_select(0, orig_rows)

            per_group_scores = []
            for leaf, lens_sorted, inv_perm, clamp_sorted in doc_groups:
                # doc pieces sized from the TRUE tile dims so the byte budget holds, with a floor of 1 doc
                b_total = leaf.size(0)
                score_pieces = []
                piece_bounds = []
                start = 0
                while start < b_total:
                    t_piece = max(1, lens_sorted[start])
                    budget_docs = self.score_bytes_budget // max(
                        1, a_size * s_len * t_piece * leaf.element_size()
                    )
                    end_ = min(b_total, start + max(1, budget_docs))
                    piece_bounds.append((start, end_, t_piece))
                    start = end_
                for g_start, g_end, t_piece in piece_bounds:
                    sim = torch.einsum(
                        "ash,bth->abst", a_emb, leaf[g_start:g_end, :t_piece]
                    )
                    piece_max = sim.max(dim=-1).values  # (a, p, s)
                    clamp = clamp_sorted[g_start:g_end]
                    if bool(clamp.any()):
                        piece_max = torch.where(
                            clamp.view(1, -1, 1), piece_max.clamp_min(0), piece_max
                        )
                    score_pieces.append(piece_max.sum(dim=-1))
                scores_sorted = torch.cat(score_pieces, dim=1)  # (a, B_total)
                # divide by the query length -> MeanMaxSim, independent of query length
                per_group_scores.append(
                    scores_sorted.index_select(1, inv_perm) / q_denom
                )

            chunk_scores = torch.cat(per_group_scores, dim=1)
            targets = self_offset + orig_rows
            ce_sum = F.cross_entropy(
                chunk_scores / self.contrastive_temperature, targets, reduction="sum"
            ) * world
            chunk_loss = self.contrastive_weight * ce_sum

            kl_sum = None
            if use_kl:
                student_scores = torch.stack(
                    [g.gather(1, targets.unsqueeze(1)).squeeze(1) for g in per_group_scores],
                    dim=1,
                )
                student_log_probs = F.log_softmax(
                    student_scores / self.student_temperature, dim=-1
                )
                kl_sum = F.kl_div(
                    student_log_probs,
                    teacher_log_probs.index_select(0, orig_rows),
                    reduction="sum",
                    log_target=True,
                )
                chunk_loss = chunk_loss + self.kldiv_weight * kl_sum

            if with_backward:
                (chunk_loss / bs_local).backward()
                ce_sum = ce_sum.detach()
                kl_sum = kl_sum.detach() if kl_sum is not None else None
            contrastive_total = contrastive_total + ce_sum
            if kl_sum is not None:
                kl_total = kl_total + kl_sum

        if with_backward and shared_tensors:
            # one traversal of each shared path (assembly, mask folds, gather collectives) with the accumulated grads
            torch.autograd.backward(
                tensors=shared_tensors,
                grad_tensors=[leaf.grad for leaf in shared_leaves],
            )

        total = self.contrastive_weight * contrastive_total / bs_local
        if use_kl:
            total = total + self.kldiv_weight * kl_total / bs_local
        return total

    # forward
    def forward(
        self,
        sentence_features: Iterable[dict[str, torch.Tensor]],
        labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        sentence_features = list(sentence_features)

        skiplist = (
            self.model.skiplist
            if hasattr(self.model, "skiplist")
            else self.model.module.skiplist
        )
        masks = extract_skiplist_mask(
            sentence_features=sentence_features, skiplist=skiplist
        )
        plan = self._fused_plan(sentence_features)

        reps_mbs: list[torch.Tensor] = []
        random_state_mbs: list[RandContext | None] = []
        for reps_mb, random_state in self.embed_minibatch_iter(
            sentence_features=sentence_features,
            with_grad=False,
            copy_random_state=True,
        ):
            reps_mbs.append(reps_mb.detach().requires_grad_())
            random_state_mbs.append(random_state)
        reps = [reps_mbs]
        self.random_states = [random_state_mbs]

        if torch.is_grad_enabled():
            loss = self.calculate_loss_and_cache_gradients(reps, masks, labels, plan)
            loss.register_hook(
                lambda grad_output: self._backward_hook(grad_output, sentence_features)
            )
        else:
            loss = self.calculate_loss(reps, masks, labels, plan)
        return loss

    # backward hook
    def _backward_hook(
        self, grad_output: torch.Tensor, sentence_features: list[dict[str, torch.Tensor]]
    ) -> None:
        assert self.cache is not None and self.random_states is not None
        grads = self.cache[0]
        total_chunks = len(grads)
        can_no_sync = self.defer_grad_sync and hasattr(self.model, "no_sync")
        with torch.enable_grad():
            chunk_iter = self.embed_minibatch_iter(
                sentence_features=sentence_features,
                with_grad=True,
                copy_random_state=False,
                random_states=self.random_states[0],
            )
            for i, grad_mb in enumerate(grads):
                # DDP syncs on the last chunk only, the others accumulate locally (allreduce is linear)
                sync_ctx = (
                    self.model.no_sync()
                    if can_no_sync and i + 1 < total_chunks
                    else contextlib.nullcontext()
                )
                with sync_ctx:
                    reps_mb, _ = next(chunk_iter)
                    surrogate = (
                        torch.dot(reps_mb.flatten(), grad_mb.flatten()) * grad_output
                    )
                    surrogate.backward()

class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_at_step: int):
        self.stop_at_step = stop_at_step

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.stop_at_step:
            print(f"\n Reached target step {self.stop_at_step}. Stopping training...")
            control.should_training_stop = True
        return control


class ColBERTCollatorSampleNeg:
    """Collator for ColBERT that randomly samples a subset of negative columns per batch.

    ``teacher_scores`` is a per-row list aligned to ``[positive, negative_0, ...]``, so
    after sampling k negatives the matching scores are re-gathered in the sampled order
    and emitted as ``batch["label"]`` for the KL-div term.
    """

    def __init__(
        self,
        tokenize_fn: Callable,
        valid_label_columns: list[str] | None = None,
        num_negatives: int = 7,
    ) -> None:
        self.tokenize_fn = tokenize_fn
        self.num_negatives = num_negatives
        self.valid_label_columns = valid_label_columns or ["teacher_scores", "label", "scores"]

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        batch = {"return_loss": True}
        columns = list(features[0].keys())

        if "dataset_name" in columns:
            columns.remove("dataset_name")
            batch["dataset_name"] = features[0]["dataset_name"]

        # detected but not materialized yet: the scores must follow the sampled order
        label_column = next(
            (c for c in self.valid_label_columns if c in columns), None
        )
        if label_column is not None:
            columns.remove(label_column)

        negative_columns = [col for col in columns if col.startswith("negative_")]
        other_columns = [col for col in columns if not col.startswith("negative_")]

        if self.num_negatives is not None and negative_columns:
            k = min(self.num_negatives, len(negative_columns))
            sampled_negatives = random.sample(negative_columns, k)
        else:
            sampled_negatives = negative_columns
        columns_to_process = other_columns + sampled_negatives

        if label_column is not None:
            if isinstance(features[0][label_column], list):
                # negative_i -> teacher_scores[i + 1]; index 0 is the positive
                indices = [0] + [int(c.split("_")[1]) + 1 for c in sampled_negatives]
                batch["label"] = torch.tensor(
                    [[row[label_column][i] for i in indices] for row in features],
                    dtype=torch.float,
                )
            else:
                batch["label"] = torch.tensor([row[label_column] for row in features])

        for column in columns_to_process:
            is_query = "query" in column or "anchor" in column
            texts = [row[column] for row in features]
            if isinstance(texts[0], list):
                texts = list(itertools.chain(*texts))
            # pad=False: pad to the batch max, not to the full 8192 token budget
            tokenized = self.tokenize_fn(texts, is_query=is_query, pad=False)
            for key, value in tokenized.items():
                batch[f"{column}_{key}"] = value

        return batch
