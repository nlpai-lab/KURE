"""Min-max normalized variants of the KL distillation losses.

`label` in our data is a raw ``logit(yes) - logit(no)`` from the reranker teacher, so it
spans roughly -9..18 (mxbai) or -17..12 (Qwen3-4B), while the student's similarity sits on
whatever scale its embeddings happen to have. :class:`DistillKLDivLoss` softmaxes both at
the same temperature, so the two distributions can end up with very different sharpness for
reasons that have nothing to do with the ranking we want to distill.

The variants here min-max each row of both score vectors to [0, 1] first, which makes the KL
match the *shape* of the teacher distribution rather than its absolute scale. Same trick as
KURE-v2's dense distillation.

Because both vectors land in [0, 1], a temperature of 1.0 leaves an almost uniform
distribution over the 11 candidates -- these losses want a temperature well below 1 (0.05-0.2)
to recover the sharpness the raw logits had.

Two consequences to budget for when switching a run over:

- The inherited ``* temperature**2`` rescale then shrinks the loss by ~100x at T=0.1, so the
  learning rate has to go up by roughly as much to take the same size step.
- The reported loss is not comparable across the two variants. On a 3-step SPLADE smoke test
  ``kldiv`` sat at 6.7e3 with a 1.5e6 gradient norm while ``kldiv_minmax`` sat at 0.03 with a
  gradient norm of 13 -- same data, same model, entirely different scale.
"""

from __future__ import annotations

import contextlib

import torch
from torch import Tensor, nn

from sentence_transformers import util
from sentence_transformers.sparse_encoder.losses import (
    CachedSpladeLoss,
    SparseDistillKLDivLoss,
    SparseMultipleNegativesRankingLoss,
    SpladeLoss,
)


class L1Regularizer(nn.Module):
    """Mean per-row L1 norm, a drop-in for SpladeLoss's FlopsLoss regularizer.

    FLOPS squares the batch mean of each dim, so a weight's gradient scales with how many rows of
    the batch share that dim: a dim only one row uses gets ~1/B of the push a shared one does. The
    idiosyncratic low-weight dims a long document accumulates are exactly those, which is why FLOPS
    leaves long documents dense. L1 pushes every active weight equally (1/B), so a row's penalty
    grows with its own size. MILCO (arXiv:2510.00671) trains with L1 on both sides.
    """

    def compute_loss_from_embeddings(self, embeddings: Tensor) -> Tensor:
        return embeddings.abs().sum(dim=1).mean()


def _row_minmax(scores: Tensor) -> Tensor:
    """Map each row to [0, 1]. The epsilon keeps a constant row from dividing by zero."""
    lo = scores.min(dim=1, keepdim=True).values
    hi = scores.max(dim=1, keepdim=True).values
    return (scores - lo) / (hi - lo + 1e-8)


class _RowMinMaxKLDivMixin:
    """Overrides the parent's score preparation with a per-row min-max rescale.

    Mixed in ahead of the loss class so ``compute_loss_from_embeddings`` wins over the
    parent's. Everything else -- ``similarity_fct``, ``temperature``, ``loss_fct``,
    ``get_config_dict`` -- is inherited unchanged.
    """

    _row_minmax = staticmethod(_row_minmax)

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> Tensor:
        embeddings_query = embeddings[0]
        student_scores = torch.stack(
            [self.similarity_fct(embeddings_query, other) for other in embeddings[1:]],
            dim=1,
        )
        # float() because a bf16 min/max spread would quantize the normalized scores badly.
        student_scores = self._row_minmax(student_scores.float()) / self.temperature
        teacher_scores = self._row_minmax(labels.float()) / self.temperature

        loss = self.loss_fct(
            torch.log_softmax(student_scores, dim=1),
            torch.softmax(teacher_scores, dim=1),
        )
        return loss * (self.temperature**2)


class MinMaxSparseDistillKLDivLoss(_RowMinMaxKLDivMixin, SparseDistillKLDivLoss):
    """:class:`SparseDistillKLDivLoss` with per-row min-max normalized scores.

    Like its parent it must be wrapped in :class:`SpladeLoss`, which calls
    ``compute_loss_from_embeddings`` directly.
    """


class SparseContrastiveKLDivLoss(nn.Module):
    """In-batch contrastive loss and teacher KL over one shared encoder pass.

    Follows the mdenseon/mLateOn fine-tuning recipe, adapted from late interaction to SPLADE:

    - the contrastive term is a cross-entropy over every document in the batch, so the other
      queries' positives act as negatives on top of the mined ones;
    - the KL term is, per query, the divergence between the student distribution over its own
      ``(positive, negative_0, ...)`` and the teacher scores of those same documents.

    The two terms answer different questions. KL alone only ever ranks a query's own eleven
    candidates, so nothing pushes a document away from the *other* queries in the batch --
    which is exactly what retrieval over a million-passage corpus asks for, and where our
    KL-only run lost most ground.

    Student and teacher get separate softmax temperatures, unlike
    :class:`SparseDistillKLDivLoss` which shares one. A sharp student against a softer teacher
    distills the teacher's ranking rather than its margins. There is no ``temperature**2``
    rescale here: it corrects for a single shared temperature and is not meaningful once the
    two sides differ.

    Wrap in :class:`SpladeLoss` like any other sparse loss so FLOPS regularization still
    applies. Returns a dict so the two terms are logged separately; the trainer sums it.
    """

    def __init__(
        self,
        model,
        contrastive_weight: float = 1.0,
        kldiv_weight: float = 1.0,
        scale: float = 1.0,
        student_temperature: float = 1.0,
        teacher_temperature: float = 1.0,
        normalize_student: bool = False,
        normalize_teacher: bool = False,
        gather_across_devices: bool = False,
    ) -> None:
        super().__init__()
        self.model = model
        self.contrastive_weight = contrastive_weight
        self.kldiv_weight = kldiv_weight
        self.student_temperature = student_temperature
        self.teacher_temperature = teacher_temperature
        self.normalize_student = normalize_student
        self.normalize_teacher = normalize_teacher
        self.contrastive = SparseMultipleNegativesRankingLoss(
            model, scale=scale, gather_across_devices=gather_across_devices
        )

    def forward(self, sentence_features, labels):
        raise AttributeError(
            "SparseContrastiveKLDivLoss should not be used alone. Use it with SpladeLoss."
        )

    def compute_loss_from_embeddings(self, embeddings: list[Tensor], labels: Tensor) -> dict:
        losses = {}
        if self.contrastive_weight:
            contrastive = self.contrastive.compute_loss_from_embeddings(embeddings, labels)
            losses["contrastive_loss"] = self.contrastive_weight * contrastive

        if self.kldiv_weight:
            query = embeddings[0]
            student = torch.stack(
                [util.pairwise_dot_score(query, doc) for doc in embeddings[1:]], dim=1
            ).float()
            teacher = labels.float()
            # Per-row min-max to [0, 1] before the temperature, as in
            # MinMaxSparseDistillKLDivLoss: it matches the SHAPE of the distribution instead of its
            # absolute range, and normalizing both sides puts them on one scale. Note this
            # compresses the spread to 1, so the temperature has to come down with it.
            if self.normalize_student:
                student = _row_minmax(student)
            if self.normalize_teacher:
                teacher = _row_minmax(teacher)
            # log_target: both sides as log-probabilities is the numerically stabler form,
            # and the raw teacher logits span roughly -17..12.
            loss = torch.nn.functional.kl_div(
                torch.log_softmax(student / self.student_temperature, dim=1),
                torch.log_softmax(teacher / self.teacher_temperature, dim=1),
                reduction="batchmean",
                log_target=True,
            )
            losses["kldiv_loss"] = self.kldiv_weight * loss
        # Kept for ComponentLoggingCachedSpladeLoss, which has to re-derive them after
        # GradCache has already folded the dict into one number.
        self.last_components = {k: v.detach().item() for k, v in losses.items()}
        return losses

    def get_config_dict(self) -> dict:
        return {
            "contrastive_weight": self.contrastive_weight,
            "kldiv_weight": self.kldiv_weight,
            "student_temperature": self.student_temperature,
            "teacher_temperature": self.teacher_temperature,
            "normalize_student": self.normalize_student,
            "normalize_teacher": self.normalize_teacher,
            **self.contrastive.get_config_dict(),
        }


class ComponentLoggingCachedSpladeLoss(CachedSpladeLoss):
    """:class:`CachedSpladeLoss` that reports the inner loss's terms separately.

    The parent folds whatever the inner loss returns into a single ``base_loss``, so a
    contrastive/KL split disappears from the logs -- and that balance is the whole point of
    watching this run. The trainer sums the returned dict for the backward pass, so the
    components are subtracted back out of ``base_loss`` exactly as the parent already does
    for the regularizers; the total is unchanged and only ``base_loss`` shrinks to the
    rounding remainder.
    """

    def forward(self, sentence_features, labels=None):
        result = super().forward(sentence_features, labels)
        components = getattr(self.loss, "last_components", None)
        # In eval mode the parent already passes the inner dict through, so there is no
        # base_loss to re-partition and nothing to do.
        if not (isinstance(result, dict) and components and "base_loss" in result):
            return result
        for name, value in components.items():
            tensor = torch.tensor(value, device=result["base_loss"].device)
            result[name] = tensor
            result["base_loss"] = result["base_loss"] - tensor
        return result


def _chunk_bounds(lengths: list[int], budget: int) -> list[tuple[int, int]]:
    """Contiguous row ranges whose padded size, rows x widest row, stays under the budget.

    A lone row wider than the budget still gets a chunk of its own.
    """
    bounds, start, width = [], 0, 0
    for row, length in enumerate(lengths):
        if row > start and (row - start + 1) * max(width, length) > budget:
            bounds.append((start, row))
            start, width = row, 0
        width = max(width, length)
    bounds.append((start, len(lengths)))
    return bounds


class TokenBudgetSpladeLoss(ComponentLoggingCachedSpladeLoss):
    """Plain :class:`SpladeLoss` for a batch under the token budget, GradCache in chunks above it.

    The extended-length path (see length_grouping.py). Memory follows the padded tokens whose
    graph is alive at once: all twelve columns for the plain loss, one chunk for GradCache. So a
    batch whose columns total more than ``chunk_tokens`` padded tokens is embedded without
    gradients first, the loss and its gradient with respect to the embeddings are computed on the
    whole batch, and each chunk is then re-embedded with gradients and backpropagated through
    that cached gradient. Unlike slicing rows in the trainer this is exact for every term -- FLOPS
    squares the batch mean, and the contrastive loss reads the other rows -- at the price of a
    second forward pass, paid only by the wide batches.

    Chunks are rows of one column, trimmed to their own widest row (the collator pads a column
    to its longest row, and the tokenizer pads on the right).

    Under DDP only the final chunk's backward may synchronise: the ranks cut different batches
    into different numbers of chunks, so one all-reduce per backward would mismatch. The earlier
    chunks accumulate locally under ``no_sync`` and the last backward reduces the accumulated
    gradient, the ordinary gradient-accumulation contract. That leaves the forwards: DDP
    re-broadcasts buffers on a forward that follows a synchronised one, which the plain path
    does eleven times a step and this path never, so a step where the ranks take different paths
    hangs unless DDP is built with ``broadcast_buffers=False`` (train.py sets it). Verified on
    two GPUs in float64 with the ranks on different paths and different chunk counts: the
    synced gradient matches plain SpladeLoss to 1e-16 (3e-9 once chunks are trimmed narrower).
    """

    def __init__(self, model, chunk_tokens: int, sync_paths: bool = False, **kwargs) -> None:
        super().__init__(model, **kwargs)
        self.chunk_tokens = chunk_tokens
        self.sync_paths = sync_paths
        self._last_column = None

    def forward(self, sentence_features, labels=None):
        sentence_features = list(sentence_features)
        padded = sum(feature["attention_mask"].numel() for feature in sentence_features)
        over = padded > self.chunk_tokens
        if self.sync_paths and torch.distributed.is_initialized():
            # An inner loss that gathers in-batch negatives across ranks makes the ranks exchange
            # embedding gradients inside the loss. The plain path sends them already scaled by
            # the trainer (1 / accumulation steps); GradCache sends them unscaled and applies the
            # scale afterwards, through the hook. When the ranks of one step disagree on the path
            # every cross-rank term is mis-scaled -- measured on 2 GPUs: a 49% gradient error at
            # a loss scale of 0.25, exact (5e-9) once they agree. So one rank over budget puts
            # them all on GradCache.
            # Not labels.device: PFT pairs carry no labels.
            flag = torch.tensor(int(over), device=sentence_features[0]["attention_mask"].device)
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MAX)
            over = bool(flag)
        if not over:
            losses = SpladeLoss.forward(self, sentence_features, labels)
        else:
            self._last_column = sentence_features[-1]
            losses = super().forward(sentence_features, labels)
        # The trainer gathers the accumulated loss components across ranks key by key, in dict
        # order, and the two paths build the dict differently: GradCache lists the regularizers
        # first and adds a base_loss remainder the plain path lacks when the inner loss returns
        # its own components. Ranks take different paths within one step, so both paths hand
        # back the same keys in the same order.
        losses.setdefault("base_loss", next(iter(losses.values())).new_zeros(()))
        return dict(sorted(losses.items()))

    def embed_minibatch_iter(self, sentence_feature, with_grad, copy_random_state, random_states=None):
        lengths = sentence_feature["attention_mask"].sum(1).tolist()
        rows = len(lengths)
        bounds = _chunk_bounds(lengths, self.chunk_tokens)
        no_sync = getattr(self.model, "no_sync", contextlib.nullcontext)
        for n, (i, j) in enumerate(bounds):
            width = max(lengths[i:j])
            chunk = {
                key: value[i:j, :width] if isinstance(value, Tensor) and value.ndim == 2
                else value[i:j] if isinstance(value, Tensor) and value.ndim and value.shape[0] == rows
                else value
                for key, value in sentence_feature.items()
            }
            final = sentence_feature is self._last_column and n == len(bounds) - 1
            # The caller runs the chunk's backward while this generator is suspended at the
            # yield, so the context covers both the forward and the backward, as no_sync needs.
            with no_sync() if with_grad and not final else contextlib.nullcontext():
                yield self.embed_minibatch(
                    sentence_feature=chunk,
                    begin=0,
                    end=j - i,
                    with_grad=with_grad,
                    copy_random_state=copy_random_state,
                    random_state=None if random_states is None else random_states[n],
                )

    def get_config_dict(self):
        config = SpladeLoss.get_config_dict(self)
        config["chunk_tokens"] = self.chunk_tokens
        return config
