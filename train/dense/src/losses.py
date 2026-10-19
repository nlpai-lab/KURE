"""Min-max normalized variants of the KL distillation losses.

`label` in the data is the teacher's own score, on the teacher's scale (a reranker's
``logit(yes) - logit(no)`` spans roughly -17..18, an embedding teacher's cosine a narrow slice of
[-1, 1]), while the student's similarity sits on whatever scale its embeddings happen to have. :class:`DistillKLDivLoss` softmaxes both at
the same temperature, so the two distributions can end up with very different sharpness for
reasons that have nothing to do with the ranking we want to distill.

The variants here min-max each row of both score vectors to [0, 1] first, which makes the KL
match the *shape* of the teacher distribution rather than its absolute scale. Same trick as
the ko-vdr-train distillation runs.

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

import random

import torch
from torch import Tensor, nn

from sentence_transformers.sentence_transformer.losses import (
    DistillKLDivLoss,
    GlobalOrthogonalRegularizationLoss,
    MatryoshkaLoss,
)
from sentence_transformers.sentence_transformer.losses.matryoshka import ForwardDecorator


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


class MinMaxDistillKLDivLoss(_RowMinMaxKLDivMixin, DistillKLDivLoss):
    """:class:`DistillKLDivLoss` with per-row min-max normalized student and teacher scores."""


class KLDivGORLoss(nn.Module):
    """KL distillation plus Global Orthogonal Regularization (GOR), from one forward pass.

    GOR (Zhang et al. 2017; EmbeddingGemma) pushes the embeddings of each input column apart
    within the batch: per column it penalizes the mean pairwise cosine and the second moment
    above 1/d. It reads no labels, so it applies to our (query, positive, negative_0..9) rows
    as-is. Defaults are EmbeddingGemma's: second-moment term only, summed over columns.

    Built to sit *inside* MatryoshkaLoss, which calls it once per dimension with the model's
    forward decorated to return cached, truncated embeddings. So GOR is applied at every
    Matryoshka dimension, with its 1/d target taken from that dimension, and no extra encoder
    pass is spent on it. It returns a scalar, which MatryoshkaLoss accumulates, but also records
    the two terms in ``last_components`` so ComponentLoggingMatryoshkaLoss can log them apart.

    ``gor_weight`` needs to match the KL variant's scale: kldiv_minmax sits around 0.03 per
    step, where a weight of ~1 is meaningful; raw kldiv sits in the thousands and swamps it.
    """

    def __init__(
        self,
        model,
        base_loss,
        gor_weight: float,
        mean_weight: float = 0.0,
        second_moment_weight: float = 1.0,
        aggregation: str = "sum",
    ) -> None:
        super().__init__()
        self.model = model
        self.base_loss = base_loss
        self.gor_weight = gor_weight
        self.gor_loss = GlobalOrthogonalRegularizationLoss(
            model, mean_weight=mean_weight, second_moment_weight=second_moment_weight,
            aggregation=aggregation,
        )
        self.last_components: dict[str, Tensor] = {}

    def forward(self, sentence_features, labels) -> Tensor:
        embeddings = [self.model(f)["sentence_embedding"] for f in sentence_features]
        loss = self.base_loss.compute_loss_from_embeddings(embeddings, labels)
        gor = self.gor_loss.compute_loss_from_embeddings(embeddings)
        gor_term = self.gor_weight * torch.stack(list(gor.values())).sum()
        # Kept for ComponentLoggingMatryoshkaLoss, which names the two terms in the logs. These
        # are the live graph tensors, not detached copies: the wrapper returns them for the
        # backward pass, so `kl + gor` has to be exactly what this returns.
        self.last_components = {"kl": loss, "gor": gor_term}
        return loss + gor_term

    def get_config_dict(self) -> dict:
        return {
            "base_loss": self.base_loss.__class__.__name__,
            "gor_weight": self.gor_weight,
            **self.gor_loss.get_config_dict(),
        }


class ComponentLoggingMatryoshkaLoss(MatryoshkaLoss):
    """MatryoshkaLoss that reports each dimension's contribution instead of one scalar.

    The parent sums the per-dimension losses into a single number, so a run can only log the
    total -- there is no way to see that dim 256 has stopped improving while 2048 still is.
    The trainer sums a returned dict before the backward pass and logs each key alongside
    `loss` (BaseTrainer.track_loss_components), so partitioning the same total into named
    parts changes no gradient; it only names them. Where the inner loss splits itself further,
    as KLDivGORLoss does into kl and gor, those names are carried through per dimension.

    The forward mirrors the parent's; it exists only to keep the parts instead of adding them.
    """

    def forward(self, sentence_features, labels) -> dict[str, Tensor]:
        if isinstance(self.loss, self.cached_losses):
            return self.loss(sentence_features, labels)

        original_forward = self.model.forward
        components: dict[str, Tensor] = {}
        try:
            decorated_forward = ForwardDecorator(original_forward)
            self.model.forward = decorated_forward

            dim_indices = range(len(self.matryoshka_dims))
            if 0 < self.n_dims_per_step < len(dim_indices):
                dim_indices = sorted(random.sample(list(dim_indices), self.n_dims_per_step))

            for idx in dim_indices:
                dim = self.matryoshka_dims[idx]
                weight = self.matryoshka_weights[idx]
                decorated_forward.set_dim(dim)
                dim_labels = labels
                if (
                    isinstance(labels, torch.Tensor)
                    and labels.ndim == 2
                    and labels.size(-1) == self.model.get_embedding_dimension()
                ):
                    dim_labels = labels[:, :dim]

                value = self.loss(sentence_features, dim_labels)
                parts = getattr(self.loss, "last_components", None)
                if parts:
                    for name, part in parts.items():
                        components[f"{name}_{dim}"] = weight * part
                else:
                    components[f"kl_{dim}"] = weight * value
        finally:
            self.model.forward = original_forward
        return components
