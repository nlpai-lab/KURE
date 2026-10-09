"""SpladePooling with the two structural fixes from Linkup's SparseUp (2026).

Vanilla SPLADE pools ``log(1 + ReLU(logits))`` over positions. On a ModernBERT-family MLM head
the logits sit far above zero (A.X-Encoder-base: ~23% of the 50k vocab is positive at every
token, content dims at 10-30, a noise tail in (0, 8)), so the pooled bag is essentially dense at
init (49.5k of 50k dims, q.d ~ 85k) and contrastive training on it collapses.

Ingredient #1, logit shift: ``log(1 + ReLU(logits - logit_shift))``. Softmax-invariant, so the
MLM geometry is untouched; it sinks the operating point so support starts sparse. Ingredient #2,
per-position top-k: each token keeps only its ``top_k`` largest vocab logits before pooling, a cap
on fan-out rather than on the bag's L0. Both are applied inside the parent's chunk loop, so peak
memory stays that of one ``chunk_size`` slice, not of the full [batch, seq, vocab] tensor.

Two extras for a PyLate-pre-finetuned body: ``prefix_token_ids`` are attended but not pooled
(their positions are masked out, as SparseUp does with [Q]/[D]), and ``masked_vocab_ids`` are
output dims never allowed to fire (the tied decoder rows of those same prefix tokens).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import torch
from sentence_transformers.sparse_encoder.modules import SpladePooling


def canonical_ids_for(tokenizer, min_chars: int = 1) -> list[int]:
    """Map every token id to a canonical term id: WordPiece "##" stripped, NFKC, lowercased.

    A.X is WordPiece, so a word-initial "계약" and the continuation "##계약" (as in 민사계약) are
    different output dims and never match lexically; ~51% of its 50k vocab collapses into 36k terms.
    Special and added tokens stay their own term. Canonical forms shorter than ``min_chars`` are not
    merged either -- min_chars=2 keeps one-syllable particles and endings (##가, ##는) apart from the
    unrelated standalone syllables (가, 는).
    """
    import unicodedata

    special = set(tokenizer.all_special_ids) | set(tokenizer.get_added_vocab().values())
    vocab = sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])
    term_of, ids = {}, []
    for token, token_id in vocab:
        form = unicodedata.normalize("NFKC", token.removeprefix("##")).lower().strip()
        key = (token_id,) if token_id in special or len(form) < max(min_chars, 1) else form
        ids.append(term_of.setdefault(key, len(term_of)))
    return ids


class ShiftedTopKSpladePooling(SpladePooling):
    # chunk_size too: SpladePooling does not persist it, so a reloaded model pools the whole
    # [batch, seq, vocab] tensor at once -- at 8192 x 50k that is what sets the evaluation batch.
    config_keys = [
        *SpladePooling.config_keys, "chunk_size", "logit_shift", "top_k", "prefix_token_ids", "masked_vocab_ids",
        "canonical_ids",
    ]

    def __init__(
        self,
        logit_shift: float = 0.0,
        top_k: int | None = None,
        prefix_token_ids: list[int] | None = None,
        masked_vocab_ids: list[int] | None = None,
        canonical_ids: list[int] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.logit_shift = float(logit_shift)
        self.top_k = int(top_k) if top_k else None
        self.prefix_token_ids = [int(t) for t in (prefix_token_ids or [])]
        self.masked_vocab_ids = [int(t) for t in (masked_vocab_ids or [])]
        # Vocabulary canonicalization: token id -> canonical term id (see canonical_ids_for). As in
        # Linkup's SparseUp (modeling_splade.py `_fold`), the pooled vector is folded by max per term
        # after the per-position top-k and the pooling, so "계약" and "##계약" are one output dim.
        self.canonical_ids = [int(c) for c in canonical_ids] if canonical_ids else None

    def forward(self, features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        logits = features["token_embeddings"]  # [batch, seq, vocab]
        mask = features["attention_mask"]
        if self.prefix_token_ids and "input_ids" in features:
            drop = torch.zeros_like(mask, dtype=torch.bool)
            for token_id in self.prefix_token_ids:
                drop |= features["input_ids"] == token_id
            mask = mask.masked_fill(drop, 0)
        mask = mask.unsqueeze(-1).to(logits.dtype)

        batch_size, seq_len, vocab_size = logits.shape
        if self.pooling_strategy == "max":
            pooled = torch.full((batch_size, vocab_size), float("-inf"), dtype=logits.dtype, device=logits.device)
        elif self.pooling_strategy == "sum":
            pooled = torch.zeros((batch_size, vocab_size), dtype=logits.dtype, device=logits.device)
        else:
            raise ValueError(f"Unsupported pooling_strategy: {self.pooling_strategy}")

        chunk = seq_len if (self.chunk_size is None or self.chunk_size <= 0) else self.chunk_size
        for i in range(0, seq_len, chunk):
            x = logits[:, i : i + chunk, :]
            if self.logit_shift:
                x = x - self.logit_shift
            if self.top_k:
                kth = x.topk(self.top_k, dim=-1).values[..., -1:]
                x = x.masked_fill(x < kth, 0.0)  # 0 -> ReLU -> log1p = 0: dropped dims stay silent
            if self.masked_vocab_ids:
                idx = torch.as_tensor(self.masked_vocab_ids, device=x.device)
                x = x.index_fill(-1, idx, 0.0)
            x = (x * mask[:, i : i + chunk, :]).relu().log1p()
            if self.activation_function == "log1p_relu":
                x = x.log1p()
            if self.pooling_strategy == "max":
                pooled = torch.maximum(pooled, x.max(dim=1).values)
            else:
                pooled = pooled + x.sum(dim=1)

        if self.canonical_ids is not None:
            canon = torch.as_tensor(self.canonical_ids, device=pooled.device).expand_as(pooled)
            folded = pooled.new_zeros((batch_size, int(canon.max()) + 1))
            pooled = folded.scatter_reduce(-1, canon, pooled, reduce="amax", include_self=False)

        if self.embedding_dimension is None:
            self.embedding_dimension = pooled.shape[1]
        features["sentence_embedding"] = pooled
        return features

    def save(self, output_path: str, *args, safe_serialization: bool = True, **kwargs) -> None:
        super().save(output_path, *args, safe_serialization=safe_serialization, **kwargs)
        # modules.json names this class as ``pooling.ShiftedTopKSpladePooling``, and ST resolves a
        # non-``sentence_transformers`` class through transformers' dynamic-module loader, which
        # looks for ``pooling.py`` in the model directory (or the Hub repo, with
        # trust_remote_code=True). Without it a checkpoint only loads where train/sparse/src is on
        # sys.path. ST does not copy custom module code itself, so the module ships its own
        # source next to modules.json -- ``output_path`` is the ``1_ShiftedTopKSpladePooling/``
        # subfolder, hence the parent.
        shutil.copyfile(Path(__file__), Path(output_path).resolve().parent / Path(__file__).name)
