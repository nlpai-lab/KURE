"""Late-interaction index backends for serving KURE-v2.

Every backend takes a list of per-document token-embedding matrices ([n_tokens, dim]
each, numpy or torch) plus doc ids, and answers top-k MaxSim searches. They mirror the
configurations benchmarked in the model card's Serving section:

* MaxSimIndex     exact exhaustive MaxSim over bf16 token vectors (quality ceiling)
* PlaidIndex      PLAID via fast-plaid: centroid + 4-bit residual compression with
                  centroid-driven candidate generation
* AsymBinaryIndex asymmetric binary quantization: documents keep only the sign bit per
                  dimension (packed, 32x smaller), queries stay full precision; scoring
                  is the exact dot of the query with the {-1,+1} signs, exhaustively
* PlaidBinaryIndex two-stage: PLAID (built from the original vectors) retrieves the
                  top candidates, which are re-scored with the exact asymmetric binary
                  MaxSim (full-precision query x 1-bit sign codes); only the sign codes
                  and the PLAID index are stored
* BinaryIvfIndex  same 1-bit document codes, but candidates come from a faiss binary
                  IVF (Hamming distance over inverted lists); only the top candidates
                  are re-scored with the exact asymmetric MaxSim, so the query is never
                  quantized in the score that ranks the output

Token pooling (hierarchical clustering of document tokens) happens at encode time via
pylate's `pool_factor`, so it composes with any backend here.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np
import torch


def _to_numpy_docs(doc_embeddings) -> list[np.ndarray]:
    return [
        e.detach().cpu().to(torch.float32).numpy() if isinstance(e, torch.Tensor)
        else np.asarray(e, dtype=np.float32)
        for e in doc_embeddings
    ]


def _maxsim(query: torch.Tensor, tokens: torch.Tensor, doc_of_tok: torch.Tensor,
            n_docs: int, chunk: int = 4_000_000) -> torch.Tensor:
    """Exact MaxSim of one query [Lq, D] against a flat token matrix [T, D].

    For each query token, take the max similarity within each document
    (scatter-reduce over token->doc ids), then sum over query tokens. Token
    chunking bounds the [Lq, chunk] similarity buffer on large corpora.
    """
    lq = query.shape[0]
    best = torch.full((lq, n_docs), float("-inf"), device=query.device, dtype=query.dtype)
    for start in range(0, tokens.shape[0], chunk):
        sim = query @ tokens[start:start + chunk].T                      # [Lq, chunk]
        idx = doc_of_tok[start:start + chunk].unsqueeze(0).expand(lq, -1)
        best.scatter_reduce_(1, idx, sim, reduce="amax")
    return best.sum(dim=0)                                               # [n_docs]


class MaxSimIndex:
    """Exhaustive exact MaxSim over bf16 token embeddings (no compression)."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._tokens = None
        self._doc_of_tok = None
        self._doc_ids: list[str] = []
        self.build_time_s = 0.0

    def build(self, doc_embeddings, doc_ids: list[str]) -> None:
        t0 = time.perf_counter()
        docs = _to_numpy_docs(doc_embeddings)
        lens = torch.tensor([d.shape[0] for d in docs])
        self._tokens = torch.from_numpy(np.vstack(docs)).to(self.device, torch.bfloat16)
        self._doc_of_tok = torch.repeat_interleave(torch.arange(len(docs)), lens).to(self.device)
        self._doc_ids = doc_ids
        self.build_time_s = time.perf_counter() - t0

    @torch.no_grad()
    def search(self, query_embeddings, k: int) -> list[tuple[list[str], list[float]]]:
        out = []
        for q in _to_numpy_docs(query_embeddings):
            qt = torch.from_numpy(q).to(self.device, torch.bfloat16)
            scores = _maxsim(qt, self._tokens, self._doc_of_tok, len(self._doc_ids))
            top = torch.topk(scores, min(k, len(self._doc_ids)))
            out.append(([self._doc_ids[i] for i in top.indices.tolist()],
                        [float(v) for v in top.values.tolist()]))
        return out

    @property
    def size_bytes(self) -> int:
        return self._tokens.numel() * 2  # bf16


class PlaidIndex:
    """PLAID (fast-plaid): centroid + 4-bit residual codes, centroid-pruned search."""

    def __init__(self, index_dir: str, device: str = "cuda", nbits: int = 4) -> None:
        self.index_dir = index_dir
        self.device = device
        self.nbits = nbits
        self._fp = None
        self._doc_ids: list[str] = []
        self.build_time_s = 0.0

    def build(self, doc_embeddings, doc_ids: list[str]) -> None:
        from fast_plaid.search import FastPlaid

        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        self._fp = FastPlaid(index=self.index_dir, device=self.device)
        embs = [torch.as_tensor(np.asarray(e, dtype=np.float32)) if not isinstance(e, torch.Tensor)
                else e.detach().cpu().to(torch.float32) for e in doc_embeddings]
        t0 = time.perf_counter()
        self._fp.create(documents_embeddings=embs, nbits=self.nbits)
        self.build_time_s = time.perf_counter() - t0
        self._doc_ids = doc_ids

    def search(self, query_embeddings, k: int) -> list[tuple[list[str], list[float]]]:
        queries = [torch.as_tensor(np.asarray(q, dtype=np.float32)) if not isinstance(q, torch.Tensor)
                   else q.detach().cpu().to(torch.float32) for q in query_embeddings]
        results = self._fp.search(queries_embeddings=queries, top_k=k, show_progress=False)
        return [([self._doc_ids[pid] for pid, _ in row], [float(s) for _, s in row])
                for row in results]

    @property
    def size_bytes(self) -> int:
        return sum(f.stat().st_size for f in Path(self.index_dir).rglob("*") if f.is_file())


class AsymBinaryIndex:
    """1-bit document signs (packed), full-precision queries, exhaustive exact scoring."""

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self._packed = None          # uint8 [T, ceil(D/8)] -- the stored index (32x smaller)
        self._signs = None           # ±1 bf16 [T, D] on device -- transient scoring buffer
        self._doc_of_tok = None
        self._doc_ids: list[str] = []
        self._dim = 0
        self.build_time_s = 0.0

    def build(self, doc_embeddings, doc_ids: list[str]) -> None:
        t0 = time.perf_counter()
        docs = _to_numpy_docs(doc_embeddings)
        lens = torch.tensor([d.shape[0] for d in docs])
        stacked = np.vstack(docs)
        self._dim = stacked.shape[1]
        self._packed = np.packbits(stacked >= 0, axis=1)
        bits = np.ascontiguousarray(np.unpackbits(self._packed, axis=1)[:, :self._dim])
        self._signs = (torch.from_numpy(bits).to(self.device, torch.bfloat16)
                       .mul_(2.0).sub_(1.0))
        self._doc_of_tok = torch.repeat_interleave(torch.arange(len(docs)), lens).to(self.device)
        self._doc_ids = doc_ids
        self.build_time_s = time.perf_counter() - t0

    @torch.no_grad()
    def search(self, query_embeddings, k: int) -> list[tuple[list[str], list[float]]]:
        out = []
        for q in _to_numpy_docs(query_embeddings):
            qt = torch.from_numpy(q).to(self.device, torch.bfloat16)
            scores = _maxsim(qt, self._signs, self._doc_of_tok, len(self._doc_ids))
            top = torch.topk(scores, min(k, len(self._doc_ids)))
            out.append(([self._doc_ids[i] for i in top.indices.tolist()],
                        [float(v) for v in top.values.tolist()]))
        return out

    @property
    def size_bytes(self) -> int:
        return int(self._packed.size)


class PlaidBinaryIndex:
    """PLAID candidate generation (from the original vectors) + 1-bit asymmetric rerank.

    fast-plaid retrieves the top `rerank_depth` candidates; only those are re-scored
    with the exact asymmetric MaxSim (full-precision query x the documents' ±1 sign
    codes). Stored index = PLAID index + packed 1-bit codes.
    """

    def __init__(self, index_dir: str, device: str = "cuda", nbits: int = 4,
                 rerank_depth: int = 1000) -> None:
        self.index_dir = index_dir
        self.device = device
        self.nbits = nbits
        self.rerank_depth = rerank_depth
        self._fp = None
        self._packed = None          # uint8 [T, ceil(D/8)] -- the 1-bit rerank codes
        self._offsets = None         # np.int64 [n_docs + 1]
        self._doc_ids: list[str] = []
        self._dim = 0
        self.build_time_s = 0.0

    def build(self, doc_embeddings, doc_ids: list[str]) -> None:
        from fast_plaid.search import FastPlaid

        Path(self.index_dir).mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        docs = _to_numpy_docs(doc_embeddings)
        self._dim = docs[0].shape[1]
        self._packed = np.packbits(np.vstack(docs) >= 0, axis=1)
        lens = np.array([d.shape[0] for d in docs], dtype=np.int64)
        self._offsets = np.insert(np.cumsum(lens), 0, 0)
        self._fp = FastPlaid(index=self.index_dir, device=self.device)
        self._fp.create(documents_embeddings=[torch.from_numpy(d) for d in docs],
                        nbits=self.nbits)
        self._doc_ids = doc_ids
        self.build_time_s = time.perf_counter() - t0

    @torch.no_grad()
    def search(self, query_embeddings, k: int) -> list[tuple[list[str], list[float]]]:
        depth = min(self.rerank_depth, len(self._doc_ids))
        queries = [torch.from_numpy(q) for q in _to_numpy_docs(query_embeddings)]
        res = self._fp.search(queries_embeddings=queries, top_k=depth, show_progress=False)
        out = []
        for q, row in zip(queries, res):
            cand = [pid for pid, _ in row]
            codes = np.concatenate([self._packed[self._offsets[c]:self._offsets[c + 1]]
                                    for c in cand])
            bits = np.unpackbits(codes, axis=1)[:, :self._dim]
            signs = (torch.from_numpy(np.ascontiguousarray(bits))
                     .to(self.device, torch.bfloat16).mul_(2.0).sub_(1.0))
            cand_lens = torch.tensor([int(self._offsets[c + 1] - self._offsets[c]) for c in cand])
            doc_of_tok = torch.repeat_interleave(torch.arange(len(cand)), cand_lens).to(self.device)
            qt = q.to(self.device, torch.bfloat16)
            scores = _maxsim(qt, signs, doc_of_tok, len(cand))
            top = torch.topk(scores, min(k, len(cand)))
            out.append(([self._doc_ids[cand[j]] for j in top.indices.tolist()],
                        [float(v) for v in top.values.tolist()]))
        return out

    @property
    def size_bytes(self) -> int:
        # honest total: PLAID candidate index + the 1-bit rerank codes
        plaid = sum(f.stat().st_size for f in Path(self.index_dir).rglob("*") if f.is_file())
        return plaid + int(self._packed.size)


class BinaryIvfIndex:
    """faiss binary-IVF Hamming candidates + exact asymmetric MaxSim rerank."""

    def __init__(self, device: str = "cuda", nprobe: int = 32, topk_tokens: int = 128,
                 rerank_depth: int = 1000, nlist: int | None = None,
                 train_max: int = 2_000_000) -> None:
        self.device = device
        self.nprobe = nprobe
        self.topk_tokens = topk_tokens
        self.rerank_depth = rerank_depth
        self.nlist = nlist
        self.train_max = train_max
        self._ivf = None
        self._packed = None          # uint8 [T, D/8] -- Hamming corpus + rerank source
        self._doc_of_tok = None      # np.int64 [T]
        self._offsets = None         # np.int64 [n_docs + 1]
        self._doc_ids: list[str] = []
        self._dim = 0
        self.build_time_s = 0.0

    def build(self, doc_embeddings, doc_ids: list[str]) -> None:
        import faiss

        t0 = time.perf_counter()
        docs = _to_numpy_docs(doc_embeddings)
        lens = np.array([d.shape[0] for d in docs], dtype=np.int64)
        stacked = np.vstack(docs)
        self._dim = stacked.shape[1]
        self._packed = np.ascontiguousarray(np.packbits(stacked >= 0, axis=1))
        self._offsets = np.insert(np.cumsum(lens), 0, 0)
        self._doc_of_tok = np.repeat(np.arange(len(docs), dtype=np.int64), lens)
        self._doc_ids = doc_ids

        n_tokens = self._packed.shape[0]
        nlist = self.nlist or int(min(65536, max(16, math.isqrt(n_tokens))))
        self._ivf = faiss.IndexBinaryIVF(faiss.IndexBinaryFlat(self._dim), self._dim, nlist)
        train = self._packed
        if n_tokens > self.train_max:  # sample for tractable IVF training on huge corpora
            sel = np.random.default_rng(0).choice(n_tokens, self.train_max, replace=False)
            train = np.ascontiguousarray(self._packed[sel])
        self._ivf.train(train)
        self._ivf.add(self._packed)
        self._ivf.nprobe = self.nprobe
        self.build_time_s = time.perf_counter() - t0

    @torch.no_grad()
    def search(self, query_embeddings, k: int) -> list[tuple[list[str], list[float]]]:
        depth = min(self.rerank_depth, len(self._doc_ids))
        topk = min(self.topk_tokens, self._packed.shape[0])
        out = []
        for q in _to_numpy_docs(query_embeddings):
            qpacked = np.ascontiguousarray(np.packbits(q >= 0, axis=1))
            _, idx = self._ivf.search(qpacked, topk)      # [Lq, topk] token ids (-1 padded)
            hits = idx.reshape(-1)
            docs = self._doc_of_tok[hits[hits >= 0]]
            # rank candidate docs by query-token hit count, keep the top `depth`
            uniq, counts = np.unique(docs, return_counts=True)
            cand = uniq[np.argsort(-counts)[:depth]]
            # exact asymmetric rerank: bf16 query MaxSim against the candidates' ±1 signs
            codes = np.concatenate([self._packed[self._offsets[c]:self._offsets[c + 1]]
                                    for c in cand])
            bits = np.unpackbits(codes, axis=1)[:, :self._dim]
            signs = (torch.from_numpy(np.ascontiguousarray(bits))
                     .to(self.device, torch.bfloat16).mul_(2.0).sub_(1.0))
            cand_lens = torch.tensor([int(self._offsets[c + 1] - self._offsets[c]) for c in cand])
            doc_of_tok = torch.repeat_interleave(torch.arange(len(cand)), cand_lens).to(self.device)
            qt = torch.from_numpy(q).to(self.device, torch.bfloat16)
            scores = _maxsim(qt, signs, doc_of_tok, len(cand))
            top = torch.topk(scores, min(k, len(cand)))
            out.append(([self._doc_ids[int(cand[j])] for j in top.indices.tolist()],
                        [float(v) for v in top.values.tolist()]))
        return out

    @property
    def size_bytes(self) -> int:
        return int(self._packed.size)  # the 1-bit token codes (the IVF stores the same bytes)
