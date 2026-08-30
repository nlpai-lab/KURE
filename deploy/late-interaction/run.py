# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers>=6.0",
#     "fast-plaid>=1.6.0",
#     "faiss-cpu>=1.12.0",
#     "datasets",
#     "torch>=2.8,<2.9",
# ]
# ///
# torch is pinned to the 2.8 line (cu128 wheels) to match the benchmark stack and to
# run on CUDA 12.x drivers; newer torch wheels require a CUDA 13 driver.
"""One-command serving experiment for KURE-v2.

Loads a Korean retrieval dataset (BeIR-style: corpus / queries / qrels), encodes it
with KURE-v2 (optionally pooling document tokens at encode time), builds the chosen
index backend, then reports quality (nDCG@10), index size, build time, and batch-1
latency (query encode + index search, the two serial stages of serving one query).

    uv run deploy/late-interaction/run.py --index plaid
    uv run deploy/late-interaction/run.py --index binary --pool 3
    uv run deploy/late-interaction/run.py --index plaid-binary --pool 3

Each run appends one JSON line to results/<dataset>.jsonl next to this script (see
--out). The inline metadata above makes uv run this in its own environment (KURE-v2
loads via sentence-transformers >= 6.0 MultiVectorEncoder), separate from the
project venv.

Backends: maxsim (exact, quality ceiling), plaid (n-bit PLAID), binary (asymmetric
binary quantization, exhaustive), plaid-binary (PLAID candidates + exact asymmetric
binary rerank), binary-ivf (Hamming candidates + exact rerank).
`--pool N` composes hierarchical token pooling (docs only) with any backend.
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nlpai-lab/KURE-v2")
    parser.add_argument("--index",
                        choices=("maxsim", "plaid", "binary", "plaid-binary", "binary-ivf"),
                        default="plaid")
    parser.add_argument("--pool", type=int, default=1,
                        help="Hierarchical token pooling factor for documents "
                             "(1 = off; 2/3 shrink the index ~2x/3x).")
    parser.add_argument("--dataset", default="yjoonjang/markers_bm",
                        help="BeIR-style HF dataset with corpus/queries/default configs.")
    parser.add_argument("--qrels-split", default="test")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32, help="Encode batch size.")
    parser.add_argument("--device", default=None, help="Defaults to cuda when available.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Cap corpus/query count for a quick pass.")
    parser.add_argument("--index-dir", default=None,
                        help="Where PLAID writes its index files "
                             "(default: index/ next to this script, gitignored).")
    parser.add_argument("--out", default=None,
                        help="jsonl the result line is appended to (default: "
                             "results/<dataset>.jsonl next to this script; 'none' skips).")
    # backend knobs (benchmark defaults)
    parser.add_argument("--nbits", type=int, default=4, help="PLAID residual bits.")
    parser.add_argument("--nprobe", type=int, default=32, help="binary-ivf IVF lists probed.")
    parser.add_argument("--rerank-depth", type=int, default=1000,
                        help="plaid-binary / binary-ivf candidates re-scored exactly.")
    parser.add_argument("--topk-tokens", type=int, default=128,
                        help="binary-ivf doc tokens fetched per query token.")
    return parser.parse_args()


def load_beir(name: str, qrels_split: str, limit: int | None):
    """Return (corpus {id: text}, queries {id: text}, qrels {qid: {did: rel}})."""
    from datasets import load_dataset

    corpus_ds = load_dataset(name, "corpus", split="corpus")
    queries_ds = load_dataset(name, "queries", split="queries")
    qrels_ds = load_dataset(name, "default", split=qrels_split)

    corpus = {str(r["_id"]): (f"{r.get('title', '')} {r['text']}".strip() if r.get("title")
                              else r["text"]) for r in corpus_ds}
    queries = {str(r["_id"]): r["text"] for r in queries_ds}
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        qrels.setdefault(str(r["query-id"]), {})[str(r["corpus-id"])] = int(r.get("score", 1))

    if limit:
        doc_ids = list(corpus)[:limit]
        keep = set(doc_ids)
        corpus = {d: corpus[d] for d in doc_ids}
        qrels = {q: {d: s for d, s in docs.items() if d in keep} for q, docs in qrels.items()}
        qrels = {q: docs for q, docs in qrels.items() if docs}
        queries = {q: queries[q] for q in list(qrels)[:limit] if q in queries}
    qrels = {q: docs for q, docs in qrels.items() if q in queries}
    queries = {q: t for q, t in queries.items() if q in qrels}
    return corpus, queries, qrels


def ndcg_at_k(results, qids: list[str], qrels: dict, k: int) -> float:
    """Mean nDCG@k with linear gain (pytrec_eval convention)."""
    total = 0.0
    for qid, (doc_ids, _) in zip(qids, results):
        rels = qrels[qid]
        dcg = sum(rels.get(d, 0) / math.log2(rank + 2) for rank, d in enumerate(doc_ids[:k]))
        ideal = sum(rel / math.log2(rank + 2)
                    for rank, rel in enumerate(sorted(rels.values(), reverse=True)[:k]))
        total += dcg / ideal if ideal > 0 else 0.0
    return total / len(qids)


def build_index(args, device: str):
    from indexes import (
        AsymBinaryIndex,
        BinaryIvfIndex,
        MaxSimIndex,
        PlaidBinaryIndex,
        PlaidIndex,
    )

    if args.index == "maxsim":
        return MaxSimIndex(device=device)
    if args.index == "plaid":
        return PlaidIndex(index_dir=str(Path(args.index_dir) / "plaid"), device=device,
                          nbits=args.nbits)
    if args.index == "binary":
        return AsymBinaryIndex(device=device)
    if args.index == "plaid-binary":
        return PlaidBinaryIndex(index_dir=str(Path(args.index_dir) / "plaid-binary"),
                                device=device, nbits=args.nbits,
                                rerank_depth=args.rerank_depth)
    return BinaryIvfIndex(device=device, nprobe=args.nprobe, topk_tokens=args.topk_tokens,
                          rerank_depth=args.rerank_depth)


def main() -> None:
    args = parse_args()
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import torch
    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    script_dir = Path(__file__).resolve().parent
    if args.index_dir is None:
        args.index_dir = str(script_dir / "index")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    corpus, queries, qrels = load_beir(args.dataset, args.qrels_split, args.limit)
    logger.info("dataset=%s: %d documents, %d queries", args.dataset, len(corpus), len(queries))

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = MultiVectorEncoder(
        args.model, device=device, trust_remote_code=True,
        model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
    )
    doc_ids = list(corpus)
    encode_kwargs = {"batch_size": args.batch_size, "show_progress_bar": True}
    if args.pool > 1:
        encode_kwargs["token_pooling"] = HierarchicalTokenPooling(pool_factor=args.pool)
    doc_embs = model.encode_document([corpus[d] for d in doc_ids], **encode_kwargs)

    index = build_index(args, device)
    index.build(doc_embs, doc_ids)
    logger.info("index=%s built in %.1fs, %.1f MB",
                args.index, index.build_time_s, index.size_bytes / 1e6)

    qids = list(queries)
    q_texts = [queries[q] for q in qids]
    q_embs = model.encode_query(q_texts, batch_size=args.batch_size, show_progress_bar=False)

    # Quality: retrieve for every query at once.
    results = index.search(q_embs, args.k)
    ndcg = ndcg_at_k(results, qids, qrels, args.k)

    # Latency: serving one query is encode + search, run serially at batch 1.
    n_timed = min(len(qids), 100)
    encode_ms, search_ms = [], []
    for text, emb in list(zip(q_texts, q_embs))[:3]:  # warmup
        model.encode_query([text], show_progress_bar=False)
        index.search([emb], args.k)
    for text, emb in list(zip(q_texts, q_embs))[:n_timed]:
        t0 = time.perf_counter()
        model.encode_query([text], show_progress_bar=False)
        encode_ms.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        index.search([emb], args.k)
        search_ms.append((time.perf_counter() - t0) * 1000)

    def stats(xs):
        xs = sorted(xs)
        return sum(xs) / len(xs), xs[min(len(xs) - 1, int(round(0.95 * len(xs))) - 1)]

    enc_mean, enc_p95 = stats(encode_ms)
    sea_mean, sea_p95 = stats(search_ms)
    pool = f" + pool x{args.pool}" if args.pool > 1 else ""
    print(f"\n=== {args.model} | {args.index}{pool} | {args.dataset} "
          f"({len(corpus)} docs, {len(qids)} queries) ===")
    print(f"nDCG@{args.k}:            {ndcg:.4f}")
    print(f"index size:          {index.size_bytes / 1e6:.1f} MB (build {index.build_time_s:.1f}s)")
    print(f"query encode (bs=1): {enc_mean:.1f} ms mean / {enc_p95:.1f} ms p95")
    print(f"index search (bs=1): {sea_mean:.1f} ms mean / {sea_p95:.1f} ms p95")
    print(f"end-to-end:          {1000 / (enc_mean + sea_mean):.1f} QPS, "
          f"{enc_p95 + sea_p95:.1f} ms p95")

    if args.out != "none":
        import json

        out = (Path(args.out) if args.out
               else script_dir / "results" / f"{args.dataset.split('/')[-1]}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model, "dataset": args.dataset, "index": args.index,
            "pool": args.pool, "nbits": args.nbits, "nprobe": args.nprobe,
            "rerank_depth": args.rerank_depth, "topk_tokens": args.topk_tokens,
            "n_docs": len(corpus), "n_queries": len(qids), "k": args.k,
            f"ndcg@{args.k}": round(ndcg, 4),
            "index_size_mb": round(index.size_bytes / 1e6, 1),
            "build_s": round(index.build_time_s, 1),
            "encode_ms_mean": round(enc_mean, 1), "encode_ms_p95": round(enc_p95, 1),
            "search_ms_mean": round(sea_mean, 1), "search_ms_p95": round(sea_p95, 1),
            "qps_e2e": round(1000 / (enc_mean + sea_mean), 1),
            "p95_ms_e2e": round(enc_p95 + sea_p95, 1),
            "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        }
        with out.open("a") as f:
            f.write(json.dumps(record) + "\n")
        print(f"saved -> {out}")


if __name__ == "__main__":
    main()
