# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sentence-transformers==6.0.0",
#     "fast-plaid==1.6.0.280",
#     "faiss-cpu==1.15.0",
#     "flash-maxsim==0.3.0",
#     "datasets",
#     "torch==2.8.0",
#     "mteb>=2.19",
#     "setproctitle",
# ]
# ///
# Versions are pinned to the exact stack behind the KURE-v2 model-card figures
# (torch 2.8.0 cu128 wheels also run on CUDA 12.x drivers). flash-maxsim is the
# Triton MaxSim kernel used for the exhaustive and rerank scoring there; on CPU
# the indexes fall back to plain torch.
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
    parser.add_argument("--dataset", default=None,
                        help="BeIR-style HF dataset with corpus/queries/default configs; "
                             "overrides --task when given.")
    parser.add_argument("--task", default="AutoRAGRetrieval",
                        help="MTEB retrieval task name (e.g. Ko-StrategyQA, BelebeleRetrieval); "
                             "loads via mteb with the Korean subset.")
    parser.add_argument("--split", default=None,
                        help="Eval split for --task (default: the task's first eval split).")
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


def _align(corpus: dict, queries: dict, qrels: dict, limit: int | None):
    """Drop qrels pointing outside the corpus, queries without qrels; apply --limit."""
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
    return _align(corpus, queries, qrels, limit)


# Korean hf_subset where an MTEB task is multilingual.
TASK_SUBSETS = {
    "BelebeleRetrieval": "kor_Hang-kor_Hang",
    "PublicHealthQA": "korean",
    "MIRACLRetrieval": "ko",
    "MultiLongDocRetrieval": "ko",
    "MrTidyRetrieval": "korean",
}


def _as_text(row) -> str:
    if isinstance(row, str):
        return row
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def _to_id_map(holder) -> dict[str, str]:
    if isinstance(holder, dict):
        return {str(doc_id): _as_text(value) for doc_id, value in holder.items()}
    return {str(row["id"]): _as_text(row) for row in holder}


def load_mteb_task(name: str, split: str | None, limit: int | None):
    """Load one MTEB retrieval task (Korean subset). Returns (corpus, queries, qrels, split).

    mteb exposes retrieval data two ways: everything under task.dataset[subset][split],
    or task.corpus/queries/relevant_docs keyed the same way. Try both.
    """
    import mteb

    task = mteb.get_tasks(tasks=[name], languages=["kor"])[0]
    task.load_data()
    split = split or task.metadata.eval_splits[0]
    subset = TASK_SUBSETS.get(name)

    if getattr(task, "dataset", None):
        key = subset if (subset and subset in task.dataset) else next(iter(task.dataset))
        data = task.dataset[key][split]
        corpus_ds, queries_ds, relevant = data["corpus"], data["queries"], data["relevant_docs"]
    else:
        def pick(holder):
            key = subset if (subset and subset in holder) else next(iter(holder))
            inner = holder[key]
            return inner[split] if split in inner else next(iter(inner.values()))

        corpus_ds, queries_ds, relevant = (pick(task.corpus), pick(task.queries),
                                           pick(task.relevant_docs))

    corpus = _to_id_map(corpus_ds)
    queries = _to_id_map(queries_ds)
    qrels = {str(q): {str(d): int(s) for d, s in docs.items() if s > 0}
             for q, docs in relevant.items()}
    qrels = {q: docs for q, docs in qrels.items() if docs}
    corpus, queries, qrels = _align(corpus, queries, qrels, limit)
    return corpus, queries, qrels, split


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
    import faulthandler

    from setproctitle import setproctitle

    faulthandler.enable()  # stack trace on native crashes (SIGSEGV/SIGABRT)
    args = parse_args()
    # The index work runs on the GPU; the CPU side is many tiny torch/rayon ops. With the
    # default pools (one thread per core) those ops spin-wait, and on a shared box under
    # load a one-minute run can stall for tens of minutes. A small fixed pool keeps the
    # batch-1 latency stable; must be set before torch is imported.
    import os

    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"):
        os.environ.setdefault(var, "8")
    logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s", level=logging.INFO)
    # visible in ps/top/nvidia-smi so others on a shared box know a latency
    # benchmark owns this GPU
    setproctitle("Latency를 재고 있습니다. 프로세스 올리지 말아주십쇼 ㅠㅠ")

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    import torch
    from sentence_transformers import MultiVectorEncoder
    from sentence_transformers.multi_vector_encoder.modules import HierarchicalTokenPooling

    script_dir = Path(__file__).resolve().parent
    if args.index_dir is None:
        args.index_dir = str(script_dir / "index")
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.dataset:
        corpus, queries, qrels = load_beir(args.dataset, args.qrels_split, args.limit)
        source, source_split = args.dataset, args.qrels_split
    else:
        corpus, queries, qrels, split = load_mteb_task(args.task, args.split, args.limit)
        source, source_split = args.task, split
    logger.info("%s [%s]: %d documents, %d queries", source, source_split,
                len(corpus), len(queries))

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = MultiVectorEncoder(
        args.model, device=device, trust_remote_code=True,
        model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
    )
    logger.info("model ready on %s", device)
    doc_ids = list(corpus)

    # Document embeddings are cached per (model, source, split, pool, limit) so the
    # index backends can be swapped without re-encoding the corpus. fp16 storage is
    # lossless for bf16 values in the embeddings' range.
    tag = f"{args.model.split('/')[-1]}__{source.split('/')[-1]}__{source_split}"
    tag += f".pool{args.pool}" + (f".limit{args.limit}" if args.limit else "")
    cache = script_dir / "cache" / f"{tag}.npz"
    doc_embs = None
    if cache.exists():
        z = np.load(cache)
        if list(z["doc_ids"]) == doc_ids:
            offs = z["offsets"]
            # bind once: NpzFile re-reads the whole member on every __getitem__
            tokens = z["tokens"]
            doc_embs = [tokens[offs[i]:offs[i + 1]] for i in range(len(offs) - 1)]
            logger.info("document embeddings from cache: %s", cache.name)
    if doc_embs is None:
        encode_kwargs = {"batch_size": args.batch_size, "show_progress_bar": True}
        if args.pool > 1:
            encode_kwargs["token_pooling"] = HierarchicalTokenPooling(pool_factor=args.pool)
        doc_embs = model.encode_document([corpus[d] for d in doc_ids], **encode_kwargs)
        doc_embs = [t.cpu().to(torch.float16).numpy() for t in doc_embs]
        cache.parent.mkdir(parents=True, exist_ok=True)
        lens = np.array([e.shape[0] for e in doc_embs], dtype=np.int64)
        np.savez(cache, tokens=np.vstack(doc_embs), offsets=np.insert(np.cumsum(lens), 0, 0),
                 doc_ids=np.array(doc_ids))
        logger.info("document embeddings cached: %s", cache.name)

    index = build_index(args, device)
    logger.info("building %s index over %d documents", args.index, len(doc_ids))
    index.build(doc_embs, doc_ids)
    logger.info("index=%s built in %.1fs, %.1f MB",
                args.index, index.build_time_s, index.size_bytes / 1e6)

    qids = list(queries)
    q_texts = [queries[q] for q in qids]
    q_embs = model.encode_query(q_texts, batch_size=args.batch_size, show_progress_bar=False)

    # Quality: retrieve for every query at once.
    results = index.search(q_embs, args.k)
    ndcg = ndcg_at_k(results, qids, qrels, args.k)

    # Latency, benchmark protocol: serving one query is encode + search at batch 1.
    # Search: 10 warmup queries, then every task query timed once.
    # Encoding: 5 warmup runs, then 50 timed runs (cycling over the task queries).
    encode_ms, search_ms = [], []
    for emb in q_embs[:10]:
        index.search([emb], args.k)
    for emb in q_embs:
        t0 = time.perf_counter()
        index.search([emb], args.k)
        search_ms.append((time.perf_counter() - t0) * 1000)
    for text in (q_texts * 2)[:5]:
        model.encode_query([text], show_progress_bar=False)
    for i in range(50):
        text = q_texts[i % len(q_texts)]
        t0 = time.perf_counter()
        model.encode_query([text], show_progress_bar=False)
        encode_ms.append((time.perf_counter() - t0) * 1000)

    def stats(xs):
        xs = sorted(xs)
        return sum(xs) / len(xs), xs[min(len(xs) - 1, int(round(0.95 * len(xs))) - 1)]

    enc_mean, enc_p95 = stats(encode_ms)
    sea_mean, sea_p95 = stats(search_ms)
    pool = f" + pool x{args.pool}" if args.pool > 1 else ""
    print(f"\n=== {args.model} | {args.index}{pool} | {source} [{source_split}] "
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
               else script_dir / "results" / f"{source.split('/')[-1]}.jsonl")
        out.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model": args.model, "dataset": source, "split": source_split,
            "index": args.index,
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
