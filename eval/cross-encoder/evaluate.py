"""Evaluate a reranker (CrossEncoder) on the nine MTEB(kor, v2) retrieval tasks, reranking.

Follows instructkr/reranker-simple-benchmark: for each query the reranker scores {all gold docs}
union {BM25 top-50 negatives} (the benchmark's cached stage-1 pools, see fetch_eval_assets.sh), and
mteb 2.x computes NDCG@10. Injecting the golds means a gold the BM25 stage missed is still rankable,
so the score measures the reranker, not BM25 recall. Every model runs in bf16 at max_length
min(8192, the model's own limit), with flash_attention_2 where the architecture supports it.

--speed also measures inference throughput (pairs per second, PPS) on the pairs the NDCG pass scored;
see speed.py for the protocol. Run it on a GPU with no other process on it.

Results are written to results/<org>/<model>/<task>.json, one file per task as soon as it finishes
(a task whose file exists is skipped), the tree make_leaderboard.py reads.

Usage:
    bash eval/cross-encoder/fetch_eval_assets.sh            # once: the stage-1 BM25 pools
    uv run python eval/cross-encoder/evaluate.py nlpai-lab/KURE-Reranker-nano
    uv run python eval/cross-encoder/evaluate.py nlpai-lab/KURE-Reranker-base --speed
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import mteb  # noqa: E402
import torch  # noqa: E402
from mteb.abstasks.retrieval import AbsTaskRetrieval  # noqa: E402
from sentence_transformers.evaluation import SentenceEvaluator  # noqa: E402

HERE = Path(__file__).resolve().parent
POOL_DIR = HERE / "stage1_pools"
RESULTS = HERE / "results"
NEG_TOP_K = 50          # golds + BM25 top-50 negatives, per the benchmark
MAX_LENGTH = 8192       # the benchmark's common length, capped at each model's own limit
TASKS = ["Ko-StrategyQA", "AutoRAGRetrieval", "PublicHealthQA", "BelebeleRetrieval",
         "MIRACLRetrieval", "MrTidyRetrieval", "MultiLongDocRetrieval", "SQuADKorV1Retrieval", "LawIRKo"]
# Small-corpus tasks, cheap enough to run every few hundred training steps.
IN_TRAIN_TASKS = ["PublicHealthQA", "BelebeleRetrieval", "AutoRAGRetrieval", "LawIRKo", "SQuADKorV1Retrieval"]
# The metrics the benchmark reports; mteb emits ~150 per task.
REPORT_METRICS = ["ndcg_at_1", "ndcg_at_5", "ndcg_at_10", "map_at_10", "mrr_at_10", "recall_at_10"]

_pool_cache: dict[str, dict[str, list[str]]] = {}


def _load_pool(task_name: str) -> dict[str, list[str]]:
    if task_name not in _pool_cache:
        path = POOL_DIR / f"{task_name}_id.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} missing; run eval/cross-encoder/fetch_eval_assets.sh first")
        pool = {}
        for line in open(path):
            d = json.loads(line)
            pool[str(d["query_id"])] = [str(x) for x in d["relevance_ids"]]
        _pool_cache[task_name] = pool
    return _pool_cache[task_name]


def _ids(container) -> set:
    if hasattr(container, "keys"):
        return {str(k) for k in container.keys()}
    return {str(r.get("id") if hasattr(r, "get") else r) for r in container}


_orig_evaluate_subset = AbsTaskRetrieval._evaluate_subset


def _gold_inject_evaluate_subset(self, model, data_split, *, hf_split, hf_subset, **kwargs):
    """Set top_ranked = {all golds} u {BM25 top-NEG_TOP_K}, so reranking never misses a gold."""
    pool = _load_pool(self.metadata.name)
    corpus, queries, rels = data_split["corpus"], data_split["queries"], data_split["relevant_docs"]
    cids = _ids(corpus)
    qids = ([str(q) for q in queries.keys()] if hasattr(queries, "keys")
            else [str(r.get("id") if hasattr(r, "get") else r) for r in queries])
    top_ranked: dict[str, list[str]] = {}
    for qid in qids:
        golds = [g for g in (rels.get(qid, {}) if hasattr(rels, "get") else {}) if g in cids]
        gset = set(golds)
        top_ranked[qid] = golds + [d for d in pool.get(qid, [])[:NEG_TOP_K] if d not in gset and d in cids]
    data_split["top_ranked"] = top_ranked
    self._top_k = max((len(v) for v in top_ranked.values()), default=1) + 1
    return _orig_evaluate_subset(self, model, data_split, hf_split=hf_split, hf_subset=hf_subset, **kwargs)


AbsTaskRetrieval._evaluate_subset = _gold_inject_evaluate_subset


def get_task(task_name: str):
    tasks = mteb.get_tasks(tasks=[task_name], languages=["kor-Hang"]) or mteb.get_tasks(tasks=[task_name])
    return tasks[0]


def _metrics(res) -> dict[str, float]:
    """REPORT_METRICS averaged over the subsets/splits mteb stores one dict for (MLDR = dev+test)."""
    dicts = []

    def walk(o):
        if isinstance(o, dict):
            if "ndcg_at_10" in o:
                dicts.append(o)
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for x in o:
                walk(x)
        else:
            for attr in ("scores", "task_results", "results"):
                if hasattr(o, attr):
                    walk(getattr(o, attr))

    walk(res)
    out = {}
    for k in REPORT_METRICS:
        vals = [float(d[k]) for d in dicts if isinstance(d.get(k), (int, float))]
        if vals:
            out[k] = sum(vals) / len(vals)
    out["main_score"] = out.get("ndcg_at_10")
    return out


def score_task(model, task_name: str, batch_size: int) -> dict[str, float]:
    with torch.no_grad():
        res = mteb.evaluate(model, get_task(task_name), cache=None, overwrite_strategy="always",
                            show_progress_bar=False, encode_kwargs={"batch_size": batch_size})
    return _metrics(res)


class RerankNDCGEvaluator(SentenceEvaluator):
    """In-training NDCG@10 on IN_TRAIN_TASKS. Runs on every rank: guarding it to rank 0 desyncs DDP."""

    def __init__(self, tasks=IN_TRAIN_TASKS, batch_size: int = 64):
        super().__init__()
        self.tasks = tasks
        self.batch_size = batch_size
        self.primary_metric = "rerank_mean_ndcg@10"

    def __call__(self, model, output_path=None, epoch=-1, steps=-1) -> dict:
        was_training = model.model.training
        scores = {t: score_task(model, t, self.batch_size)["ndcg_at_10"] for t in self.tasks}
        if was_training:
            model.model.train()  # mteb left the model in eval()
        metrics = {f"rerank_{t}_ndcg@10": v for t, v in scores.items()}
        metrics["rerank_mean_ndcg@10"] = sum(scores.values()) / len(scores)
        logging.info(f"rerank eval @ step {steps}: {metrics}")
        return metrics


def load_model(model_name: str, device: str):
    """A sentence-transformers CrossEncoder in bf16, capped at min(MAX_LENGTH, its own limit)."""
    from sentence_transformers import CrossEncoder
    from speed import set_attention

    ce = CrossEncoder(model_name, trust_remote_code=True, device=device,
                      model_kwargs={"dtype": torch.bfloat16})
    # An absolute-position model (XLM-R, 514) forced to 8192 overflows its position ids.
    ce.max_length = min(MAX_LENGTH, ce.max_length or MAX_LENGTH)
    if ce.tokenizer.pad_token is None and ce.tokenizer.eos_token is not None:
        ce.tokenizer.pad_token = ce.tokenizer.eos_token  # causal-LM rerankers, batch > 1
    # Scoring is one forward, so a KV cache is useless and costs ~19 GB at bs16 x 8192.
    if getattr(ce.model.config, "use_cache", False):
        ce.model.config.use_cache = False
    attn = set_attention(ce.model)
    return ce, attn


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face model name or local path, e.g. nlpai-lab/KURE-Reranker-base")
    parser.add_argument("--tasks", nargs="+", default=TASKS, help="subset of tasks to run (default: all nine)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--speed", action="store_true", help="also measure PPS (see speed.py)")
    parser.add_argument("--speed-samples", type=int, default=512,
                        help="pairs per task for PPS, whole queries sampled from the scored pairs")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: results/<model> (a local path keeps its last two components)")
    parser.add_argument("--overwrite", action="store_true", help="re-run tasks whose json already exists")
    args = parser.parse_args()

    name = "/".join(Path(args.model).parts[-2:])
    out_dir = args.out_dir or RESULTS / name
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [t for t in args.tasks if args.overwrite or not (out_dir / f"{t}.json").exists()]
    if not todo:
        print(f"all tasks already in {out_dir}; --overwrite to re-run")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, attn = load_model(args.model, device)
    tap = None
    if args.speed:
        from speed import PairTap, speed_record
        tap = PairTap(model)
    for task_name in todo:
        if tap:
            tap.take()
            tap.on = True  # record the pairs NDCG scores, to time PPS on the same pairs
        scores = score_task(model, task_name, args.batch_size)
        if tap:
            tap.on = False
        scores.update({"_model": args.model, "_task": task_name,
                       "_split": "+".join(get_task(task_name).metadata.eval_splits),
                       "_max_length": model.max_length, "_neg_top_k": NEG_TOP_K, "_attn": attn})
        if tap:
            scores.update(speed_record(model, tap, attn, device, args.speed_samples))
        json.dump(scores, open(out_dir / f"{task_name}.json", "w"), indent=2)
        pps = f"  PPS {scores['pps']:.1f} (bs {scores['pps_batch_size']})" if scores.get("pps") else ""
        print(f"[done] {task_name}: NDCG@10 {scores['ndcg_at_10']:.4f}{pps}", flush=True)
    print(f"results under {out_dir}; rebuild the table with: uv run python eval/cross-encoder/make_leaderboard.py")


if __name__ == "__main__":
    main()
