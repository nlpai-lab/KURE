"""Evaluate a SparseEncoder on the nine retrieval tasks of MTEB(kor, v2).

The 20-task suite is the one used in
https://github.com/embeddings-benchmark/results/pull/654; this runs its nine retrieval tasks.
Pass --all-tasks for all twenty, or --tasks to name specific ones. --suite runs an English set instead:
beir_eng, the 13 BEIR tasks plus MS MARCO (dev), as SparseUp reports them, or rteb_eng, the eight public
non-code tasks of RTEB(eng, beta). Vectors are used as the model produces them, with no max_active_dims
pruning unless --max-active-dims asks for it. The reported results use none.

Results go under eval/sparse/results/<model>/<suite>/ (suite = mteb_kor, beir_eng or rteb_eng), one mteb
result file per task, and eval/sparse/make_leaderboard_<suite>.py turns them into the leaderboards.
Each task file also carries the sparsity numbers ST's SparseInformationRetrievalEvaluator
reports -- query/corpus active dims and sparsity ratio, and avg_flops, the dot product of the
mean query and mean corpus activation-indicator vectors -- computed from the very embeddings
mteb scored, per split and subset.

mteb ships a `SparseEncoderWrapper` for the OpenSearch sparse models. It takes any
SparseEncoder path and already handles batched sparse similarity, so it is reused verbatim
rather than reimplemented.

    uv run python eval/sparse/evaluate.py nlpai-lab/KURE-Sparse-nano
    uv run python eval/sparse/evaluate.py output/kure-sparse-base/final --tasks PublicHealthQA
    uv run python eval/sparse/evaluate.py nlpai-lab/KURE-Sparse-base --suite beir_eng --max-seq-length 8192
    uv run python eval/sparse/evaluate.py nlpai-lab/KURE-Sparse-base --suite rteb_eng --max-seq-length 8192
    uv run python eval/sparse/evaluate.py utahnlp/laconic-1b --suite rteb_eng --revision refs/pr/1 --max-seq-length 8192
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
import sys
from pathlib import Path

# A checkpoint trained with train/sparse/src/pooling.py names that module in its modules.json.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "train" / "sparse" / "src"))

import mteb
import torch
from mteb.models.model_implementations.opensearch_neural_sparse_models import (
    SparseEncoderWrapper,
)
from mteb.models.model_meta import ModelMeta
from mteb.types import PromptType

BENCHMARK = "MTEB(kor, v2)"
RETRIEVAL_TYPES = ("Retrieval",)
# --suite beir_eng: the 13 BEIR tasks plus MS MARCO (dev), as SparseUp reports them. Biggest corpus first.
BEIR_TASKS = ["MSMARCO", "ClimateFEVER", "FEVER", "HotpotQA", "DBPedia", "NQ", "QuoraRetrieval", "Touche2020",
              "TRECCOVID", "SCIDOCS", "FiQA2018", "ArguAna", "NFCorpus", "SciFact"]


# --suite rteb_eng: the public non-code tasks of RTEB(eng, beta); its other tasks are private or code. Biggest first.
RTEB_TASKS = ["CUREv1", "ChatDoctorRetrieval", "FinQARetrieval", "AILACasedocs", "LegalSummarization",
              "HC3FinanceRetrieval", "FinanceBenchRetrieval", "AILAStatutes"]


def select_tasks(suite: str, all_tasks: bool, only: list[str] | None):
    if suite in ("beir_eng", "rteb_eng"):
        tasks = list(mteb.get_tasks(tasks=BEIR_TASKS if suite == "beir_eng" else RTEB_TASKS, languages=["eng"]))
    else:
        tasks = list(mteb.get_benchmark(BENCHMARK).tasks)
    if only:
        by_name = {t.metadata.name: t for t in tasks}
        missing = [n for n in only if n not in by_name]
        if missing:
            raise SystemExit(f"unknown task(s) {missing}; benchmark has {sorted(by_name)}")
        return [by_name[n] for n in only]
    if all_tasks:
        return tasks
    return [t for t in tasks if t.metadata.type in RETRIEVAL_TYPES]


def is_oom(exc: BaseException) -> bool:
    """True for the resource exhaustions that a smaller batch can fix.

    cuSPARSE reports its own exhaustion as "insufficient resources" rather than an
    OutOfMemoryError, and the sparse matmul that raises it is reached on the same path. Leaving
    it out cost a full MrTidy evaluation: the loop had halved to 128 and would have succeeded
    at 32, as it did on every earlier model, but re-raised instead.
    """
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return isinstance(exc, RuntimeError) and (
        "out of memory" in message or "insufficient resources" in message
    )


def sparse_dense_similarity(queries, corpus, query_chunk: int = 128):
    """Dot product between a sparse corpus and a densified query batch.

    mteb multiplies two sparse tensors, and cusparseSpGEMM answers "insufficient resources" on
    MrTidy's 1.5M passages for any model whose queries carry a few hundred nonzeros -- which a
    PFT-initialised model does, keeping ~200 against ~70 for one trained from scratch. The
    batch size does not help: it failed identically from 512 down to 1, because the batch sizes
    the query rows, not the workspace SpGEMM needs for the product.

    Densifying only the queries turns it into SpMM, which has no such blowup. The arithmetic is
    the same dot product; the query chunk is at most 128 x 50000 floats and the result is
    n_docs x 128.
    """
    out = []
    dense_corpus = not corpus.is_sparse
    for start in range(0, queries.size(0), query_chunk):
        stop = min(start + query_chunk, queries.size(0))
        if queries.is_sparse:
            # Sparse COO supports index_select but not slicing.
            rows = torch.arange(start, stop, device=queries.device)
            q = queries.index_select(0, rows).to_dense().float()
        else:
            q = queries[start:stop].float()
        if dense_corpus:
            out.append(q @ corpus.float().T)
        else:
            out.append(torch.sparse.mm(corpus.float(), q.T).T)
    return torch.cat(out, dim=0)


# Replace the wrapper's own batched sparse x sparse similarity. SentenceTransformer exposes
# `similarity` as a property, so the instance cannot be patched; this method is the single
# entry point both mteb call sites go through.
SparseEncoderWrapper.similarity = lambda self, queries, corpus: sparse_dense_similarity(
    queries, corpus
)


class SparsityStatsWrapper(SparseEncoderWrapper):
    """The mteb wrapper, keeping the activation counts of everything it encodes.

    mteb names the split, subset and side (query/document) on every encode call, so the
    statistics land exactly on the score entries they belong to. The count vector is the number
    of rows in which each vocabulary dimension is active; its mean over rows is what ST dots
    for avg_flops.
    """

    def __init__(self, model_name, dtype: torch.dtype, **kwargs):
        from sentence_transformers import SparseEncoder

        # Both KURE-Sparse models ship custom module code (pooling.py, milco_encoder.py), and every model
        # is scored in the same dtype; MILCO keeps fp32 master weights on load, so cast after loading.
        super().__init__(SparseEncoder(model_name, trust_remote_code=True, **kwargs).to(dtype))
        # The OpenSearch wrapper leaves the meta to its registry entries; a bare path has none,
        # and the result cache needs a name (it becomes the results directory) and a revision.
        self.mteb_model_meta = ModelMeta.create_empty(
            overwrites=dict(name=model_name, revision="no_revision_available",
                            model_type=["sparse"], loader=type(self)),
        )
        self.stats = defaultdict(lambda: {"rows": 0, "nnz": 0, "count": None})

    def encode(self, inputs, *, task_metadata, hf_split, hf_subset, prompt_type=None, **kwargs):
        embeddings = super().encode(
            inputs, task_metadata=task_metadata, hf_split=hf_split, hf_subset=hf_subset,
            prompt_type=prompt_type, **kwargs,
        )
        side = "query" if prompt_type == PromptType.query else "corpus"
        # index_select on the sparse tensor makes no copy of the whole thing; the counts are
        # the only dense state kept, one vocab-sized vector per side.
        emb = embeddings.coalesce() if embeddings.is_sparse else embeddings.to_sparse().coalesce()
        dims = emb.indices()[1]
        count = torch.bincount(dims, minlength=emb.shape[1]).to("cpu", torch.float64)
        stat = self.stats[(hf_split, hf_subset, side)]
        stat["rows"] += emb.shape[0]
        stat["nnz"] += int(emb.values().ne(0).sum())
        stat["count"] = count if stat["count"] is None else stat["count"] + count
        return embeddings

    def sparsity_scores(self, hf_split: str, hf_subset: str) -> dict[str, float]:
        """ST's SparseInformationRetrievalEvaluator numbers for one split/subset."""
        q, d = self.stats.get((hf_split, hf_subset, "query")), self.stats.get((hf_split, hf_subset, "corpus"))
        if not q or not d:
            return {}
        vocab = len(q["count"])
        return {
            "query_active_dims": q["nnz"] / q["rows"],
            "query_sparsity_ratio": 1 - q["nnz"] / q["rows"] / vocab,
            "corpus_active_dims": d["nnz"] / d["rows"],
            "corpus_sparsity_ratio": 1 - d["nnz"] / d["rows"] / vocab,
            "avg_flops": float(torch.dot(q["count"] / q["rows"], d["count"] / d["rows"])),
        }


class SuiteResultCache(mteb.ResultCache):
    """mteb's result cache with the language suite in place of the revision directory.

    Results land in results/<model>/<suite>/<task>.json (suite = mteb_kor, beir_eng or rteb_eng), so a model's
    Korean and English runs sit side by side. Saving and the skip-if-on-disk check both go through this one path method.
    """

    def __init__(self, cache_path: Path, suite: str):
        super().__init__(cache_path=cache_path)
        self.suite = suite

    def get_task_result_path(self, task_name, model_name, model_revision=None, remote=False, experiment_name=None):
        if isinstance(model_name, str) and model_revision is None:
            model_revision = "no_revision_available"
        path = super().get_task_result_path(task_name, model_name, model_revision, remote)
        return path.parent.parent / self.suite / path.name


def evaluate_task(model, task, args, cache):
    """Evaluate one task, halving the batch size on every OOM.

    Corpus sizes here span three orders of magnitude (AutoRAG 720 passages, MrTidy over a
    million) and a SPLADE forward is dominated by a [batch, seq_len, vocab] tensor, so no
    single batch size fits every task. Starting high and backing off finds the largest one
    that works instead of pinning everything to the worst case.
    """
    name = task.metadata.name
    batch_size = args.batch_size
    for attempt in range(1, args.max_oom_retries + 1):
        model.stats.clear()
        try:
            result = mteb.evaluate(
                model, [task], encode_kwargs={"batch_size": batch_size}, cache=cache,
                overwrite_strategy="always" if args.overwrite else "only-missing",
            )
            break
        except Exception as exc:  # noqa: BLE001 - re-raised below unless it is an OOM
            if not is_oom(exc) or batch_size <= 1:
                raise
            torch.cuda.empty_cache()
            batch_size //= 2
            # Keep the allocation size: it separates an encode-batch OOM, which halving fixes,
            # from one caused by holding a million-passage index, which it will not.
            detail = str(exc).split("\n")[0][:160]
            logging.warning(f"  {name}: OOM (attempt {attempt}), retrying at batch_size {batch_size} -- {detail}")
    else:
        raise RuntimeError(f"{name}: still OOM after {args.max_oom_retries} attempts")
    if attempt > 1:
        logging.info(f"  {name}: succeeded at batch_size {batch_size}")

    task_result = result.task_results[0]
    if not model.stats:
        logging.info(f"  {name}: already on disk, kept as is")
        return task_result
    # Add the sparsity numbers to the score entries and rewrite the task file mteb just saved:
    # one file per task, still in mteb's own format, so the leaderboard and mteb both read it.
    for split, entries in task_result.scores.items():
        for entry in entries:
            entry.update(model.sparsity_scores(split, entry["hf_subset"]))
    task_result.to_disk(cache.get_task_result_path(
        name, result.model_name, model_revision=result.model_revision,
        experiment_name=result.experiment_name,
    ))
    return task_result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="SparseEncoder checkpoint path or hub id")
    parser.add_argument("--batch-size", type=int, default=512,
                        help="Starting batch size; halved on each OOM")
    parser.add_argument("--max-oom-retries", type=int, default=20)
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-seq-length", type=int, default=0,
                        help="Override the checkpoint's own limit; 0 keeps it")
    parser.add_argument("--suite", default="mteb_kor", choices=["mteb_kor", "beir_eng", "rteb_eng"],
                        help="mteb_kor: MTEB(kor, v2) retrieval; beir_eng: the 13 BEIR tasks + MS MARCO; "
                             "rteb_eng: the public non-code RTEB(eng) tasks")
    parser.add_argument("--revision", default=None, help="Hub revision, e.g. refs/pr/1")
    parser.add_argument("--max-active-dims", type=int, default=None,
                        help="Keep the top-k dims per vector (unset for the reported results)")
    parser.add_argument("--all-tasks", action="store_true", help="Include the non-retrieval tasks")
    parser.add_argument("--tasks", nargs="+", help="Run only these task names")
    parser.add_argument("--overwrite", action="store_true", help="Redo tasks already on disk")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    tasks = select_tasks(args.suite, args.all_tasks, args.tasks)
    logging.info(f"{len(tasks)} tasks: {[t.metadata.name for t in tasks]}")

    extra = {k: v for k, v in (("revision", args.revision), ("max_active_dims", args.max_active_dims)) if v}
    model = SparsityStatsWrapper(args.model_path, dtype=getattr(torch, args.dtype), **extra)
    if args.max_seq_length:
        # The wrapper builds the SparseEncoder itself and takes no length argument, so reach
        # through to it. Our models trained at 1024 but sit on a backbone pretrained at 16384,
        # so a longer evaluation window is what the long-document tasks actually need.
        model.model.max_seq_length = args.max_seq_length
        logging.info(f"max_seq_length overridden to {args.max_seq_length}")

    cache = SuiteResultCache(Path(__file__).resolve().parent, args.suite)
    for task in tasks:
        task_result = evaluate_task(model, task, args, cache)
        extra = next((e for entries in task_result.scores.values() for e in entries if "avg_flops" in e), {})
        logging.info(
            f"{task.metadata.name}: {task_result.get_score():.4f}"
            + (f" | query {extra['query_active_dims']:.1f} / corpus {extra['corpus_active_dims']:.1f} "
               f"active dims, avg_flops {extra['avg_flops']:.2f}" if extra else "")
        )
    print(f"done; results under eval/sparse/results/. Table: uv run python eval/sparse/make_leaderboard_{args.suite}.py")


if __name__ == "__main__":
    main()
