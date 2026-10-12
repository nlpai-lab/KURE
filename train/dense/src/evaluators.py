"""In-training retrieval evaluation on the Korean benchmark tasks.

The same five MTEB(kor, v2) retrieval tasks the final eval reports, so the training curve and
the final number measure the same thing:

    PublicHealthQA, BelebeleRetrieval (kor-kor), AutoRAGRetrieval, LawIRKo, SQuADKorV1Retrieval

Together they are ~5.8k passages and ~10.4k queries. That is roughly a seventh of the 13-subset
NanoBEIR pair this replaces (56.7k passages per language), which is what makes evaluating on the
real benchmark affordable every `eval_steps` -- one pass costs a couple of minutes instead of
twelve.

Each task becomes its own `InformationRetrievalEvaluator`, so `eval_sequential_score` is the
mean NDCG@10 over the five. Evaluation is always at the model's full embedding dimension --
Matryoshka training truncates the *loss*, not this.
"""

from __future__ import annotations

import logging
import os

import mteb

from sentence_transformers.base.evaluation import BaseEvaluator
from sentence_transformers.sentence_transformer.evaluation import (
    InformationRetrievalEvaluator,
    SequentialEvaluator,
)

# task name -> hf_subset to score, or None when the task has a single unnamed subset.
# Belebele ships three directions; the benchmark's Korean number is the kor-kor one.
TASKS: dict[str, str | None] = {
    "PublicHealthQA": "korean",
    "BelebeleRetrieval": "kor_Hang-kor_Hang",
    "AutoRAGRetrieval": None,
    "LawIRKo": None,
    "SQuADKorV1Retrieval": None,
}


def _split_data(task, want_subset: str | None):
    """Return (corpus, queries, relevant_docs) for the task's eval split.

    mteb 2.x exposes retrieval data in two shapes and which one a task uses is not declared:
    some fill `task.dataset[subset][split]` with HF `Dataset`s, others fill `task.corpus[subset]
    [split]` with plain dicts. Handle both rather than special-casing task names.
    """
    splits = task.metadata.eval_splits

    dataset = getattr(task, "dataset", None)
    if isinstance(dataset, dict) and dataset:
        for subset, by_split in dataset.items():
            if want_subset is not None and subset != want_subset:
                continue
            for split, data in by_split.items():
                if split in splits and isinstance(data, dict) and "corpus" in data:
                    return data["corpus"], data["queries"], data["relevant_docs"]

    corpus = getattr(task, "corpus", None)
    if corpus:
        for subset, by_split in corpus.items():
            if want_subset is not None and subset != want_subset:
                continue
            for split in by_split:
                if split in splits:
                    return (by_split[split], task.queries[subset][split],
                            task.relevant_docs[subset][split])

    raise ValueError(f"{task.metadata.name}: no data for subset {want_subset!r} in {splits}")


def _as_text_map(rows, id_key: str = "id") -> dict[str, str]:
    """{id: text} from either an HF Dataset of {id, text, title} or a dict of {id: text|{...}}."""
    out: dict[str, str] = {}
    if hasattr(rows, "items"):  # dict shape
        for doc_id, value in rows.items():
            if isinstance(value, dict):
                title, text = value.get("title") or "", value.get("text") or ""
                out[str(doc_id)] = f"{title} {text}".strip() if title else text
            else:
                out[str(doc_id)] = value
        return out
    for row in rows:  # HF Dataset shape
        title, text = row.get("title") or "", row.get("text") or ""
        out[str(row[id_key] if id_key in row else row["_id"])] = (
            f"{title} {text}".strip() if title else text
        )
    return out


def _as_relevant(qrels: dict) -> dict[str, set[str]]:
    """{qid: {doc_id}} keeping only positively-judged pairs."""
    relevant: dict[str, set[str]] = {}
    for query_id, docs in qrels.items():
        kept = {str(d) for d, score in docs.items() if float(score) > 0}
        if kept:
            relevant[str(query_id)] = kept
    return relevant


def load_evaluator(data_args, training_args) -> BaseEvaluator:
    """Build a SequentialEvaluator over `data_args.eval_subsets` (task names from TASKS).

    Under DDP only rank 0 evaluates: the IR evaluator uses `model.encode`, which runs under
    `no_grad` and issues no collectives, so having every rank encode the corpus wastes memory
    and time without keeping anything in sync. The other ranks wait at the next training
    collective (bounded by `ddp_timeout`).
    """
    if int(os.environ.get("RANK", "0")) != 0:
        return SequentialEvaluator([])

    unknown = set(data_args.eval_subsets) - TASKS.keys()
    if unknown:
        raise ValueError(f"unknown eval tasks {sorted(unknown)}; known: {sorted(TASKS)}")

    evaluators = []
    for name in data_args.eval_subsets:
        task = mteb.get_tasks(tasks=[name], languages=["kor"])[0]
        task.load_data()
        corpus_rows, query_rows, qrels = _split_data(task, TASKS[name])
        corpus = _as_text_map(corpus_rows)
        queries = _as_text_map(query_rows)
        relevant_docs = _as_relevant(qrels)
        # Drop queries with no judgement: the evaluator averages over what it is given, and a
        # query that cannot be right would just dilute every score by a constant.
        queries = {q: t for q, t in queries.items() if q in relevant_docs}
        logging.info(f"{name}: {len(queries):,} queries over {len(corpus):,} passages")
        evaluators.append(
            InformationRetrievalEvaluator(
                queries=queries,
                corpus=corpus,
                relevant_docs=relevant_docs,
                query_prompt=training_args.query_prompt or None,
                corpus_prompt=training_args.document_prompt or None,
                mrr_at_k=[10],
                ndcg_at_k=[10],
                accuracy_at_k=[10],
                precision_recall_at_k=[10],
                map_at_k=[10],
                batch_size=training_args.per_device_eval_batch_size,
                name=name,
                write_csv=False,
            )
        )

    logging.info(f"in-training eval: {len(evaluators)} Korean retrieval tasks")
    return _AveragingSequentialEvaluator(evaluators, main_score_function=lambda s: sum(s) / len(s))


class _AveragingSequentialEvaluator(SequentialEvaluator):
    """SequentialEvaluator that also reports the mean over tasks under a stable key.

    `sequential_score` already averages, but it is named for the evaluator rather than the
    metric; `avg_ndcg@10` is the number to watch across runs and is what the final leaderboard
    reports.
    """

    def __call__(self, *args, **kwargs) -> dict[str, float]:
        results = super().__call__(*args, **kwargs)
        for metric in ("ndcg@10", "mrr@10"):
            values = [v for k, v in results.items() if k.endswith(metric) and not k.startswith("avg_")]
            if values:
                results[f"avg_{metric}"] = sum(values) / len(values)
        logging.info(
            "eval averages: "
            + "  ".join(f"{k}={results[k]:.4f}" for k in sorted(results) if k.startswith("avg_"))
        )
        return results
