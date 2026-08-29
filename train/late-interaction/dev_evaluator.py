"""Small Korean retrieval dev evaluators for in-training monitoring.

Three cheap MTEB(kor, v2) tasks (~11k documents total) so a learning curve shows up
in wandb instead of only loss. These are dev signals, not the reported result: the
final numbers come from the full MTEB harness (eval/evaluate.py).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# (task name, hf_subset or None) -- subset needed where a task is multilingual.
DEV_TASKS = [
    ("AutoRAGRetrieval", None),
    ("Ko-StrategyQA", None),
    ("BelebeleRetrieval", "kor_Hang-kor_Hang"),
]


def _resolve(task, subset: str | None, split: str):
    """Return (corpus, queries, relevant_docs) for one task/subset/split.

    mteb exposes retrieval data two ways: some tasks put everything under
    task.dataset[subset][split], others expose task.corpus/queries/relevant_docs
    keyed the same way. Try both.
    """
    if getattr(task, "dataset", None):
        key = subset if (subset and subset in task.dataset) else next(iter(task.dataset))
        data = task.dataset[key][split]
        return data["corpus"], data["queries"], data["relevant_docs"]

    def pick(holder):
        key = subset if (subset and subset in holder) else next(iter(holder))
        inner = holder[key]
        return inner[split] if split in inner else next(iter(inner.values()))

    return pick(task.corpus), pick(task.queries), pick(task.relevant_docs)


def _as_text(row) -> str:
    if isinstance(row, str):
        return row
    title = (row.get("title") or "").strip()
    text = (row.get("text") or "").strip()
    return f"{title} {text}".strip() if title else text


def _to_id_map(holder) -> dict[str, str]:
    """Normalize either shape into {id: text}.

    Tasks differ: some return an HF Dataset of rows carrying an "id" field, others
    return a plain dict already keyed by id (Belebele).
    """
    if isinstance(holder, dict):
        return {doc_id: _as_text(value) for doc_id, value in holder.items()}
    return {row["id"]: _as_text(row) for row in holder}


def build_dev_evaluator(batch_size: int = 64):
    """SequentialEvaluator over the dev tasks, or None if none could be loaded."""
    import mteb
    from pylate import evaluation
    from sentence_transformers.evaluation import SequentialEvaluator

    evaluators = []
    for name, subset in DEV_TASKS:
        try:
            task = mteb.get_tasks(tasks=[name], languages=["kor"])[0]
            task.load_data()
            split = task.metadata.eval_splits[0]
            corpus_ds, queries_ds, relevant = _resolve(task, subset, split)

            corpus = _to_id_map(corpus_ds)
            queries = _to_id_map(queries_ds)
            # mteb gives qid -> {docid: relevance}; keep the positives.
            relevant_docs = {
                qid: {doc_id for doc_id, score in docs.items() if score > 0}
                for qid, docs in relevant.items()
            }
            relevant_docs = {q: d for q, d in relevant_docs.items() if d and q in queries}
            queries = {q: t for q, t in queries.items() if q in relevant_docs}

            evaluators.append(
                evaluation.PyLateInformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus,
                    relevant_docs=relevant_docs,
                    name=f"dev-{name}",
                    batch_size=batch_size,
                    ndcg_at_k=[10],
                    accuracy_at_k=[1, 10],
                    precision_recall_at_k=[10, 100],
                    map_at_k=[100],
                )
            )
            logger.info(
                "dev evaluator %s: %d queries, %d documents", name, len(queries), len(corpus)
            )
        except Exception as exc:  # a missing dev task must not stop training
            logger.warning("dev evaluator %s unavailable: %s: %s", name, type(exc).__name__, exc)

    if not evaluators:
        return None
    return SequentialEvaluator(evaluators)
