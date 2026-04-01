"""NanoBEIR-ko custom retrieval tasks (13 subsets)."""

from __future__ import annotations

import logging
from typing import Any

from datasets import load_dataset
from mteb.abstasks import AbsTaskRetrieval
from mteb import TaskMetadata

logger = logging.getLogger(__name__)

# All 13 NanoBEIR-ko subsets
NANOBEIR_KO_SUBSETS = [
    "NanoArguAna",
    "NanoClimateFEVER",
    "NanoDBPedia",
    "NanoFEVER",
    "NanoFiQA2018",
    "NanoHotpotQA",
    "NanoMSMARCO",
    "NanoNFCorpus",
    "NanoNQ",
    "NanoQuoraRetrieval",
    "NanoSCIDOCS",
    "NanoSciFact",
    "NanoTouche2020",
]


def _create_nanobeir_ko_task(subset: str) -> type[AbsTaskRetrieval]:
    """Factory function to create a NanoBEIR-ko task class for a specific subset."""

    class NanoBEIRKoTask(AbsTaskRetrieval):
        metadata = TaskMetadata(
            name=f"{subset}Ko",
            description=f"Korean translation of {subset} from NanoBEIR benchmark",
            reference="https://huggingface.co/datasets/sionic-ai/NanoBEIR-ko",
            dataset={
                "path": "sionic-ai/NanoBEIR-ko",
                "revision": "main",
            },
            type="Retrieval",
            category="t2t",  # text to text
            modalities=["text"],
            eval_splits=["test"],
            eval_langs=["kor-Hang"],
            main_score="ndcg_at_10",
            date=("2024-01-01", "2024-12-31"),
            domains=["Encyclopaedic", "Written"],
            task_subtypes=["Question answering"],
            license="cc-by-4.0",
            annotations_creators="derived",
            dialect=[],
            sample_creation="machine-translated",
            bibtex_citation="",
        )

        def __init__(self, **kwargs: Any):
            super().__init__(**kwargs)
            self._subset = subset

        def load_data(self, **kwargs: Any) -> None:
            """Load the NanoBEIR-ko dataset for this subset."""
            if self.data_loaded:
                return

            # Load queries, corpus, and qrels
            queries_ds = load_dataset(
                "sionic-ai/NanoBEIR-ko",
                "queries",
                split=self._subset,
            )
            corpus_ds = load_dataset(
                "sionic-ai/NanoBEIR-ko",
                "corpus",
                split=self._subset,
            )
            qrels_ds = load_dataset(
                "sionic-ai/NanoBEIR-ko",
                "qrels",
                split=self._subset,
            )

            # Transform to MTEB format
            # Queries: {query_id: query_text}
            queries = {}
            for row in queries_ds:
                queries[str(row["_id"])] = row["text"]

            # Corpus: {doc_id: {"text": text, "title": title}}
            corpus = {}
            empty_count = 0
            for row in corpus_ds:
                text = row["text"] or ""
                title = row.get("title") or ""
                # Skip documents with no text content (causes reshape errors in some models)
                if not text.strip() and not title.strip():
                    empty_count += 1
                    continue
                doc_entry = {"text": text if text.strip() else title}
                if title.strip():
                    doc_entry["title"] = title
                corpus[str(row["_id"])] = doc_entry
            if empty_count:
                logger.warning(f"{self._subset}Ko: Skipped {empty_count} empty documents")

            # Qrels: {query_id: {doc_id: relevance_score}}
            # NanoBEIR-ko uses binary relevance (no score column), default to 1
            qrels = {}
            for row in qrels_ds:
                query_id = str(row["query-id"])
                doc_id = str(row["corpus-id"])
                score = int(row.get("score", 1))  # Default to 1 for binary relevance

                if query_id not in qrels:
                    qrels[query_id] = {}
                qrels[query_id][doc_id] = score

            # Set the data in MTEB format
            self.queries = {"test": queries}
            self.corpus = {"test": corpus}
            self.relevant_docs = {"test": qrels}

            self.data_loaded = True
            logger.info(
                f"Loaded {self._subset}Ko: {len(queries)} queries, {len(corpus)} docs, {len(qrels)} qrels"
            )

    # Set unique class name
    NanoBEIRKoTask.__name__ = f"{subset}Ko"
    NanoBEIRKoTask.__qualname__ = f"{subset}Ko"

    return NanoBEIRKoTask


def create_nanobeir_ko_tasks() -> list[AbsTaskRetrieval]:
    """Create instances of all 13 NanoBEIR-ko tasks."""
    tasks = []
    for subset in NANOBEIR_KO_SUBSETS:
        task_class = _create_nanobeir_ko_task(subset)
        tasks.append(task_class())
    return tasks


# Pre-create task classes for direct import if needed
NanoArguAnaKo = _create_nanobeir_ko_task("NanoArguAna")
NanoClimateFEVERKo = _create_nanobeir_ko_task("NanoClimateFEVER")
NanoDBPediaKo = _create_nanobeir_ko_task("NanoDBPedia")
NanoFEVERKo = _create_nanobeir_ko_task("NanoFEVER")
NanoFiQA2018Ko = _create_nanobeir_ko_task("NanoFiQA2018")
NanoHotpotQAKo = _create_nanobeir_ko_task("NanoHotpotQA")
NanoMSMARCOKo = _create_nanobeir_ko_task("NanoMSMARCO")
NanoNFCorpusKo = _create_nanobeir_ko_task("NanoNFCorpus")
NanoNQKo = _create_nanobeir_ko_task("NanoNQ")
NanoQuoraRetrievalKo = _create_nanobeir_ko_task("NanoQuoraRetrieval")
NanoSCIDOCSKo = _create_nanobeir_ko_task("NanoSCIDOCS")
NanoSciFactKo = _create_nanobeir_ko_task("NanoSciFact")
NanoTouche2020Ko = _create_nanobeir_ko_task("NanoTouche2020")


# ---------------------------------------------------------------------------
# Register custom tasks into MTEB's internal task registry so that
# mteb.get_task("NanoHotpotQAKo") works.  This prevents KeyError when
# MTEB's prompt-resolution fallback (abs_encoder.py:get_instruction) calls
# mteb.get_task() for tasks not shipped with the MTEB package.
#
# Behavior after registration:
#   1. Model's own prompts (prompts_dict) are checked first  → used if found
#   2. Fallback: AbsTaskRetrieval.abstask_prompt ("Retrieve text based on
#      user query.") is returned — identical to all other MTEB retrieval tasks.
# ---------------------------------------------------------------------------
def _register_nanobeir_ko_tasks() -> None:
    """Register NanoBEIR-ko task classes in MTEB's _TASKS_REGISTRY."""
    try:
        from mteb.get_tasks import _TASKS_REGISTRY
    except ImportError:
        logger.debug("Could not import _TASKS_REGISTRY, skipping custom task registration")
        return

    _TASK_CLASSES = {
        "NanoArguAnaKo": NanoArguAnaKo,
        "NanoClimateFEVERKo": NanoClimateFEVERKo,
        "NanoDBPediaKo": NanoDBPediaKo,
        "NanoFEVERKo": NanoFEVERKo,
        "NanoFiQA2018Ko": NanoFiQA2018Ko,
        "NanoHotpotQAKo": NanoHotpotQAKo,
        "NanoMSMARCOKo": NanoMSMARCOKo,
        "NanoNFCorpusKo": NanoNFCorpusKo,
        "NanoNQKo": NanoNQKo,
        "NanoQuoraRetrievalKo": NanoQuoraRetrievalKo,
        "NanoSCIDOCSKo": NanoSCIDOCSKo,
        "NanoSciFactKo": NanoSciFactKo,
        "NanoTouche2020Ko": NanoTouche2020Ko,
    }

    for name, cls in _TASK_CLASSES.items():
        if name not in _TASKS_REGISTRY:
            _TASKS_REGISTRY[name] = cls
            logger.debug(f"Registered custom task '{name}' in MTEB registry")


_register_nanobeir_ko_tasks()
