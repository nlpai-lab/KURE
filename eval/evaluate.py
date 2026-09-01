"""Evaluate a model on the nine MTEB(kor, v2) retrieval tasks.

Results are written under eval/results/<org__model>/<revision>/, the tree that
eval/make_leaderboard.py turns into the README leaderboard.

Usage:
    uv run python eval/evaluate.py nlpai-lab/KURE-v2 multi-vector
    uv run python eval/evaluate.py nlpai-lab/KURE-v1 single-vector
"""
import argparse
from pathlib import Path

import mteb

TASKS = ["AutoRAGRetrieval", "PublicHealthQA", "Ko-StrategyQA", "LawIRKo",
         "SQuADKorV1Retrieval", "BelebeleRetrieval", "MrTidyRetrieval",
         "MultiLongDocRetrieval", "MIRACLRetrieval"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face model name, e.g. nlpai-lab/KURE-v2")
    parser.add_argument("paradigm", choices=["single-vector", "multi-vector"],
                        help="single-vector: dense embedding model; "
                             "multi-vector: late-interaction (ColBERT-style) model, "
                             "served with PyLate + PLAID")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tasks", nargs="+", default=TASKS,
                        help="subset of tasks to run (default: all nine)")
    args = parser.parse_args()

    tasks = mteb.get_tasks(tasks=args.tasks, languages=["kor"])
    if args.paradigm == "multi-vector":
        from mteb.models.model_implementations.pylate_models import MultiVectorModel
        model = MultiVectorModel(args.model)
    else:
        model = mteb.get_model(args.model)

    cache = mteb.ResultCache(cache_path=Path(__file__).resolve().parent)
    mteb.evaluate(model, tasks, encode_kwargs={"batch_size": args.batch_size}, cache=cache)
    print("done; results under eval/results/. Rebuild the table with "
          "`uv run python eval/make_leaderboard.py`")


if __name__ == "__main__":
    main()
