"""Evaluate a model on the nine MTEB(kor, v2) retrieval tasks, or with --suite rteb_eng on the eight public
non-code tasks of RTEB(eng, beta) (its other tasks are private or code).

Results are written under eval/dense/results/<org__model>/<suite>/<revision>/ (suite = mteb_kor or rteb_eng), the
tree eval/dense/make_leaderboard_mteb_kor.py and make_leaderboard_rteb_eng.py turn into the leaderboards.

Usage:
    uv run --extra late-interaction --no-default-groups python eval/dense/evaluate.py nlpai-lab/KURE-v2 multi-vector
    uv run python eval/dense/evaluate.py nlpai-lab/KURE-v1 single-vector
    uv run python eval/dense/evaluate.py nlpai-lab/KURE-v3 single-vector --suite rteb_eng
"""
import argparse
from pathlib import Path

import mteb

SUITES = {
    "mteb_kor": (["AutoRAGRetrieval", "PublicHealthQA", "Ko-StrategyQA", "LawIRKo", "SQuADKorV1Retrieval",
                  "BelebeleRetrieval", "MrTidyRetrieval", "MultiLongDocRetrieval", "MIRACLRetrieval"], "kor"),
    "rteb_eng": (["CUREv1", "ChatDoctorRetrieval", "FinQARetrieval", "AILACasedocs", "LegalSummarization",
                  "HC3FinanceRetrieval", "FinanceBenchRetrieval", "AILAStatutes"], "eng"),
}


class SuiteResultCache(mteb.ResultCache):
    """mteb's result cache with the suite inserted under the model: <model>/<suite>/<revision>/<task>.json."""

    def __init__(self, cache_path: Path, suite: str):
        super().__init__(cache_path=cache_path)
        self.suite = suite

    def get_task_result_path(self, task_name, model_name, model_revision=None, remote=False, experiment_name=None):
        path = super().get_task_result_path(task_name, model_name, model_revision, remote, experiment_name)
        root = self.cache_path / "results"
        model, *rest = path.relative_to(root).parts
        return root / model / self.suite / Path(*rest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help="Hugging Face model name, e.g. nlpai-lab/KURE-v2")
    parser.add_argument("paradigm", choices=["single-vector", "multi-vector"],
                        help="single-vector: dense embedding model; "
                             "multi-vector: late-interaction (ColBERT-style) model, "
                             "served with PyLate + PLAID")
    parser.add_argument("--suite", default="mteb_kor", choices=list(SUITES))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tasks", nargs="+", help="subset of the suite's tasks to run (default: all of them)")
    args = parser.parse_args()

    names, language = SUITES[args.suite]
    tasks = mteb.get_tasks(tasks=args.tasks or names, languages=[language])
    if args.paradigm == "multi-vector":
        from mteb.models.model_implementations.pylate_models import MultiVectorModel
        from mteb.models.model_meta import ModelMeta
        # fast-plaid searches 2**18 queries per batch by default, i.e. all of them at once: CUREv1's ~6k queries
        # ask for 17 GiB. Queries are scored independently, so a smaller batch changes no score.
        model = MultiVectorModel(args.model, index_kwargs={"batch_size": 64})
        # the wrapper carries no ModelMeta, and mteb.evaluate would then try to load an empty one
        try:
            model.mteb_model_meta = mteb.get_model_meta(args.model)
        except KeyError:
            model.mteb_model_meta = ModelMeta.create_empty().model_copy(
                update={"name": args.model, "revision": "local"})
    else:
        model = mteb.get_model(args.model)

    cache = SuiteResultCache(Path(__file__).resolve().parent, args.suite)
    mteb.evaluate(model, tasks, encode_kwargs={"batch_size": args.batch_size}, cache=cache)
    print("done; results under eval/dense/results/. Rebuild the table with "
          f"`uv run python eval/dense/make_leaderboard_{args.suite}.py`")


if __name__ == "__main__":
    main()
