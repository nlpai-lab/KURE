"""RTEB(eng) leaderboard over eval/dense/results/<model>/rteb_eng: nDCG@10 on the eight public non-code tasks of
RTEB(eng, beta), their mean, and each model's type and size, one aligned row per model.

A task's score is the mean over all its splits and subsets, as mteb scores it (CUREv1: "all" plus its ten specialties,
English subset). RTEB's other tasks are private or code, so this mean is not the RTEB leaderboard number. Type and size
come from the model's mteb_kor model_meta.json when there is one.

    uv run python eval/dense/make_leaderboard_rteb_eng.py
"""
import argparse
import json
from pathlib import Path

from make_leaderboard_mteb_kor import fmt_params

TASKS = ["CUREv1", "ChatDoctorRetrieval", "FinQARetrieval", "AILACasedocs", "LegalSummarization",
         "HC3FinanceRetrieval", "FinanceBenchRetrieval", "AILAStatutes"]
ABBR = ["CURE", "ChatDoc", "FinQA", "AILAcase", "LegalSum", "HC3Fin", "FinBench", "AILAstat"]


def task_score(scores_by_split) -> float:
    """main_score averaged over every split and subset entry."""
    vals = [e["main_score"] for split in scores_by_split.values() for e in split]
    return sum(vals) / len(vals)


def model_row(model_dir: Path):
    kor_meta = next(model_dir.glob("mteb_kor/*/model_meta.json"), None)
    meta = json.loads(kor_meta.read_text()) if kor_meta else {}
    got = {}
    for f in (model_dir / "rteb_eng").glob("**/*.json"):
        if f.stem == "model_meta":
            meta = meta or json.loads(f.read_text())
        elif f.stem in TASKS:
            got[f.stem] = task_score(json.loads(f.read_text())["scores"])
    if len(got) != len(TASKS):
        return None
    frameworks = set(meta.get("framework") or [])
    return {"name": meta.get("name") or model_dir.name.replace("__", "/"),
            "type": "Late-interaction" if frameworks & {"PyLate", "ColBERT"} else "Dense",
            "params": fmt_params(meta.get("n_parameters")),
            "avg": sum(got.values()) / len(got), "scores": [got[t] for t in TASKS]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    rows = [r for p in sorted(args.results.iterdir()) if (p / "rteb_eng").is_dir() if (r := model_row(p))]
    rows.sort(key=lambda r: -r["avg"])
    name_w = max([len("model")] + [len(r["name"]) for r in rows])
    type_w = max([len("Type")] + [len(r["type"]) for r in rows])
    header = (f"{'#':>2}  {'model':<{name_w}}  {'Type':<{type_w}}  {'Params':>6}     Avg  "
              + "  ".join(f"{a:>8}" for a in ABBR))
    print(header)
    print("-" * len(header))
    for i, r in enumerate(rows, 1):
        print(f"{i:>2}  {r['name']:<{name_w}}  {r['type']:<{type_w}}  {r['params']:>6}  {r['avg']:.4f}  "
              + "  ".join(f"{v:8.4f}" for v in r["scores"]))


if __name__ == "__main__":
    main()
