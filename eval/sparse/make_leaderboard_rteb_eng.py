"""RTEB(eng) leaderboard over eval/sparse/results/<model>/rteb_eng: nDCG@10 on the eight public non-code tasks of
RTEB(eng, beta), their mean, plus the sparsity numbers evaluate.py records (avg query/corpus active dims, avg_flops)
averaged over the tasks.

A task's score is the mean over all its splits and subsets, as mteb scores it (CUREv1: "all" plus its ten specialties,
English subset). RTEB's other tasks are private or code, so this mean is not the RTEB leaderboard number.

    uv run python eval/sparse/make_leaderboard_rteb_eng.py
"""
import argparse
import json
from pathlib import Path

TASKS = ["CUREv1", "ChatDoctorRetrieval", "FinQARetrieval", "AILACasedocs", "LegalSummarization",
         "HC3FinanceRetrieval", "FinanceBenchRetrieval", "AILAStatutes"]
ABBR = ["CURE", "ChatDoc", "FinQA", "AILAcase", "LegalSum", "HC3Fin", "FinBench", "AILAstat"]
FIELDS = ["main_score", "query_active_dims", "corpus_active_dims", "avg_flops"]


def task_cell(scores_by_split) -> list[float]:
    """Each field averaged over every split and subset entry."""
    entries = [e for split in scores_by_split.values() for e in split]
    return [sum(e[f] for e in entries) / len(entries) for f in FIELDS]


def model_row(model_dir: Path):
    meta, got = {}, {}
    for f in (model_dir / "rteb_eng").glob("*.json"):
        if f.stem == "model_meta":
            meta = json.loads(f.read_text())
        elif f.stem in TASKS:
            got[f.stem] = task_cell(json.loads(f.read_text())["scores"])
    if len(got) != len(TASKS):
        return None
    avgs = [sum(got[t][i] for t in TASKS) / len(TASKS) for i in range(len(FIELDS))]
    return meta.get("name") or model_dir.name.replace("__", "/"), [got[t][0] for t in TASKS], avgs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    rows = [r for p in sorted(args.results.iterdir()) if (p / "rteb_eng").is_dir() if (r := model_row(p))]
    rows.sort(key=lambda r: -r[2][0])
    name_w = max([len("model")] + [len(name) for name, _, _ in rows])
    print(f"{'model':<{name_w}}     Avg  " + "  ".join(f"{a:>8}" for a in ABBR) + "   q_dim   c_dim   FLOPs")
    print("-" * (name_w + 10 + 10 * len(ABBR) + 24))
    for name, scores, (avg, q, c, flops) in rows:
        print(f"{name:<{name_w}}  {avg:.4f}  " + "  ".join(f"{v:8.4f}" for v in scores)
              + f"  {q:6.1f}  {c:6.1f}  {flops:6.2f}")


if __name__ == "__main__":
    main()
