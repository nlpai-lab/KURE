"""Rebuild the README reranking table from eval/cross-encoder/results.

Reads results/<org>/<model>/<task>.json (written by evaluate.py, or brought from
instructkr/reranker-simple-benchmark, which uses the same protocol) and prints a markdown table:
nine-task mean NDCG@10 and mean PPS, sorted by NDCG@10. A model missing a task (the listwise jina
rerankers have no MultiLongDocRetrieval at 8192) is listed below the ranked rows without a mean.

    uv run python eval/cross-encoder/make_leaderboard.py
"""

import argparse
import json
from pathlib import Path

TASKS = ["Ko-StrategyQA", "AutoRAGRetrieval", "PublicHealthQA", "BelebeleRetrieval",
         "MIRACLRetrieval", "MrTidyRetrieval", "MultiLongDocRetrieval", "SQuADKorV1Retrieval", "LawIRKo"]


def load(model_dir: Path) -> dict[str, dict]:
    return {t: json.loads((model_dir / f"{t}.json").read_text())
            for t in TASKS if (model_dir / f"{t}.json").exists()}


def mean(values: list) -> float | None:
    return sum(values) / len(values) if values and None not in values else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    rows = []
    for model_dir in sorted(p for p in args.results.glob("*/*") if p.is_dir()):
        tasks = load(model_dir)
        name = f"{model_dir.parent.name}/{model_dir.name}"
        complete = len(tasks) == len(TASKS)
        ndcg = mean([r["ndcg_at_10"] for r in tasks.values()]) if complete else None
        pps = mean([r.get("pps") for r in tasks.values()]) if complete else None
        rows.append((name, ndcg, pps, len(tasks)))

    rows.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))
    print("| Model | Mean NDCG@10 | Mean PPS |")
    print("|---|---:|---:|")
    for name, ndcg, pps, n in rows:
        shown = f"**{name}**" if name.startswith("nlpai-lab/KURE") else name
        ndcg_s = f"{ndcg:.4f}" if ndcg is not None else f"— ({n}/{len(TASKS)} tasks)"
        pps_s = f"{pps:.1f}" if pps is not None else "—"
        print(f"| {shown} | {ndcg_s} | {pps_s} |")


if __name__ == "__main__":
    main()
