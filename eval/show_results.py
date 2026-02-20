#!/usr/bin/env python3
"""Show evaluation results summary for all models."""

import json
import sys
from pathlib import Path


def collect_results(results_dir: str = "eval/results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    models = {}

    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir():
            continue

        # Find JSON result files (nested: model_dir/subfolder/hash/*.json)
        json_files = list(model_dir.rglob("*.json"))
        task_files = [f for f in json_files if f.name != "model_meta.json"]

        if not task_files:
            models[model_dir.name] = {"tasks": 0, "ndcg": [], "recall": [], "mrr": []}
            continue

        ndcg_scores = []
        recall_scores = []
        mrr_scores = []
        task_count = 0

        for task_file in task_files:
            try:
                with open(task_file) as f:
                    data = json.load(f)

                scores = data.get("scores", {}).get("test", [])
                if not scores:
                    continue

                s = scores[0]
                task_count += 1

                if "ndcg_at_10" in s:
                    ndcg_scores.append(s["ndcg_at_10"])
                if "recall_at_10" in s:
                    recall_scores.append(s["recall_at_10"])
                if "mrr_at_10" in s:
                    mrr_scores.append(s["mrr_at_10"])

            except (json.JSONDecodeError, KeyError, IndexError):
                continue

        models[model_dir.name] = {
            "tasks": task_count,
            "ndcg": ndcg_scores,
            "recall": recall_scores,
            "mrr": mrr_scores,
        }

    return models


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "eval/results"
    models = collect_results(results_dir)

    if not models:
        print("No results found.")
        return

    # Header
    name_width = max(len(name) for name in models) + 2
    print()
    print(f"{'Model':<{name_width}} {'Tasks':>5}  {'NDCG@10':>8}  {'Recall@10':>9}  {'MRR@10':>8}")
    print("-" * (name_width + 38))

    # Sort by avg ndcg@10 descending
    def sort_key(item):
        scores = item[1]["ndcg"]
        return sum(scores) / len(scores) if scores else -1

    for name, data in sorted(models.items(), key=sort_key, reverse=True):
        tasks = data["tasks"]
        avg_ndcg = sum(data["ndcg"]) / len(data["ndcg"]) * 100 if data["ndcg"] else 0
        avg_recall = sum(data["recall"]) / len(data["recall"]) * 100 if data["recall"] else 0
        avg_mrr = sum(data["mrr"]) / len(data["mrr"]) * 100 if data["mrr"] else 0

        if tasks == 0:
            print(f"{name:<{name_width}} {tasks:>5}  {'--':>8}  {'--':>9}  {'--':>8}")
        else:
            print(f"{name:<{name_width}} {tasks:>5}  {avg_ndcg:>7.2f}%  {avg_recall:>8.2f}%  {avg_mrr:>7.2f}%")

    print()


if __name__ == "__main__":
    main()
