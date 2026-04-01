#!/usr/bin/env python3
"""Show evaluation results summary for all models."""

import json
import sys
from pathlib import Path


def is_nanobeir(task_name: str) -> bool:
    return task_name.startswith("Nano")


def collect_results(results_dir: str = "eval/results"):
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    models = {}

    for model_dir in sorted(results_path.iterdir()):
        if not model_dir.is_dir():
            continue

        json_files = list(model_dir.rglob("*.json"))
        task_files = [f for f in json_files if f.name != "model_meta.json"]

        if not task_files:
            models[model_dir.name] = {
                "all": {"tasks": 0, "ndcg": [], "recall": []},
                "nanobeir": {"tasks": 0, "ndcg": [], "recall": []},
                "mteb": {"tasks": 0, "ndcg": [], "recall": []},
            }
            continue

        buckets = {
            "all": {"tasks": 0, "ndcg": [], "recall": []},
            "nanobeir": {"tasks": 0, "ndcg": [], "recall": []},
            "mteb": {"tasks": 0, "ndcg": [], "recall": []},
        }

        for task_file in task_files:
            try:
                with open(task_file) as f:
                    data = json.load(f)

                scores_dict = data.get("scores", {})

                if "dev" in scores_dict and "test" not in scores_dict:
                    s = scores_dict["dev"][0]
                elif "test" in scores_dict and "dev" not in scores_dict:
                    s = scores_dict["test"][0]
                elif "dev" in scores_dict and "test" in scores_dict:
                    s = scores_dict["test"][0]
                else:
                    continue

                task_name = task_file.stem
                category = "nanobeir" if is_nanobeir(task_name) else "mteb"

                for bucket_key in ["all", category]:
                    buckets[bucket_key]["tasks"] += 1
                    if "ndcg_at_10" in s:
                        buckets[bucket_key]["ndcg"].append(s["ndcg_at_10"])
                    if "recall_at_10" in s:
                        buckets[bucket_key]["recall"].append(s["recall_at_10"])

            except (json.JSONDecodeError, KeyError, IndexError):
                continue

        models[model_dir.name] = buckets

    return models


def print_table(models, bucket_key, title):
    print(f"\n{'=' * 20} {title} {'=' * 20}")

    name_width = max(len(name) for name in models) + 2
    print(f"\n{'Model':<{name_width}} {'Tasks':>5}  {'NDCG@10':>8}  {'Recall@10':>9}")
    print("-" * (name_width + 28))

    def sort_key(item):
        scores = item[1][bucket_key]["ndcg"]
        return sum(scores) / len(scores) if scores else -1

    for name, data in sorted(models.items(), key=sort_key, reverse=True):
        b = data[bucket_key]
        tasks = b["tasks"]
        avg_ndcg = sum(b["ndcg"]) / len(b["ndcg"]) * 100 if b["ndcg"] else 0
        avg_recall = sum(b["recall"]) / len(b["recall"]) * 100 if b["recall"] else 0

        if tasks == 0:
            print(f"{name:<{name_width}} {tasks:>5}  {'--':>8}  {'--':>9}")
        else:
            print(f"{name:<{name_width}} {tasks:>5}  {avg_ndcg:>7.2f}%  {avg_recall:>8.2f}%")

    print()


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "eval/results"
    models = collect_results(results_dir)

    if not models:
        print("No results found.")
        return

    print_table(models, "all", "ALL TASKS")
    print_table(models, "mteb", "MTEB")
    print_table(models, "nanobeir", "NanoBEIR")


if __name__ == "__main__":
    main()
