"""Rebuild the README leaderboard table from eval/results.

Discovers every model under the results directory, reads its per-task mteb result
files, applies the MTEB(kor, v2) conventions (Korean subsets only; Belebele =
kor_Hang-kor_Hang; MLDR = mean of dev/test; MIRACL = dev split) and prints a
markdown table. Model type and parameter count come from each model's
model_meta.json; models under the nlpai-lab organization are bolded.

Usage:
    uv run python eval/make_leaderboard.py
    uv run python eval/make_leaderboard.py --metrics ndcg_at_10 recall_at_10 mrr_at_10
"""
import argparse
import json
from pathlib import Path

TASKS = ["AutoRAGRetrieval", "PublicHealthQA", "Ko-StrategyQA", "LawIRKo",
         "SQuADKorV1Retrieval", "BelebeleRetrieval", "MrTidyRetrieval",
         "MultiLongDocRetrieval", "MIRACLRetrieval"]
KO_SUBSETS = {"BelebeleRetrieval": {"kor_Hang-kor_Hang"}, "MrTidyRetrieval": {"korean", "ko"},
              "MultiLongDocRetrieval": {"ko"}, "MIRACLRetrieval": {"ko"},
              "PublicHealthQA": {"korean", "ko"}}
METRIC_LABEL = {"ndcg_at_10": "Avg nDCG@10", "recall_at_10": "Avg Recall@10",
                "mrr_at_10": "Avg MRR@10", "map_at_10": "Avg MAP@10"}


def cell(scores_by_split, task, metrics):
    """Per-task metric tuple under the benchmark's split/subset conventions."""
    want = KO_SUBSETS.get(task)
    per_split = {}
    for split, entries in scores_by_split.items():
        for e in entries:
            sub = e.get("hf_subset", "default")
            if (want is None and sub == "default") or (want is not None and sub in want):
                vals = [e.get(m) for m in metrics]
                if all(v is not None for v in vals):
                    per_split[split] = vals
    if not per_split:
        return None
    if task == "MultiLongDocRetrieval" and {"dev", "test"} <= per_split.keys():
        return [(per_split["dev"][i] + per_split["test"][i]) / 2 for i in range(len(metrics))]
    for pref in ("test", "dev"):
        if pref in per_split:
            return per_split[pref]
    return next(iter(per_split.values()))


def fmt_params(n):
    if not n:
        return "?"
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.0f}M"


def model_row(model_dir, metrics):
    got, meta = {}, {}
    for f in model_dir.glob("*/*.json"):
        if f.stem == "model_meta":
            meta = json.loads(f.read_text())
        elif f.stem in TASKS:
            c = cell(json.loads(f.read_text())["scores"], f.stem, metrics)
            if c:
                got[f.stem] = c
    if len(got) != len(TASKS):
        return None, [t for t in TASKS if t not in got]
    name = meta.get("name") or model_dir.name.replace("__", "/")
    frameworks = set(meta.get("framework") or [])
    mtype = "Late-interaction" if frameworks & {"PyLate", "ColBERT"} else "Dense"
    avgs = [sum(got[t][i] for t in TASKS) / len(TASKS) for i in range(len(metrics))]
    return {"name": name, "type": mtype, "params": fmt_params(meta.get("n_parameters")),
            "avgs": avgs}, None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--metrics", nargs="+", default=["ndcg_at_10", "recall_at_10"],
                        help="mteb metric keys; the table is sorted by the first one")
    args = parser.parse_args()

    rows = []
    for model_dir in sorted(p for p in args.results.iterdir() if p.is_dir()):
        row, missing = model_row(model_dir, args.metrics)
        if row is None:
            print(f"WARNING: {model_dir.name} missing {missing}; skipped")
        else:
            rows.append(row)
    rows.sort(key=lambda r: -r["avgs"][0])

    heads = [METRIC_LABEL.get(m, f"Avg {m}") for m in args.metrics]
    print(f"| Model | Type | Params | {' | '.join(heads)} |")
    print("|---|---|---:|" + "---:|" * len(args.metrics))
    best = rows[0]["name"] if rows else None
    for r in rows:
        label = r["name"]
        if label.startswith("nlpai-lab/"):
            label = f"**[{label}](https://huggingface.co/{label})**"
        vals = [f"{v:.4f}" for v in r["avgs"]]
        if r["name"] == best:
            vals = [f"**{v}**" for v in vals]
        print(f"| {label} | {r['type']} | {r['params']} | {' | '.join(vals)} |")


if __name__ == "__main__":
    main()
