"""Leaderboard over eval/sparse/results/<model>/ko: Avg nDCG@10 (main) plus the sparsity numbers eval.py
records -- avg query/corpus active dims and avg_flops -- one aligned row per model.

The result files are mteb's ResultCache format, identical to eval/dense, so the same MTEB(kor, v2)
split/subset conventions apply (Korean subsets only; Belebele = kor_Hang-kor_Hang; MLDR = mean of
dev/test; MIRACL = dev). The sparsity stats sit on the very same score entries as the metrics, so
they are read under the identical convention and averaged over the tasks a model reports. Unlike
eval/dense, a model with fewer than nine tasks is still shown (its `n` and averages cover what is on
disk), so the table is readable while an eval is mid-run.

    uv run python eval/sparse/make_leaderboard.py
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
# main metric first; the rest are the sparsity numbers, averaged the same way.
FIELDS = ["ndcg_at_10", "query_active_dims", "corpus_active_dims", "avg_flops"]
HEADS = ["Avg nDCG@10", "Avg q_dim", "Avg c_dim", "Avg FLOPs"]
FMT = ["{:.4f}", "{:.1f}", "{:.1f}", "{:.2f}"]


def cell(scores_by_split, task, fields):
    """Per-task field tuple under the benchmark's split/subset conventions."""
    want = KO_SUBSETS.get(task)
    per_split = {}
    for split, entries in scores_by_split.items():
        for e in entries:
            sub = e.get("hf_subset", "default")
            if (want is None and sub == "default") or (want is not None and sub in want):
                vals = [e.get(f) for f in fields]
                if all(v is not None for v in vals):
                    per_split[split] = vals
    if not per_split:
        return None
    if task == "MultiLongDocRetrieval" and {"dev", "test"} <= per_split.keys():
        return [(per_split["dev"][i] + per_split["test"][i]) / 2 for i in range(len(fields))]
    for pref in ("test", "dev"):
        if pref in per_split:
            return per_split[pref]
    return next(iter(per_split.values()))


def model_row(model_dir):
    got, meta = {}, {}
    for f in (model_dir / "ko").glob("*.json"):
        if f.stem == "model_meta":
            meta = json.loads(f.read_text())
        elif f.stem in TASKS:
            c = cell(json.loads(f.read_text())["scores"], f.stem, FIELDS)
            if c:
                got[f.stem] = c
    if not got:
        return None
    name = meta.get("name") or model_dir.name.replace("__", "/")
    avgs = [sum(got[t][i] for t in got) / len(got) for i in range(len(FIELDS))]
    return {"name": name, "n": len(got), "avgs": avgs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    rows = [r for p in sorted(args.results.iterdir()) if p.is_dir()
            if (r := model_row(p)) is not None]
    rows.sort(key=lambda r: -r["avgs"][0])

    cell_strs = [[fmt.format(v) for fmt, v in zip(FMT, r["avgs"])] for r in rows]
    name_w = max([len("model")] + [len(r["name"]) for r in rows])
    widths = [max([len(h)] + [len(cs[i]) for cs in cell_strs]) for i, h in enumerate(HEADS)]
    header = (f"{'#':>2}  {'model':<{name_w}}  {'n':>2}  "
              + "  ".join(f"{h:>{w}}" for h, w in zip(HEADS, widths)))
    print(header)
    print("-" * len(header))
    for i, (r, cs) in enumerate(zip(rows, cell_strs), 1):
        cells = "  ".join(f"{c:>{w}}" for c, w in zip(cs, widths))
        print(f"{i:>2}  {r['name']:<{name_w}}  {r['n']:>2}  {cells}")


if __name__ == "__main__":
    main()
