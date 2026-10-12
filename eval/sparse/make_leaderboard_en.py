"""English leaderboard over eval/sparse/results/<model>/en: nDCG@10 on the 13 BEIR tasks plus MS MARCO,
the 14-task mean (Avg) and the mean without MS MARCO (BEIR-13), plus the sparsity numbers evaluate.py records,
averaged over the 14 tasks.

Tasks are listed in the order of SparseUp's table, and SparseUp's reported numbers are added as a reference row.
Rows are sorted by Avg, which is only shown for rows with every task on disk; incomplete rows go last. BEIR tasks
have only the default subset, so make_leaderboard.py's reader takes test, and dev for MS MARCO.

    uv run python eval/sparse/make_leaderboard_en.py
"""
import argparse
import json
from pathlib import Path

from make_leaderboard import FIELDS, cell

TASKS = ["ArguAna", "ClimateFEVER", "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "MSMARCO", "NFCorpus", "NQ",
         "QuoraRetrieval", "SCIDOCS", "SciFact", "TRECCOVID", "Touche2020"]
ABBR = ["Argu", "CFEV", "DBP", "FEV", "FiQA", "Hotp", "MSM", "NFC", "NQ", "Quora", "SciD", "SciF", "TCOV", "Touc"]
# https://www.linkup.so/blog/introducing-sparseup-by-linkup (reported to three decimals)
SPARSEUP_REPORTED = [0.568, 0.361, 0.415, 0.897, 0.469, 0.772, 0.431, 0.351, 0.571, 0.866, 0.184, 0.739, 0.784, 0.350]


def model_row(model_dir):
    got, meta = {}, {}
    for f in (model_dir / "en").glob("*.json"):
        if f.stem == "model_meta":
            meta = json.loads(f.read_text())
        elif f.stem in TASKS:
            c = cell(json.loads(f.read_text())["scores"], f.stem, FIELDS)
            if c:
                got[f.stem] = c
    if not got:
        return None
    name = meta.get("name") or model_dir.name.replace("__", "/")
    scores = [got[t][0] if t in got else None for t in TASKS]
    sparsity = [sum(got[t][i] for t in got) / len(got) for i in range(1, len(FIELDS))]
    return name, scores, sparsity


def means(scores):
    full = None if None in scores else sum(scores) / len(scores)
    rest = [v for t, v in zip(TASKS, scores) if t != "MSMARCO"]
    return full, (None if None in rest else sum(rest) / len(rest))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path,
                        default=Path(__file__).resolve().parent / "results")
    args = parser.parse_args()

    table = [("SparseUp (reported)", SPARSEUP_REPORTED, None)]
    table += [r for p in sorted(args.results.iterdir()) if p.is_dir() if (r := model_row(p)) is not None]
    table.sort(key=lambda row: (means(row[1])[0] is None, -(means(row[1])[0] or 0)))

    fmt = lambda v: "-" if v is None else f"{v:.4f}"  # noqa: E731
    name_w = max(len(name) for name, _, _ in table)
    header = (f"{'model':<{name_w}}  {'Avg':>6}  {'BEIR13':>6}  " + "  ".join(f"{a:>6}" for a in ABBR)
              + "   q_dim   c_dim   FLOPs")
    print(header)
    print("-" * len(header))
    for name, scores, sparsity in table:
        full, beir13 = means(scores)
        stats = f"  {sparsity[0]:6.1f}  {sparsity[1]:6.1f}  {sparsity[2]:6.2f}" if sparsity else ""
        print(f"{name:<{name_w}}  {fmt(full):>6}  {fmt(beir13):>6}  "
              + "  ".join(f"{fmt(v):>6}" for v in scores) + stats)


if __name__ == "__main__":
    main()
