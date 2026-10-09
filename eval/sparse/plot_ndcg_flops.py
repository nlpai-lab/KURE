"""Avg nDCG@10 vs avg FLOPs of the sparse models in eval/sparse/results, for the README.

Reads the same per-task cells as make_leaderboard.py (MTEB(kor, v2) conventions). Styled like the KURE reranker
figure: colour = model-size bin, label = short name plus its size in small muted type, ours drawn as a star.
matplotlib is not a project dependency:

    uv run --with matplotlib python eval/sparse/plot_ndcg_flops.py
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import make_leaderboard as L  # noqa: E402

INK = "#1a2027"
MUTED = "#8a94a6"
# Times New Roman, else a metric-compatible substitute (TeX Gyre Termes, Nimbus Roman).
TIMES = ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"]
# (upper bound in B, legend name, colour) -- the reranker figure's palette, with the first two colours swapped so
# KURE-Sparse-base (0.5-1B) is the red one
SIZE_BINS = [
    (0.5, "< 0.5B", "#6f9fc2"),
    (1.0, "0.5–1B", "#b8646e"),
    (3.0, "1–3B", "#d6a55c"),
    (float("inf"), "≥ 3B", "#4d4a7a"),
]

OURS = ("KURE-Sparse-base", "KURE-Sparse-nano")
# results directory -> (label, parameters; HF safetensors totals, ours counted from the checkpoint)
MODELS = {
    "nlpai-lab__KURE-Sparse-base": ("KURE-Sparse-base", 593e6),
    "nlpai-lab__KURE-Sparse-nano": ("KURE-Sparse-nano", 149e6),
    "omai-research__milco-650m": ("milco-650m", 593e6),
    "telepix__PIXIE-Splade-v1.5": ("PIXIE-Splade-v1.5", 149e6),
    "yjoonjang__splade-ko-v1": ("splade-ko-v1", 149e6),
    "telepix__PIXIE-Splade-v1.0": ("PIXIE-Splade-v1.0", 149e6),
    "opensearch-project__opensearch-neural-sparse-encoding-multilingual-v1":
        ("opensearch-neural-sparse-multilingual-v1", 167e6),
}
# dense references drawn as horizontal dashed lines (--no-reference to drop them):
# label, eval/dense/results dir, label above or below its line
REFERENCES = [("Qwen3-Embedding-8B", "Qwen__Qwen3-Embedding-8B", "above"),
              ("pplx-embed-v1-4b", "perplexity-ai__pplx-embed-v1-4b", "below")]
DENSE_RESULTS = Path(__file__).resolve().parents[1] / "dense" / "results"

# label offsets in points where the default (right of the marker) would collide; dx < 0 = left, right-aligned
OFFSET = {"KURE-Sparse-base": (-14, 0), "KURE-Sparse-nano": (-14, -11), "milco-650m": (10, -1),
          "PIXIE-Splade-v1.5": (10, -12), "splade-ko-v1": (-10, -10)}
# labels that sit on a reference line get a white halo so the line does not cross the text
HALO = {"boxstyle": "square,pad=0.1", "facecolor": "white", "edgecolor": "none"}
HALOED = {"milco-650m"}


def dense_average(model_dir: Path) -> float:
    """Avg nDCG@10 of an eval/dense/results model under the same per-task conventions."""
    cells = [L.cell(json.loads(f.read_text())["scores"], f.stem, ["ndcg_at_10"])
             for f in model_dir.glob("**/*.json") if f.stem in L.TASKS]
    cells = [c for c in cells if c]
    if len(cells) != len(L.TASKS):
        raise SystemExit(f"{model_dir.name}: {len(cells)}/{len(L.TASKS)} tasks")
    return sum(c[0] for c in cells) / len(cells)


def averages(model_dir: Path) -> tuple[float, float]:
    got = {}
    for f in model_dir.glob("**/*.json"):
        if f.stem in L.TASKS:
            cell = L.cell(json.loads(f.read_text())["scores"], f.stem, L.FIELDS)
            if cell:
                got[f.stem] = cell
    if len(got) != len(L.TASKS):
        raise SystemExit(f"{model_dir.name}: {len(got)}/{len(L.TASKS)} tasks")
    ndcg = sum(c[0] for c in got.values()) / len(got)
    flops = sum(c[3] for c in got.values()) / len(got)
    return ndcg, flops


def size_color(billions: float) -> str:
    return next(color for upper, _, color in SIZE_BINS if billions < upper)


def label(ax, x, y, name, size, dx, dy, bold=False, halo=False):
    """'name (size)' with the size small and muted; dx < 0 puts it left of the marker, right-aligned."""
    left = dx < 0
    name_style = {"fontsize": 13, "color": INK, "fontweight": "bold" if bold else "normal"}
    size_style = {"fontsize": 10.5, "color": MUTED}
    first_text, first_style = (f"({size})", size_style) if left else (name, name_style)
    second_text, second_style = (name, name_style) if left else (f"({size})", size_style)
    first = ax.annotate(first_text, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha="right" if left else "left", va="center", bbox=HALO if halo else None, zorder=4,
                        **first_style)
    ax.annotate(second_text, xy=(0 if left else 1, 0), xycoords=first, textcoords="offset points",
                xytext=(-3 if left else 3, 0), ha="right" if left else "left", va="bottom",
                bbox=HALO if halo else None, zorder=4,
                **second_style)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results", type=Path, default=Path(__file__).resolve().parent / "results")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "assets" / "sparse_ndcg_flops.png")
    parser.add_argument("--no-reference", action="store_true", help="Leave out the dense reference lines")
    args = parser.parse_args()

    pts = [(*averages(args.results / d), name, params) for d, (name, params) in MODELS.items()]
    plt.rcParams.update({"font.family": "serif", "font.serif": TIMES, "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(9, 6), dpi=150)

    for ndcg, flops, name, params in pts:
        ours = name in OURS
        ax.scatter(flops, ndcg * 100, s=330 if ours else 110, marker="*" if ours else "o",
                   color=size_color(params / 1e9), edgecolor="black", linewidth=0.8, zorder=3)
        label(ax, flops, ndcg * 100, name, f"{params / 1e9:.1f}B", *OFFSET.get(name, (10, 0)), bold=ours,
              halo=name in HALOED)

    ax.set_xscale("log")
    ticks = [5, 10, 20, 40]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    flops_all = [f for _, f, _, _ in pts]
    ndcg_all = [n * 100 for n, _, _, _ in pts]
    ax.set_xlim(min(min(flops_all) * 0.8, ticks[0] * 0.9), max(flops_all) * 2.0)  # keep the first tick
    ax.set_ylim(min(ndcg_all) - 2, max(ndcg_all) + 4)  # headroom for the legend

    for ref_name, ref_dir, side in [] if args.no_reference else REFERENCES:
        ref = dense_average(DENSE_RESULTS / ref_dir) * 100
        ax.axhline(ref, color="#7d848f", linestyle=(0, (3, 2)), linewidth=1.3, zorder=2)
        # at the left end, which is empty; the right end runs under our models
        above = side == "above"
        ax.annotate(f"{ref_name} (dense)", xy=(0, ref), xycoords=("axes fraction", "data"),
                    textcoords="offset points", xytext=(6, 4 if above else -7), ha="left",
                    va="bottom" if above else "top", fontsize=11.5, color="#7d848f")

    ax.set_xlabel("Avg. FLOPs", fontsize=16, color=INK)
    ax.set_ylabel("Avg. nDCG@10", fontsize=16, color=INK)
    # size legend: one row in a rounded box at the top, only the bins on the plot
    shown = {size_color(p / 1e9) for *_, p in pts}
    items = [TextArea("Model Size", textprops={"fontsize": 12, "color": INK})]
    for _, name, color in SIZE_BINS:
        if color not in shown:
            continue
        dot = DrawingArea(12, 12)
        dot.add_artist(Circle((6, 6), 5.5, facecolor=color, edgecolor="black", linewidth=0.6))
        items.append(HPacker(children=[dot, TextArea(name, textprops={"fontsize": 12, "color": INK})],
                             pad=0, sep=4, align="center"))
    box = AnchoredOffsetbox(loc="upper left", child=HPacker(children=items, pad=0, sep=16, align="center"),
                            frameon=True, pad=0.45, borderpad=0.6)
    box.patch.set_boxstyle("round,pad=0.2,rounding_size=0.4")
    box.patch.set_edgecolor("#d5dae2")
    box.patch.set_linewidth(0.8)
    box.patch.set_facecolor("white")
    ax.add_artist(box)

    ax.grid(True, which="major", color="#e6e9ef", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=14)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    for ndcg, flops, name, params in sorted(pts, key=lambda p: -p[0]):
        print(f"{name:45s} nDCG@10 {ndcg:.4f}  FLOPs {flops:.1f}  {params / 1e6:.0f}M")
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
