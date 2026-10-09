"""Mean nDCG@10 vs. mean PPS of the rerankers in eval/cross-encoder/results, for the README.

Only models with all nine tasks are plotted (the listwise jina rerankers have no MLDR at 8192). Colour marks the
model-size bin, ours are drawn as bold-labelled stars. The x-axis is a piecewise log scale that stretches 20-26 and
compresses 26-50 pairs/s so nearby points stay readable. matplotlib is not a project dependency:

    uv run --with matplotlib python eval/cross-encoder/plot_pps_vs_ndcg.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.offsetbox import AnchoredOffsetbox, DrawingArea, HPacker, TextArea  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OUT = HERE.parents[1] / "assets" / "reranker_pps_vs_ndcg.png"
TASKS = ["Ko-StrategyQA", "AutoRAGRetrieval", "PublicHealthQA", "BelebeleRetrieval",
         "MIRACLRetrieval", "MrTidyRetrieval", "MultiLongDocRetrieval", "SQuADKorV1Retrieval", "LawIRKo"]

INK = "#1a2027"
MUTED = "#8a94a6"
# Times New Roman, else a metric-compatible substitute (TeX Gyre Termes, Nimbus Roman).
TIMES = ["Times New Roman", "TeX Gyre Termes", "Nimbus Roman", "STIXGeneral"]
# (upper bound in B, legend name, colour)
SIZE_BINS = [
    (0.5, "< 0.5B", "#b8646e"),
    (1.0, "0.5–1B", "#6f9fc2"),
    (3.0, "1–3B", "#d6a55c"),
    (float("inf"), "≥ 3B", "#4d4a7a"),
]

OURS = ("nlpai-lab/KURE-Reranker-base", "nlpai-lab/KURE-Reranker-nano")
# Parameter counts, summed from each checkpoint's safetensors header.
MODEL_SIZES = {
    "zeroentropy/zerank-2-reranker": 4_022_468_096,
    "lightonai/LightOn-rerank-PW-4B": 4_539_265_536,
    "mixedbread-ai/mxbai-rerank-large-v2": 1_543_714_304,
    "BAAI/bge-reranker-v2-m3": 567_755_777,
    "nvidia/llama-nemotron-rerank-1b-v2": 1_235_816_448,
    "nlpai-lab/LAMAR-600m": 567_755_777,
    "dragonkue/bge-reranker-v2-m3-ko": 567_755_777,
    "BAAI/bge-reranker-v2-gemma": 2_506_172_416,
    "upskyy/ko-reranker-8k": 567_755_777,
    "Dongjin-kr/ko-reranker": 559_891_457,
    "telepix/PIXIE-Spell-Reranker-Preview-0.6B": 595_777_536,
    "nlpai-lab/KURE-Reranker-nano": 149_323_009,
    "nlpai-lab/KURE-Reranker-base": 1_720_574_976,
    "Qwen/Qwen3-Reranker-8B": 8_188_548_096,
    "Qwen/Qwen3-Reranker-4B": 4_021_784_576,
    "Qwen/Qwen3-Reranker-0.6B": 595_776_512,
    "KaLM-Embedding/KaLM-Reranker-V1-Large-R2": 7_508_928_880,
}
# Short labels; the basename otherwise.
SHORT = {
    "Qwen/Qwen3-Reranker-8B": "Qwen3-8B",
    "Qwen/Qwen3-Reranker-4B": "Qwen3-4B",
    "Qwen/Qwen3-Reranker-0.6B": "Qwen3-0.6B",
    "KaLM-Embedding/KaLM-Reranker-V1-Large-R2": "KaLM-Large-R2",
    "zeroentropy/zerank-2-reranker": "zerank-2",
    "lightonai/LightOn-rerank-PW-4B": "LightOn-4B",
    "mixedbread-ai/mxbai-rerank-large-v2": "mxbai-large-v2",
    "BAAI/bge-reranker-v2-m3": "bge-v2-m3",
    "nvidia/llama-nemotron-rerank-1b-v2": "nemotron-1b",
    "dragonkue/bge-reranker-v2-m3-ko": "bge-v2-m3-ko",
    "BAAI/bge-reranker-v2-gemma": "bge-v2-gemma",
    "telepix/PIXIE-Spell-Reranker-Preview-0.6B": "PIXIE-0.6B",
}
# Label offsets in points where the default (right of the marker) would collide: dx < 0 puts the label left of
# the marker, right-aligned; "center" centres the name on dx.
OFFSET = {
    "zerank-2": (-16, -15, "center"),
    "Qwen3-8B": (10, 5),
    "Qwen3-4B": (-14, -14, "center"),
    "mxbai-large-v2": (10, 6),
    "LightOn-4B": (10, -5),
    "Qwen3-0.6B": (10, 5),
    "PIXIE-0.6B": (10, -5),
    "bge-v2-m3-ko": (-10, 0),
    "ko-reranker-8k": (-10, 0),
    "KURE-Reranker-base": (12, 0),
    "KURE-Reranker-nano": (12, 0),
}


def load_points() -> list[tuple[float, float, str]]:
    """(mean PPS, mean nDCG@10, model) for every model with nDCG@10 and PPS on all nine tasks."""
    pts = []
    for model_dir in sorted(p for p in RESULTS.glob("*/*") if p.is_dir()):
        files = [model_dir / f"{t}.json" for t in TASKS]
        if not all(f.exists() for f in files):
            continue
        rows = [json.loads(f.read_text()) for f in files]
        if any(r.get("pps") is None for r in rows):
            continue
        model = f"{model_dir.parent.name}/{model_dir.name}"
        pts.append((sum(r["pps"] for r in rows) / 9, sum(r["ndcg_at_10"] for r in rows) / 9, model))
    return pts


def size_color(billions: float) -> str:
    return next(color for upper, _, color in SIZE_BINS if billions < upper)


def label(ax, x, y, name, size, dx, dy, align=None, bold=False):
    """'name (size)' with the size small and muted."""
    weight = "bold" if bold else "normal"
    name_style = {"fontsize": 12, "color": INK, "fontweight": weight}
    size_style = {"fontsize": 9.5, "color": MUTED}
    if align == "center":
        name_t = ax.annotate(name, (x, y), textcoords="offset points", xytext=(dx, dy),
                             ha="center", va="center", **name_style)
        ax.annotate(f"({size})", xy=(1, 0), xycoords=name_t, textcoords="offset points", xytext=(3, 0),
                    ha="left", va="bottom", **size_style)
        return
    left = dx < 0
    first_text, first_style = (f"({size})", size_style) if left else (name, name_style)
    second_text, second_style = (name, name_style) if left else (f"({size})", size_style)
    first = ax.annotate(first_text, (x, y), textcoords="offset points", xytext=(dx, dy),
                        ha="right" if left else "left", va="center", **first_style)
    ax.annotate(second_text, xy=(0 if left else 1, 0), xycoords=first,
                textcoords="offset points", xytext=(-3 if left else 3, 0),
                ha="right" if left else "left", va="bottom", **second_style)


def piecewise_log_scale(ax):
    """log10 scale with per-band stretch: <20 x0.5, 20-26 x4, 26-50 x0.6, >50 x1."""
    a, b, c = np.log10([20, 26, 50])
    s0, s1, s2, s3 = 0.5, 4.0, 0.6, 1.0
    ub, uc = s1 * (b - a), s1 * (b - a) + s2 * (c - b)

    def fwd(x):
        t = np.log10(np.clip(x, 1e-9, None))
        return np.select([t < a, t < b, t < c], [s0 * (t - a), s1 * (t - a), ub + s2 * (t - b)], uc + s3 * (t - c))

    def inv(u):
        return 10 ** np.select([u < 0, u < ub, u < uc], [a + u / s0, a + u / s1, b + (u - ub) / s2], c + (u - uc) / s3)

    ax.set_xscale("function", functions=(fwd, inv))
    ticks = [10, 20, 25, 50, 100, 200, 500, 1000]
    ax.xaxis.set_major_locator(FixedLocator(ticks))
    ax.xaxis.set_major_formatter(FixedFormatter([f"{t:g}" for t in ticks]))
    ax.xaxis.set_minor_formatter(NullFormatter())


def size_legend(ax):
    """One row in a rounded box inside the plot, its right end at x=900 (before the 1000 tick)."""
    items = [TextArea("Model Size", textprops={"fontsize": 11, "color": INK})]
    for _, name, color in SIZE_BINS:
        dot = DrawingArea(11, 11)
        dot.add_artist(Circle((5.5, 5.5), 5, facecolor=color, edgecolor="black", linewidth=0.6))
        items.append(HPacker(children=[dot, TextArea(name, textprops={"fontsize": 11, "color": INK})],
                             pad=0, sep=4, align="center"))
    box = AnchoredOffsetbox(loc="upper right", child=HPacker(children=items, pad=0, sep=16, align="center"),
                            bbox_to_anchor=(900, 0.985), bbox_transform=ax.get_xaxis_transform(), frameon=True,
                            pad=0.45, borderpad=0)
    box.patch.set_boxstyle("round,pad=0.2,rounding_size=0.4")
    box.patch.set_edgecolor("#d5dae2")
    box.patch.set_linewidth(0.8)
    box.patch.set_facecolor("white")
    ax.add_artist(box)


def main() -> None:
    pts = load_points()
    plt.rcParams.update({"font.family": "serif", "font.serif": TIMES, "mathtext.fontset": "stix"})
    fig, ax = plt.subplots(figsize=(11, 6.8), dpi=150)

    for pps, ndcg, model in pts:
        ours = model in OURS
        size = MODEL_SIZES[model] / 1e9
        ax.scatter(pps, ndcg, s=330 if ours else 110, marker="*" if ours else "o", color=size_color(size),
                   edgecolor="black", linewidth=0.8, zorder=3)
        name = SHORT.get(model, model.split("/")[-1])
        label(ax, pps, ndcg, name, f"{size:.1f}B", *OFFSET.get(name, (10, 0)), bold=ours)

    piecewise_log_scale(ax)
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    ax.set_xlim(9, max(xs) * 2.3)
    ax.set_ylim(min(ys) - 0.02, max(ys) + 0.03)  # headroom for the legend
    ax.set_xlabel("Mean PPS (pairs/s)", fontsize=14, color=INK)
    ax.set_ylabel("Mean nDCG@10", fontsize=14, color=INK)
    size_legend(ax)

    ax.grid(True, which="major", color="#e6e9ef", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=12)

    fig.tight_layout()
    fig.savefig(OUT, bbox_inches="tight", facecolor="white")
    print(f"wrote {OUT} ({len(pts)} models)")


if __name__ == "__main__":
    main()
