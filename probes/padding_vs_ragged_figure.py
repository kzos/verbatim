# SPDX-License-Identifier: Apache-2.0
"""Why fixed-shape padding buys batch invariance, and what it costs.

Four panels, all from measurements taken on an NVIDIA B300 on 2026-09-13/14 and
published under rows/exploratory/ in the Verbatim repository. Nothing here is modelled
except the two row-count curves in panel A, which are the scheduler's own arithmetic
(fixed steps exactly B rows at any occupancy; ragged steps the live rows) and are
labelled as arithmetic rather than measurement.

Run:  python3 scripts/padding_vs_ragged_figure.py --out figures/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# --- measured, all from rows/exploratory/ -------------------------------------------
BUCKET = 128
S_FIXED, S_RAGGED = 42, 59  # one seed each, shortened durations
# The same server at frozen durations over three seeds,
# capacity-canonical-b300-2026-09-14.json. This spread is larger than the
# fixed-vs-ragged gap above, which is why no price is claimed.
S_BY_SEED = {"20260914": 20, "20260915": 24, "20260916": 46}
DIVERGE_FIXED = 0  # invariance-control-arm-b300-2026-09-14.json
DIVERGE_RAGGED_CONST = 66  # streams differing, max vs concurrency 1
DIVERGE_RAGGED_CHURN = 88  # invariance-churn-b300-2026-09-14.json
STREAMS = 256

# warm-up p95 reading spread, per rung, from the two matched ladders
SPREAD_FIXED = {20: 4.6, 23: 1.8, 27: 5.2, 32: 2.7, 37: 4.8, 40: 4.1, 41: 2.8, 42: 39.4, 43: 62.2}
SPREAD_RAGGED = {
    20: 0.5,
    23: 1.8,
    27: 0.4,
    32: 0.2,
    37: 4.1,
    43: 1.0,
    50: 8.9,
    58: 9.4,
    59: 5.0,
    60: 17.4,
    62: 13.0,
    67: 19.7,
}

INK, PAD_C, RAG_C, WARN = "#1b2430", "#2f6f8f", "#c2673b", "#9b2c2c"


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": "#d8dee6",
            "font.size": 11,
            "axes.titlesize": 12.5,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
        }
    )


def panel_rows(ax: plt.Axes) -> None:
    """What the encoder is asked to compute, per tick, as occupancy varies."""
    live = np.arange(0, BUCKET + 1)
    ax.fill_between(
        live, live, np.full_like(live, BUCKET), color=PAD_C, alpha=0.18, label="pad rows (the cost)"
    )
    ax.plot(
        live, np.full_like(live, BUCKET), color=PAD_C, lw=2.6, label=f"fixed: always {BUCKET} rows"
    )
    ax.plot(live, live, color=RAG_C, lw=2.6, label="ragged: exactly the live rows")
    for n, c in ((S_FIXED, PAD_C), (S_RAGGED, RAG_C)):
        ax.axvline(n, color=c, ls=":", lw=1.6, alpha=0.85)
        ax.annotate(
            f"S={n}", (n, BUCKET * 0.06), color=c, fontsize=10, ha="center", fontweight="semibold"
        )
    ax.set_xlabel("live sessions")
    ax.set_ylabel("rows stepped per tick")
    ax.set_title("A · The shape the encoder sees\n(scheduler arithmetic, not a measurement)")
    ax.legend(frameon=False, fontsize=9, loc="lower right")


def panel_divergence(ax: plt.Axes) -> None:
    labels = ["fixed\n(constant)", "fixed\n(churned)", "ragged\n(constant)", "ragged\n(churned)"]
    values = [DIVERGE_FIXED, DIVERGE_FIXED, DIVERGE_RAGGED_CONST, DIVERGE_RAGGED_CHURN]
    colors = [PAD_C, PAD_C, RAG_C, WARN]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v}",
            (bar.get_x() + bar.get_width() / 2, v + 2),
            ha="center",
            fontsize=11,
            fontweight="semibold",
        )
    ax.set_ylim(0, STREAMS * 0.42)
    ax.set_ylabel(f"streams differing from\nconcurrency 1  (of {STREAMS})")
    ax.set_title("B · Does the transcript depend on the neighbours?\n(measured, B300, bfloat16)")


def panel_price(ax: plt.Axes) -> None:
    """The comparison, and the noise that swamps it. No price is claimed."""
    seeds = sorted(S_BY_SEED)
    labels = ["fixed\n1 seed", "ragged\n1 seed"] + [f"fixed\nseed {s[-4:]}" for s in seeds]
    values = [S_FIXED, S_RAGGED] + [S_BY_SEED[s] for s in seeds]
    colors = [PAD_C, RAG_C] + [PAD_C] * len(seeds)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    for bar, v in zip(bars, values, strict=True):
        ax.annotate(
            f"{v}",
            (bar.get_x() + bar.get_width() / 2, v + 1.0),
            ha="center",
            fontsize=11,
            fontweight="semibold",
        )
    for bar in bars[2:]:
        bar.set_hatch("//")
        bar.set_edgecolor("white")
    lo, hi = min(S_BY_SEED.values()), max(S_BY_SEED.values())
    ax.axhspan(lo, hi, color=WARN, alpha=0.10)
    ax.annotate(
        f"same server, three seeds: {hi - lo} streams apart.\n"
        f"Larger than the {S_RAGGED - S_FIXED}-stream gap on the left,\n"
        "so no price is claimed.",
        (3.0, hi * 1.19),
        ha="center",
        fontsize=9.5,
        color=WARN,
        fontweight="semibold",
    )
    ax.set_ylim(0, hi * 1.45)
    ax.set_ylabel("sustained concurrent streams")
    ax.set_title("C \u00b7 Why no price is published\nthe seed spread exceeds the effect")
    ax.tick_params(axis="x", labelsize=8.5)


def panel_spread(ax: plt.Axes) -> None:
    for data, colour, name in ((SPREAD_FIXED, PAD_C, "fixed"), (SPREAD_RAGGED, RAG_C, "ragged")):
        xs = sorted(data)
        ax.plot(xs, [data[x] for x in xs], marker="o", ms=5, lw=2.2, color=colour, label=name)
    ax.axhline(10.0, color=WARN, ls="--", lw=1.6)
    ax.annotate("10% convergence tolerance", (21, 11.6), color=WARN, fontsize=9.5)
    ax.set_yscale("log")
    ax.set_xlabel("concurrent streams")
    ax.set_ylabel("warm-up p95 spread\n(% of median)")
    ax.set_title("D · What actually ends a run\nvariance, not latency")
    ax.legend(frameon=False, fontsize=9.5, loc="upper left")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    style()
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10.5))
    panel_rows(axes[0][0])
    panel_divergence(axes[0][1])
    panel_price(axes[1][0])
    panel_spread(axes[1][1])
    fig.suptitle(
        "Fixed-shape padding and batch invariance — NVIDIA B300, bfloat16, 160 ms, 256 utterances",
        fontsize=14.5,
        fontweight="semibold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.005,
        "Verbatim · rows/exploratory/ · the 29% price published on 14 Sep is WITHDRAWN: "
        "the seed-to-seed spread on one unchanged configuration exceeds it",
        ha="center",
        fontsize=9,
        color="#5a6672",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "padding-vs-ragged-b300.png"
    fig.savefig(path, dpi=170)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
