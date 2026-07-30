"""
make_figure5.py
===============

Generate Figure 5 of the manuscript

    *Cross-Domain Evaluation of Modern Deep Learning Architectures for
    Microscopic Diatom Classification*

Figure 5 is the cost-accuracy trade-off scatter of the five evaluated
backbones across the three protocols (Single-Source, Standard Mixed,
Leave-One-Source-Out). Each backbone contributes one point per protocol
(scenario colour) with a backbone-specific marker; dotted lines connect a
given backbone's three points so that the per-architecture trajectory
across protocols is visible.

Inputs
------
- ``../results/master_results.csv`` — one row per experiment, with
  columns ``Model``, ``Mode`` ∈ {SINGLE, STANDARD, LOSO}, ``Acc``,
  ``Time_Sec``. SINGLE and LOSO have one row per source domain; STANDARD
  has a single row per backbone.

Output
------
- ``../figures/Figure_5_Cost_Accuracy_Tradeoff.png`` (300 dpi).
- ``../figures/Figure_5_data.csv`` — the 15-row aggregated table that
  drives the scatter, released for transparency.

Usage
-----
::

    cd code
    python make_figure5.py
"""

import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

RESULTS_CSV = "../results/master_results.csv"
FIG_DIR = "../figures"

MODELS = ["resnet50", "efficientnet_v2_s", "convnext_tiny", "swin_v2_t", "maxvit_t"]
MODEL_LABELS = {"resnet50": "ResNet-50",
                "efficientnet_v2_s": "EfficientNetV2-S",
                "convnext_tiny": "ConvNeXt-Tiny",
                "swin_v2_t": "Swin V2-Tiny",
                "maxvit_t": "MaxViT-Tiny"}

SCENARIOS = ["SINGLE", "STANDARD", "LOSO"]
SCENARIO_LABELS = {"SINGLE": "Single-Source",
                   "STANDARD": "Standard Mixed",
                   "LOSO": "LOSO"}

# Scenario palette: cool-greens for easiest, blue for mixed, red for LOSO.
SCENARIO_COLORS = {"Single-Source": "#2ecc71",
                   "Standard Mixed": "#3498db",
                   "LOSO": "#e74c3c"}

# One distinct marker per backbone (kept stable across all three scenarios
# so that the connecting line tells a per-architecture story).
MARKERS = {"resnet50": "o",
           "efficientnet_v2_s": "s",
           "convnext_tiny": "D",
           "swin_v2_t": "X",
           "maxvit_t": "^"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 300, "savefig.dpi": 300})


def aggregate(df):
    """Return one (Model, Scenario, mean_Acc, mean_Time) row per cell.

    SINGLE and LOSO are averaged over the six source domains; STANDARD
    is left as-is because it has a single row per backbone.
    """
    rows = []
    for scenario in SCENARIOS:
        sub = df[df["Mode"] == scenario]
        for model in MODELS:
            mdf = sub[sub["Model"] == model]
            if len(mdf) == 0:
                continue
            rows.append({"Model": model,
                         "Scenario": SCENARIO_LABELS[scenario],
                         "Acc": mdf["Acc"].mean(),
                         "Time_Sec": mdf["Time_Sec"].mean()})
    return pd.DataFrame(rows)


def figure_5(agg):
    """Cost-accuracy trade-off scatter with per-architecture trajectories."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Scatter — colour by scenario, marker by architecture.
    sns.scatterplot(
        data=agg,
        x="Time_Sec", y="Acc",
        hue="Scenario", style="Model",
        markers=MARKERS, s=120,
        palette=SCENARIO_COLORS,
        hue_order=[SCENARIO_LABELS[s] for s in SCENARIOS],
        alpha=0.55, edgecolor="black",
        ax=ax,
    )

    # Connect a given backbone's three points with a dotted line so that
    # the per-architecture trajectory across scenarios is visible.
    for model in MODELS:
        mdf = (agg[agg["Model"] == model]
               .sort_values("Acc", ascending=False))
        ax.plot(mdf["Time_Sec"], mdf["Acc"],
                color="black", linestyle=":", alpha=0.5, zorder=0)

    ax.set_xlabel("Training Time (Seconds) [Efficiency]")
    ax.set_ylabel("Accuracy [Performance]")
    ax.grid(True, linestyle="--", alpha=0.8)

    # Replace raw backbone slugs in the legend with readable labels and
    # place it on the right (matches the paper figure layout).
    handles, labels = ax.get_legend_handles_labels()
    pretty = [MODEL_LABELS.get(lbl, lbl) for lbl in labels]
    ax.legend(handles, pretty, loc="lower right", ncol=1, frameon=True)

    plt.tight_layout()
    return fig


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df = pd.read_csv(RESULTS_CSV)
    agg = aggregate(df)

    out_csv = os.path.join(FIG_DIR, "Figure_5_data.csv")
    agg.to_csv(out_csv, index=False)
    print(f"[ok] wrote {out_csv}")

    fig = figure_5(agg)
    out_png = os.path.join(FIG_DIR, "Figure_5_Cost_Accuracy_Tradeoff.png")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_png}")


if __name__ == "__main__":
    main()
