
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = "../results/per_experiment"
FIG_DIR = "../figures"
MASTER_CSV = "../results/master_results.csv"

MODELS = ["resnet50", "efficientnet_v2_s", "convnext_tiny", "swin_v2_t", "maxvit_t"]
MODEL_LABELS = {"resnet50": "ResNet-50",
                "efficientnet_v2_s": "EfficientNetV2-S",
                "convnext_tiny": "ConvNeXt-Tiny",
                "swin_v2_t": "Swin V2-Tiny",
                "maxvit_t": "MaxViT-Tiny"}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 300, "savefig.dpi": 300})


def figure_2():
    df = pd.read_csv(MASTER_CSV)

    rows = []
    for m in MODELS:
        single = df[(df.Model == m) & (df.Mode == "SINGLE")]["Acc"].values
        mixed = df[(df.Model == m) & (df.Mode == "STANDARD")]["Acc"].values
        loso = df[(df.Model == m) & (df.Mode == "LOSO")]["Acc"].values
        rows.append({
            "Model": MODEL_LABELS[m],
            "Single-Source": single[0] * 100 if len(single) else np.nan,
            "Standard Mixed": mixed[0] * 100 if len(mixed) else np.nan,
            "LOSO (mean of 6 folds)": loso.mean() * 100 if len(loso) else np.nan,
            "LOSO std": loso.std(ddof=1) * 100 if len(loso) else np.nan,
            "Gap (Single − LOSO)": (single[0] - loso.mean()) * 100
                                   if len(single) and len(loso) else np.nan,
        })
    gdf = pd.DataFrame(rows).sort_values("LOSO (mean of 6 folds)",
                                          ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(gdf))
    width = 0.27
    ax.bar(x - width, gdf["Single-Source"], width, label="Single-Source (ADIAC)",
           color="#3b6db5", edgecolor="white", linewidth=0.8)
    ax.bar(x, gdf["Standard Mixed"], width, label="Standard Mixed (6-source pool)",
           color="#4da673", edgecolor="white", linewidth=0.8)
    ax.bar(x + width, gdf["LOSO (mean of 6 folds)"], width,
           yerr=gdf["LOSO std"], capsize=4,
           label="LOSO (mean ± SD across 6 held-out domains)",
           color="#c05a5a", edgecolor="white", linewidth=0.8,
           error_kw={"ecolor": "#3a1a1a", "alpha": 0.9})

    for i, row in gdf.iterrows():
        gap = row["Gap (Single − LOSO)"]
        sh = row["Single-Source"]
        lh = row["LOSO (mean of 6 folds)"]
        ax.annotate("", xy=(x[i] - width, sh), xytext=(x[i] + width, lh),
                    arrowprops=dict(arrowstyle="<->", color="#444", lw=1.2, alpha=0.7))
        ax.text(x[i], (sh + lh) / 2, f"Δ {gap:+.1f}%", ha="center", va="center",
                fontsize=9, color="#222",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#888",
                          lw=0.6, alpha=0.9))
        ax.text(x[i] - width, sh + 1, f"{sh:.1f}", ha="center", fontsize=8, color="#3b6db5")
        ax.text(x[i], row["Standard Mixed"] + 1, f"{row['Standard Mixed']:.1f}",
                ha="center", fontsize=8, color="#2f7d4a")
        ax.text(x[i] + width, lh + row["LOSO std"] + 1.5, f"{lh:.1f}",
                ha="center", fontsize=8, color="#aa3a3a")

    ax.set_xticks(x)
    ax.set_xticklabels(gdf["Model"])
    ax.set_xlabel("Architecture")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Figure 2.  Generalization gap across the three evaluation scenarios\n"
                 "(Δ = drop from Single-Source to LOSO mean; "
                 "error bars on LOSO show ±1 SD across K = 6 held-out domains)",
                 pad=12, fontsize=11)
    ax.grid(True, axis="y", alpha=0.35, linestyle="--", zorder=0)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "Figure_2_Generalization_Gap.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    gdf.to_csv(os.path.join(FIG_DIR, "Figure_2_gap_data.csv"), index=False)
    print(f"  wrote {out}")


def figure_7():
    rows = []
    for dom in ["ADIAC_Database", "AFD_Database", "DIA_Database",
                "DONA_Database", "FCE_LTER_Database", "LOIR_Database"]:
        sub = f"S2_LOSO_convnext_tiny_{dom}"
        pred_path = os.path.join(RESULTS_DIR, sub, "test_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        cols = df.columns.tolist()[2:]
        y_true = df["True_Label"].astype(int).values
        y_pred = df["Predicted_Label"].astype(int).values
        for t, p in zip(y_true, y_pred):
            if t < len(cols) and p < len(cols):
                rows.append((cols[t], cols[p]))

    big = pd.DataFrame(rows, columns=["True", "Pred"])
    misses = big[big["True"] != big["Pred"]]

    pair_counts = (misses.groupby(["True", "Pred"]).size()
                   .reset_index(name="count")
                   .sort_values("count", ascending=False))
    pair_counts.to_csv(os.path.join(FIG_DIR, "Figure_7_top_confusions.csv"),
                       index=False)
    print("  top 5 confusions:")
    print(pair_counts.head().to_string(index=False))

    top_true = misses["True"].value_counts().head(12).index.tolist()
    top_pred = misses["Pred"].value_counts().head(12).index.tolist()
    classes = sorted(set(top_true) | set(top_pred))

    sub = big[(big["True"].isin(classes)) & (big["Pred"].isin(classes))]
    cm = pd.crosstab(sub["True"], sub["Pred"], normalize="index") * 100
    cm = cm.reindex(index=classes, columns=classes, fill_value=0)

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(cm, annot=True, fmt=".0f", cmap="OrRd", ax=ax,
                cbar_kws={"label": "Row-normalized prediction frequency (%)"},
                linewidths=0.3, linecolor="white", vmin=0, vmax=80,
                annot_kws={"size": 8})
    ax.set_xlabel("Predicted genus / class")
    ax.set_ylabel("True genus / class")
    ax.set_title("Figure 7.  ConvNeXt-Tiny aggregated LOSO confusion structure\n"
                 "(rows = true label, columns = predicted label; "
                 "aggregated over K = 6 held-out domains; row-normalized)",
                 pad=12, fontsize=11)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "Figure_7_genus_confusion_convnext_aggregate.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    if not os.path.exists(MASTER_CSV):
        print(f"Missing {MASTER_CSV}. Run compute_statistics.py first.")
        return

    print("Figure 2 (generalization gap) ...")
    figure_2()

    print("\nFigure 7 (per-genus confusion) ...")
    figure_7()


if __name__ == "__main__":
    main()
