
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats


def _ddir():
    """Return the repository data/ directory (with trailing slash), located by
    walking up from this script or the CWD; falls back to ../data/."""
    from pathlib import Path as _P
    for b in [_P(__file__).resolve().parent, *_P(__file__).resolve().parents, _P.cwd(), *_P.cwd().parents]:
        if (b / "data" / "diatom_metadata.csv").exists():
            return str(b / "data") + "/"
    return "../data/"


RESULTS_DIR = "../results/per_experiment"
FIG_DIR = "../figures"
PER_FOLD_CSV = "../results/per_fold_loso.csv"
METADATA_CSV = _ddir()+"diatom_metadata.csv"

MODELS = ["resnet50", "efficientnet_v2_s", "convnext_tiny", "swin_v2_t", "maxvit_t"]
MODEL_LABELS = {"resnet50": "ResNet-50",
                "efficientnet_v2_s": "EfficientNetV2-S",
                "convnext_tiny": "ConvNeXt-Tiny",
                "swin_v2_t": "Swin V2-Tiny",
                "maxvit_t": "MaxViT-Tiny"}
DOMAINS = ["ADIAC_Database", "AFD_Database", "DIA_Database",
           "DONA_Database", "FCE_LTER_Database", "LOIR_Database"]
DOMAIN_LABELS = {d: d.replace("_Database", "") for d in DOMAINS}

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "figure.dpi": 300, "savefig.dpi": 300})


def figure_3(per_fold):
    """LOSO heatmap with 95% bootstrap CI annotations."""
    acc = per_fold.pivot(index="Model", columns="Domain", values="Acc").loc[MODELS, DOMAINS]
    lo = per_fold.pivot(index="Model", columns="Domain", values="Acc_CI_lo").loc[MODELS, DOMAINS]
    hi = per_fold.pivot(index="Model", columns="Domain", values="Acc_CI_hi").loc[MODELS, DOMAINS]

    annot = np.empty(acc.shape, dtype=object)
    for i in range(acc.shape[0]):
        for j in range(acc.shape[1]):
            annot[i, j] = (f"{acc.iat[i, j]*100:.1f}\n"
                           f"[{lo.iat[i, j]*100:.1f}, {hi.iat[i, j]*100:.1f}]")

    fig, ax = plt.subplots(figsize=(11, 5.5))
    sns.heatmap(acc.values * 100, annot=annot, fmt="", cmap=sns.color_palette("YlGnBu", as_cmap=True),
                xticklabels=[DOMAIN_LABELS[d] for d in DOMAINS],
                yticklabels=[MODEL_LABELS[m] for m in MODELS],
                cbar_kws={"label": "Held-out-domain accuracy (%)"},
                linewidths=0.5, linecolor="white",
                vmin=0, vmax=80, annot_kws={"size": 8}, ax=ax)
    ax.set_xlabel("Held-out target domain (LOSO)")
    ax.set_ylabel("Architecture")
    ax.set_title("Figure 3.  LOSO cross-domain accuracy and 95 % bootstrap CIs\n"
                 "(2 000 resamples of the held-out test set per cell)",
                 pad=12, fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0)
    plt.setp(ax.get_xticklabels(), rotation=0)
    plt.tight_layout()

    out = os.path.join(FIG_DIR, "Figure_3_LOSO_heatmap_with_CI.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def class_size_vs_f1_for_loso(meta_path):
    """Build a (class, domain, n_train, F1) table for ConvNeXt-Tiny on LOSO.
    F1 is the per-class F1 on the held-out domain and n_train is the
    number of images contributed by the five other domains."""
    md = pd.read_csv(meta_path)
    md_dg = md[md["is_dg_class"]].copy()

    # n_train per (held-out domain, class index)
    n_train = {}
    for dom in DOMAINS:
        cnts = md_dg[md_dg["domain"] != dom]["label_idx"].value_counts()
        for c, n in cnts.items():
            n_train[(dom, int(c))] = int(n)

    rows = []
    for dom in DOMAINS:
        sub = f"S2_LOSO_convnext_tiny_{dom}"
        pred_path = os.path.join(RESULTS_DIR, sub, "test_predictions.csv")
        if not os.path.exists(pred_path):
            continue
        df = pd.read_csv(pred_path)
        y_true = df["True_Label"].astype(int).values
        y_pred = df["Predicted_Label"].astype(int).values
        class_columns = df.columns.tolist()[2:]
        for c in sorted(set(y_true.tolist())):
            tp = int(((y_true == c) & (y_pred == c)).sum())
            fp = int(((y_true != c) & (y_pred == c)).sum())
            fn = int(((y_true == c) & (y_pred != c)).sum())
            if (tp + fp) == 0 or (tp + fn) == 0:
                f1 = 0.0
            else:
                prec = tp / (tp + fp)
                rec = tp / (tp + fn)
                f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)
            name = class_columns[c] if c < len(class_columns) else "?"
            rows.append({"Model": "convnext_tiny",
                         "Domain": dom,
                         "Class_idx": c,
                         "Class_name": name,
                         "N_train_imgs": n_train.get((dom, c), 0),
                         "F1": f1})
    return pd.DataFrame(rows)


def figure_4(df_classes):
    """OLS regression of mean Macro-F1 on log10(N_train+1) with 95% CI band."""
    agg = (df_classes.groupby(["Class_idx", "Class_name"])
                     .agg(F1_mean=("F1", "mean"),
                          F1_std=("F1", "std"),
                          N_train_mean=("N_train_imgs", "mean"))
                     .reset_index())
    agg = agg[agg["N_train_mean"] > 0]

    x = np.log10(agg["N_train_mean"].values + 1)
    y = agg["F1_mean"].values
    slope, intercept, r, p, _ = stats.linregress(x, y)
    print(f"  OLS: F1 = {intercept:.3f} + {slope:.3f}*log10(N+1)  "
          f"r = {r:.3f}, p = {p:.4f}")

    fig, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(agg["N_train_mean"], agg["F1_mean"],
                    s=80, c=agg["F1_mean"], cmap="viridis",
                    edgecolors="white", linewidths=0.8, zorder=3)
    plt.colorbar(sc, ax=ax).set_label("Mean LOSO Macro-F1")

    xg = np.linspace(x.min(), x.max(), 100)
    yg = intercept + slope * xg
    label = (f"OLS fit (log10 scale): F1 = {intercept:.3f} + {slope:.3f}"
             f"·log10(N+1),  r = {r:.3f}, p = {p:.3f}")
    ax.plot(10 ** xg - 1, yg, color="#C00000", lw=2, label=label, zorder=4)

    # Parametric 95% CI on the regression line
    n = len(x)
    mse = np.sum((y - (intercept + slope * x)) ** 2) / (n - 2)
    x_bar = x.mean()
    sxx = np.sum((x - x_bar) ** 2)
    se_y = np.sqrt(mse * (1 / n + (xg - x_bar) ** 2 / sxx))
    t_crit = stats.t.ppf(0.975, df=n - 2)
    ax.fill_between(10 ** xg - 1, yg - t_crit * se_y, yg + t_crit * se_y,
                    color="#C00000", alpha=0.12, zorder=2,
                    label="95 % CI of the linear fit")
    ax.set_xscale("log")
    ax.set_xlabel("Number of training images per class (log scale)\n"
                  "(images from the five source domains other than the held-out one)")
    ax.set_ylabel("Mean LOSO Macro-F1 across the 6 held-out domains")
    ax.set_title("Figure 4.  Class-imbalance vs. LOSO Macro-F1 for ConvNeXt-Tiny",
                 pad=10, fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.4, zorder=1)
    ax.set_ylim(-0.05, 1.05)

    for _, row in agg.iterrows():
        ax.annotate(row["Class_name"], xy=(row["N_train_mean"], row["F1_mean"]),
                    xytext=(5, 6), textcoords="offset points",
                    fontsize=7.5, color="#333333", alpha=0.85, zorder=5)

    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    plt.tight_layout()
    out = os.path.join(FIG_DIR, "Figure_4_class_imbalance_vs_F1.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    if not os.path.exists(PER_FOLD_CSV):
        print(f"Missing {PER_FOLD_CSV}. Run compute_statistics.py first.")
        return

    per_fold = pd.read_csv(PER_FOLD_CSV)

    print("Figure 3 (LOSO heatmap with bootstrap CI) ...")
    figure_3(per_fold)

    print("\nFigure 4 (class-imbalance OLS fit) ...")
    df_classes = class_size_vs_f1_for_loso(METADATA_CSV)
    df_classes.to_csv(os.path.join(FIG_DIR, "class_imbalance_data.csv"), index=False)
    figure_4(df_classes)


if __name__ == "__main__":
    main()
