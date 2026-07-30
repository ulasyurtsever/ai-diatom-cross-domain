"""Publication-quality revision figures (large fonts ->) from the
aggregated CSVs. Saves 300-dpi PNGs under results_revision/aggregated/figures/.
Robust to missing inputs.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config_revision as C
import os
def _repo_root():
    cands=[os.environ.get("REPO_ROOT"), os.getcwd(),
           os.path.dirname(os.path.abspath(__file__)),
           os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]
    for c in cands:
        if c and os.path.exists(os.path.join(c,"train.py")) and os.path.exists(os.path.join(c,"data","diatom_metadata.csv")):
            return os.path.abspath(c)
    return os.environ.get("REPO_ROOT") or os.getcwd()
REPO_ROOT=_repo_root()


plt.rcParams.update({"font.size": 15, "axes.titlesize": 17, "axes.labelsize": 15,
                     "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
                     "figure.dpi": 300, "savefig.bbox": "tight"})
AGG = os.path.join(REPO_ROOT, C.RESULTS_SUBDIR, "aggregated")
FIG = os.path.join(AGG, "figures"); os.makedirs(FIG, exist_ok=True)


def _short(m): return {"resnet50": "ResNet-50", "efficientnet_v2_s": "EffNetV2-S",
                       "convnext_tiny": "ConvNeXt-T", "swin_v2_t": "Swin V2-T",
                       "maxvit_t": "MaxViT-T"}.get(m, m)


def fig_multiseed():
    p = os.path.join(AGG, "multiseed_LOSO_mean_std.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p, index_col=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(d))
    ax.bar(x, d["F1_mean"], yerr=d["F1_std"].fillna(0), capsize=5, color="#2f5d9e")
    ax.set_xticks(x); ax.set_xticklabels([_short(m) for m in d.index], rotation=15)
    ax.set_ylabel("LOSO Macro-F1 (mean ± std)"); ax.set_title("Multi-seed stability (LOSO)")
    ax.grid(axis="y", alpha=.3); fig.savefig(os.path.join(FIG, "R_multiseed_LOSO.png")); plt.close(fig)


def fig_dg():
    p = os.path.join(AGG, "dg_comparison_LOSO_F1.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p, index_col=0)
    meths = [m for m in ["erm", "coral", "mixstyle"] if m in d.columns]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = np.arange(len(d)); w = 0.8 / max(len(meths), 1)
    for i, m in enumerate(meths):
        ax.bar(x + i * w, d[m], w, label=m.upper())
    ax.set_xticks(x + w * (len(meths) - 1) / 2)
    ax.set_xticklabels([_short(m) for m in d.index], rotation=15)
    ax.set_ylabel("LOSO Macro-F1"); ax.set_title("Dedicated DG methods vs ERM")
    ax.legend(); ax.grid(axis="y", alpha=.3)
    fig.savefig(os.path.join(FIG, "R_dg_comparison.png")); plt.close(fig)


def fig_ablation():
    p = os.path.join(AGG, "ablation_LOSO.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p, index_col=0)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(d))
    ax.bar(x, d["F1"], color="#3a9d6a")
    ax.set_xticks(x); ax.set_xticklabels(d.index, rotation=20, ha="right")
    ax.set_ylabel("LOSO Macro-F1"); ax.set_title("Augmentation ablation (ConvNeXt-T)")
    ax.grid(axis="y", alpha=.3); fig.savefig(os.path.join(FIG, "R_ablation.png")); plt.close(fig)


def fig_sensitivity():
    p = os.path.join(AGG, "sensitivity.csv")
    if not os.path.exists(p): return
    d = pd.read_csv(p)
    if "Factor" not in d: return
    facs = d["Factor"].unique()
    fig, axes = plt.subplots(1, len(facs), figsize=(4.2 * len(facs), 4))
    if len(facs) == 1: axes = [axes]
    for ax, f in zip(axes, facs):
        s = d[d["Factor"] == f]
        ax.errorbar(range(len(s)), s["F1_mean"], yerr=s.get("F1_std", 0), marker="o", capsize=4)
        ax.set_xticks(range(len(s))); ax.set_xticklabels(s["Value"].astype(str), rotation=20)
        ax.set_title(f); ax.set_ylabel("Macro-F1"); ax.grid(alpha=.3)
    fig.suptitle("Hyper-parameter sensitivity (ConvNeXt-T)")
    fig.savefig(os.path.join(FIG, "R_sensitivity.png")); plt.close(fig)


def fig_efficiency():
    p = os.path.join(AGG, "efficiency_metrics.csv")
    ms = os.path.join(AGG, "multiseed_LOSO_mean_std.csv")
    if not (os.path.exists(p) and os.path.exists(ms)): return
    e = pd.read_csv(p); d = pd.read_csv(ms, index_col=0)
    e = e.set_index("Model")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for m in e.index:
        if m in d.index:
            ax.scatter(e.loc[m, "Latency_ms_mean"], d.loc[m, "F1_mean"], s=120)
            ax.annotate(_short(m), (e.loc[m, "Latency_ms_mean"], d.loc[m, "F1_mean"]),
                        xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Latency (ms/image)"); ax.set_ylabel("LOSO Macro-F1")
    ax.set_title("Accuracy vs inference cost"); ax.grid(alpha=.3)
    fig.savefig(os.path.join(FIG, "R_efficiency_tradeoff.png")); plt.close(fig)


def main():
    for f in (fig_multiseed, fig_dg, fig_ablation, fig_sensitivity, fig_efficiency):
        try:
            f()
        except Exception as e:
            print(f"  ({f.__name__} skipped: {e})")
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
