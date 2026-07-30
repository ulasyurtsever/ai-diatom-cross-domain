"""Aggregate ALL revision experiments into ready-to-write tables + summary.

Robust to partial results: only summarizes what exists. Reads:
  <repo>/results/per_experiment            (original, seed 42)
  <repo>/results_revision/multiseed/...    (extra seeds)
  <repo>/results_revision/dg/...           (CORAL, MixStyle)
  <repo>/results_revision/ablation/...
  <repo>/results_revision/sensitivity/...
  <repo>/results_revision/efficiency/efficiency_metrics.csv
Writes everything under  <repo>/results_revision/aggregated/ .
"""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
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


RR = os.path.join(REPO_ROOT, C.RESULTS_SUBDIR)
ORIG = os.path.join(REPO_ROOT, "results", "per_experiment")
OUT = os.path.join(RR, "aggregated")
os.makedirs(OUT, exist_ok=True)
MODELS = C.ALL_MODELS


def load(root):
    frames = []
    # glob.escape so paths containing [ ] (e.g. the project folder name) still match
    for f in glob.glob(os.path.join(glob.escape(root), "**", "final_metrics.csv"), recursive=True):
        try:
            frames.append(pd.read_csv(f))
        except Exception:
            pass
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _num(s):
    return pd.to_numeric(s, errors="coerce")


# --------------------------------------------------------------------------
def build_multiseed(summary, numbers):
    orig = load(ORIG)
    if not orig.empty:
        orig = orig.copy(); orig["Seed"] = 42; orig["Method"] = "erm"
    extra = load(os.path.join(RR, "multiseed"))
    df = pd.concat([d for d in [orig, extra] if not d.empty], ignore_index=True)
    if df.empty:
        return
    # If a seed was re-run on this machine, prefer it over the original (same-hardware).
    if 'Seed' in df:
        df = df.drop_duplicates(subset=['Model','Mode','Domain','Seed'], keep='last')
    df = df[df.get("Method", "erm").fillna("erm") == "erm"] if "Method" in df else df
    df["Acc"] = _num(df["Acc"]); df["F1"] = _num(df["F1"])

    # ---- LOSO: per (Model,Seed) mean over held-out domains, then across seeds
    loso = df[df["Mode"] == "LOSO"]
    if not loso.empty:
        per_seed = loso.groupby(["Model", "Seed"])[["Acc", "F1"]].mean().reset_index()
        agg = per_seed.groupby("Model")[["Acc", "F1"]].agg(["mean", "std"]).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]
        agg = agg.reindex([m for m in MODELS if m in agg.index])
        agg["n_seeds"] = per_seed.groupby("Model")["Seed"].nunique()
        agg.to_csv(os.path.join(OUT, "multiseed_LOSO_mean_std.csv"))
        summary.append("## Multi-seed LOSO (mean ± std over seeds)\n")
        summary.append(agg.to_markdown() + "\n")
        numbers["multiseed_LOSO"] = agg.reset_index().to_dict("records")

        # ranking stability: Kendall's W across seeds (rank models by F1)
        piv = per_seed.pivot(index="Seed", columns="Model", values="F1").dropna(axis=1)
        if piv.shape[0] >= 2 and piv.shape[1] >= 2:
            ranks = piv.rank(axis=1, ascending=False)
            n, k = ranks.shape
            Rj = ranks.sum(axis=0); S = ((Rj - Rj.mean()) ** 2).sum()
            W = 12 * S / (n ** 2 * (k ** 3 - k))
            numbers["kendall_W_seeds"] = round(float(W), 4)
            summary.append(f"\n**Ranking stability across seeds — Kendall's W = {W:.3f}** "
                           f"(1.0 = identical model ordering across seeds)\n")

        # Friedman/Nemenyi on seed-averaged per-domain F1 (domains x models)
        try:
            from scipy.stats import friedmanchisquare
            dm = (loso.groupby(["Domain", "Model"])["F1"].mean().reset_index()
                  .pivot(index="Domain", columns="Model", values="F1").dropna(axis=1))
            dm = dm[[m for m in MODELS if m in dm.columns]]
            if dm.shape[1] >= 3:
                chi, p = friedmanchisquare(*[dm[c].values for c in dm.columns])
                numbers["friedman_multiseed"] = {"chi2": round(float(chi), 4), "p": float(p)}
                summary.append(f"\n**Friedman (multi-seed avg per-domain F1): "
                               f"χ²={chi:.3f}, p={p:.4g}**\n")
                dm.to_csv(os.path.join(OUT, "multiseed_domain_model_F1.csv"))
        except Exception as e:
            summary.append(f"\n(Friedman skipped: {e})\n")

    # ---- SINGLE / STANDARD across seeds
    for scen in ["SINGLE", "STANDARD"]:
        s = df[df["Mode"] == scen]
        if s.empty:
            continue
        a = s.groupby("Model")[["Acc", "F1"]].agg(["mean", "std"]).round(4)
        a.columns = ["_".join(c) for c in a.columns]
        a.to_csv(os.path.join(OUT, f"multiseed_{scen}_mean_std.csv"))


# --------------------------------------------------------------------------
def build_dg(summary, numbers):
    dg = load(os.path.join(RR, "dg"))
    ms42 = load(os.path.join(RR, "multiseed", "seed_42"))
    orig = load(ORIG)
    erm_src = ms42 if (not ms42.empty and (ms42.get("Mode") == "LOSO").any()) else orig
    if dg.empty or erm_src.empty:
        return
    erm = erm_src[erm_src["Mode"] == "LOSO"].copy(); erm["Method"] = "erm"
    dg["Acc"] = _num(dg["Acc"]); dg["F1"] = _num(dg["F1"])
    erm["Acc"] = _num(erm["Acc"]); erm["F1"] = _num(erm["F1"])
    allm = pd.concat([erm[["Model", "Domain", "Method", "Acc", "F1"]],
                      dg[["Model", "Domain", "Method", "Acc", "F1"]]], ignore_index=True)
    tab = allm.groupby(["Model", "Method"])[["Acc", "F1"]].mean().round(4).reset_index()
    piv = tab.pivot(index="Model", columns="Method", values="F1")
    piv = piv.reindex([m for m in MODELS if m in piv.index])
    for meth in ["coral", "mixstyle"]:
        if meth in piv and "erm" in piv:
            piv[f"delta_{meth}"] = (piv[meth] - piv["erm"]).round(4)
    piv.to_csv(os.path.join(OUT, "dg_comparison_LOSO_F1.csv"))
    tab.to_csv(os.path.join(OUT, "dg_comparison_long.csv"), index=False)
    summary.append("\n## DG baselines vs ERM (LOSO mean Macro-F1)\n")
    summary.append(piv.to_markdown() + "\n")
    numbers["dg_comparison"] = piv.reset_index().to_dict("records")


# --------------------------------------------------------------------------
def build_ablation(summary, numbers):
    ab = load(os.path.join(RR, "ablation"))
    if ab.empty:
        return
    ab["Acc"] = _num(ab["Acc"]); ab["F1"] = _num(ab["F1"])
    key = "Variant" if "Variant" in ab else "Experiment"
    t = ab.groupby(key)[["Acc", "F1"]].mean().round(4)
    order = [v for v in C.ABLATION_VARIANTS if v in t.index] or list(t.index)
    t = t.reindex(order)
    if "full" in t.index:
        t["dF1_vs_full"] = (t["F1"] - t.loc["full", "F1"]).round(4)
    t.to_csv(os.path.join(OUT, "ablation_LOSO.csv"))
    summary.append("\n## Augmentation ablation (ConvNeXt-Tiny, LOSO mean)\n")
    summary.append(t.to_markdown() + "\n")
    numbers["ablation"] = t.reset_index().to_dict("records")


# --------------------------------------------------------------------------
def build_sensitivity(summary, numbers):
    se = load(os.path.join(RR, "sensitivity"))
    if se.empty:
        return
    se["Acc"] = _num(se["Acc"]); se["F1"] = _num(se["F1"])
    if "Factor" in se and "Value" in se:
        t = se.groupby(["Factor", "Value"])[["Acc", "F1"]].agg(["mean", "std"]).round(4)
        t.columns = ["_".join(c) for c in t.columns]
        t.to_csv(os.path.join(OUT, "sensitivity.csv"))
        summary.append("\n## Hyper-parameter sensitivity (ConvNeXt-Tiny, mean over folds)\n")
        summary.append(t.to_markdown() + "\n")
        numbers["sensitivity"] = t.reset_index().to_dict("records")


# --------------------------------------------------------------------------
def build_efficiency(summary, numbers):
    p = os.path.join(RR, "efficiency", "efficiency_metrics.csv")
    if not os.path.exists(p):
        return
    e = pd.read_csv(p)
    e.to_csv(os.path.join(OUT, "efficiency_metrics.csv"), index=False)
    summary.append("\n## Efficiency / inference metrics\n")
    summary.append(e.to_markdown(index=False) + "\n")
    numbers["efficiency"] = e.to_dict("records")


# --------------------------------------------------------------------------
def build_imbalance(summary, numbers):
    """ — WCE (baseline, from seed_42) vs Focal vs plain CE, ConvNeXt-Tiny LOSO."""
    im = load(os.path.join(RR, "imbalance"))
    ms42 = load(os.path.join(RR, "multiseed", "seed_42"))
    if im.empty or ms42.empty:
        return
    arch = C.PRIMARY_MODEL
    wce = ms42[(ms42["Model"] == arch) & (ms42["Mode"] == "LOSO")].copy()
    if wce.empty:
        return
    wce["Loss"] = "WCE"
    im["Loss"] = im["Group"].astype(str).str.replace("imbalance_", "", regex=False).str.upper()
    for d in (wce, im):
        d["Acc"] = _num(d["Acc"]); d["F1"] = _num(d["F1"])
    allm = pd.concat([wce[["Loss", "Domain", "Acc", "F1"]],
                      im[["Loss", "Domain", "Acc", "F1"]]], ignore_index=True)
    t = allm.groupby("Loss")[["Acc", "F1"]].mean().round(4)
    order = [l for l in ["WCE", "FOCAL", "CE"] if l in t.index] or list(t.index)
    t = t.reindex(order)
    if "WCE" in t.index:
        t["dF1_vs_WCE"] = (t["F1"] - t.loc["WCE", "F1"]).round(4)
    t.to_csv(os.path.join(OUT, "imbalance_loss_LOSO.csv"))
    summary.append("\n## Imbalance-loss comparison (ConvNeXt-Tiny, LOSO mean, seed 42)\n")
    summary.append(t.to_markdown() + "\n")
    numbers["imbalance_loss"] = t.reset_index().to_dict("records")


# --------------------------------------------------------------------------
def main():
    summary = ["# Revision (R1) — consolidated results\n",
               "_Auto-generated by aggregate_results.py. Every number the manuscript "
               "revision needs is in this folder (CSV) and summarized below._\n"]
    numbers = {}
    for fn in (build_multiseed, build_dg, build_ablation, build_sensitivity,
               build_efficiency, build_imbalance):
        try:
            fn(summary, numbers)
        except Exception as e:
            summary.append(f"\n(section {fn.__name__} failed: {e})\n")
    open(os.path.join(OUT, "REVISION_RESULTS_SUMMARY.md"), "w").write("\n".join(summary))
    json.dump(numbers, open(os.path.join(OUT, "revision_numbers.json"), "w"), indent=2)
    print("Aggregation written to", OUT)
    print("  - REVISION_RESULTS_SUMMARY.md  (human-readable)")
    print("  - revision_numbers.json        (machine-readable, all headline numbers)")


if __name__ == "__main__":
    main()
