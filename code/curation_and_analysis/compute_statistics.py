
import os
from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


RESULTS_DIR = "../results/per_experiment"
OUT_DIR = "../results"

MODELS = ["resnet50", "efficientnet_v2_s", "convnext_tiny", "swin_v2_t", "maxvit_t"]
DOMAINS = ["ADIAC_Database", "AFD_Database", "DIA_Database",
           "DONA_Database", "FCE_LTER_Database", "LOIR_Database"]

B = 2000
ALPHA = 0.05
Q_ALPHA = 2.728     # studentised range at alpha=0.05 for k=5 architectures


def load_master():
    rows = []
    for d in sorted(os.listdir(RESULTS_DIR)):
        f = os.path.join(RESULTS_DIR, d, "final_metrics.csv")
        if os.path.isfile(f):
            rows.append(pd.read_csv(f))
    return pd.concat(rows, ignore_index=True)


def bootstrap_acc(y_true, y_pred, B=B, seed=42):
    rng = np.random.default_rng(seed)
    correct = (y_true == y_pred).astype(np.int8)
    n = len(correct)
    samples = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        samples[i] = correct[idx].mean()
    return correct.mean(), np.percentile(samples, 2.5), np.percentile(samples, 97.5)


def bootstrap_macro_f1(y_true, y_pred, classes, B=B, seed=43):
    """Inlined macro-F1 to avoid sklearn overhead in the hot loop."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    n = len(y_true)

    def macro_f1(yt, yp):
        scores = []
        for c in classes:
            tp = np.sum((yt == c) & (yp == c))
            fp = np.sum((yt != c) & (yp == c))
            fn = np.sum((yt == c) & (yp != c))
            if (tp + fp) == 0 or (tp + fn) == 0:
                scores.append(0.0)
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            scores.append(0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec))
        return float(np.mean(scores))

    point = macro_f1(y_true, y_pred)
    samples = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n)
        samples[i] = macro_f1(y_true[idx], y_pred[idx])
    return point, float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def per_fold_loso_table():
    rows = []
    for m in MODELS:
        for dom in DOMAINS:
            sub = f"S2_LOSO_{m}_{dom}"
            pred_path = os.path.join(RESULTS_DIR, sub, "test_predictions.csv")
            metr_path = os.path.join(RESULTS_DIR, sub, "final_metrics.csv")
            if not (os.path.exists(pred_path) and os.path.exists(metr_path)):
                print(f"  missing: {sub}")
                continue

            metrics = pd.read_csv(metr_path).iloc[0]
            df = pd.read_csv(pred_path)
            y_true = df["True_Label"].astype(int).values
            y_pred = df["Predicted_Label"].astype(int).values
            # Per-experiment class set (used by F1 macro-average so that a
            # genus never seen in this fold's test set does not contribute a
            # zero to the average).
            present_classes = sorted(set(y_true) | set(y_pred))

            acc, acc_lo, acc_hi = bootstrap_acc(y_true, y_pred)
            f1, f1_lo, f1_hi = bootstrap_macro_f1(y_true, y_pred, present_classes)

            rows.append({
                "Model": m, "Domain": dom, "N_test": len(y_true),
                # Fixed 27-class LOSO core set per protocol (Scenario C);
                # individual folds may exhibit fewer present classes.
                "N_classes": 27,
                "Acc": acc, "Acc_CI_lo": acc_lo, "Acc_CI_hi": acc_hi,
                "F1": f1, "F1_CI_lo": f1_lo, "F1_CI_hi": f1_hi,
                "Reported_Acc": metrics["Acc"], "Reported_F1": metrics["F1"],
                "Time_Sec": metrics["Time_Sec"],
            })
            print(f"  {m:20s} {dom:20s} acc={acc:.4f} [{acc_lo:.4f}, {acc_hi:.4f}]")
    return pd.DataFrame(rows)


def loso_mean_std(per_fold):
    agg = per_fold.groupby("Model").agg(
        Acc_mean=("Acc", "mean"), Acc_std=("Acc", "std"),
        F1_mean=("F1", "mean"), F1_std=("F1", "std"),
        Acc_min=("Acc", "min"), Acc_max=("Acc", "max"),
        Time_sum=("Time_Sec", "sum"),
    ).reset_index()

    # Parametric 95% CI on the mean across the K=6 folds
    t_crit = stats.t.ppf(0.975, df=5)
    agg["Acc_CI_lo"] = agg["Acc_mean"] - t_crit * agg["Acc_std"] / np.sqrt(6)
    agg["Acc_CI_hi"] = agg["Acc_mean"] + t_crit * agg["Acc_std"] / np.sqrt(6)
    agg["F1_CI_lo"] = agg["F1_mean"] - t_crit * agg["F1_std"] / np.sqrt(6)
    agg["F1_CI_hi"] = agg["F1_mean"] + t_crit * agg["F1_std"] / np.sqrt(6)
    return agg.sort_values("Acc_mean", ascending=False).reset_index(drop=True)


def friedman_nemenyi(per_fold):
    pivot = per_fold.pivot(index="Domain", columns="Model", values="Acc")[MODELS]
    chi2, p = stats.friedmanchisquare(*[pivot[m].values for m in MODELS])

    ranks = pivot.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean(axis=0)

    k, n = len(MODELS), pivot.shape[0]
    CD = Q_ALPHA * np.sqrt(k * (k + 1) / (6 * n))

    rows = []
    for a, b in combinations(MODELS, 2):
        diff = abs(mean_ranks[a] - mean_ranks[b])
        rows.append({
            "Model_A": a, "Model_B": b,
            "Rank_A": mean_ranks[a], "Rank_B": mean_ranks[b],
            "Rank_diff": diff, "CD_0.05": CD,
            "Significant_at_0.05": diff > CD,
        })
    return chi2, p, mean_ranks, CD, pivot, pd.DataFrame(rows)


def wilcoxon_pairs(per_fold):
    pivot = per_fold.pivot(index="Domain", columns="Model", values="Acc")[MODELS]
    rows = []
    for a, b in combinations(MODELS, 2):
        va, vb = pivot[a].values, pivot[b].values
        try:
            w, p = stats.wilcoxon(va, vb, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            w, p = float('nan'), float('nan')
        rows.append({
            "Model_A": a, "Model_B": b,
            "Mean_diff_acc": float(np.mean(va - vb)),
            "Median_diff_acc": float(np.median(va - vb)),
            "Wilcoxon_W": w, "p_value_two_sided": p,
        })
    df = pd.DataFrame(rows)

    # Holm step-down on the 10 pairwise comparisons
    sorted_df = df.sort_values("p_value_two_sided").reset_index(drop=True)
    m = len(sorted_df)
    adj = np.empty(m)
    running_max = 0.0
    for i, p in enumerate(sorted_df["p_value_two_sided"].values):
        running_max = max(running_max, (m - i) * p)
        adj[i] = min(running_max, 1.0)
    sorted_df["p_holm"] = adj
    return df.merge(sorted_df[["Model_A", "Model_B", "p_holm"]],
                    on=["Model_A", "Model_B"], how="left")


def unified_pairwise_table(nemenyi, wilcoxon_df, mean_ranks, CD):
    rows = []
    for _, r in nemenyi.iterrows():
        a, b = r["Model_A"], r["Model_B"]
        w_row = wilcoxon_df[(wilcoxon_df["Model_A"] == a) &
                            (wilcoxon_df["Model_B"] == b)].iloc[0]
        delta_acc = w_row["Mean_diff_acc"]
        if delta_acc < 0:
            a, b = b, a
            delta_acc = -delta_acc
        delta_rank = abs(mean_ranks[a] - mean_ranks[b])
        nemenyi_sig = delta_rank > CD
        p_unc = w_row["p_value_two_sided"]

        if nemenyi_sig:
            verdict = "Significant (Nemenyi)"
        elif p_unc is not None and p_unc < 0.05:
            verdict = "Directional only (uncorrected Wilcoxon)"
        elif delta_rank < 0.5 and p_unc > 0.5:
            verdict = "Co-leaders / indistinguishable"
        else:
            verdict = "Not significant"

        rows.append({
            "A": a, "B": b,
            "delta_rank": delta_rank, "Nemenyi": nemenyi_sig,
            "delta_acc_pct": delta_acc * 100,
            "W": w_row["Wilcoxon_W"],
            "p_unc": p_unc, "p_holm": w_row["p_holm"],
            "Verdict": verdict,
        })
    return (pd.DataFrame(rows)
            .sort_values("delta_rank", ascending=False)
            .reset_index(drop=True))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading master metrics ...")
    master = load_master()
    master.to_csv(os.path.join(OUT_DIR, "master_results.csv"), index=False)
    print(f"  {len(master)} experiments")

    print("\nPer-fold LOSO bootstrap CIs ...")
    per_fold = per_fold_loso_table()
    per_fold.to_csv(os.path.join(OUT_DIR, "per_fold_loso.csv"), index=False)

    print("\nLOSO mean +- SD ...")
    agg = loso_mean_std(per_fold)
    agg.to_csv(os.path.join(OUT_DIR, "loso_mean_std.csv"), index=False)
    print(agg[["Model", "Acc_mean", "Acc_std", "Acc_CI_lo", "Acc_CI_hi"]].to_string(index=False))

    print("\nFriedman + Nemenyi ...")
    chi2, p, mean_ranks, CD, pivot, nemenyi = friedman_nemenyi(per_fold)
    print(f"  chi2 = {chi2:.3f}  p = {p:.5f}  CD = {CD:.3f}")
    print("  mean ranks:")
    print(mean_ranks.to_string())
    nemenyi.to_csv(os.path.join(OUT_DIR, "nemenyi_pairwise.csv"), index=False)
    with open(os.path.join(OUT_DIR, "friedman.txt"), "w") as f:
        f.write(f"Friedman chi2 = {chi2:.4f}\n")
        f.write(f"p-value       = {p:.5f}\n")
        f.write(f"k = {len(MODELS)}, n = {pivot.shape[0]}\n")
        f.write("Mean ranks:\n")
        f.write(mean_ranks.to_string() + "\n")
        f.write(f"\nNemenyi CD (alpha=0.05): {CD:.4f}\n")

    print("\nPaired Wilcoxon (Holm-corrected) ...")
    wilcoxon_df = wilcoxon_pairs(per_fold)
    wilcoxon_df.to_csv(os.path.join(OUT_DIR, "wilcoxon_pairwise.csv"), index=False)
    print(wilcoxon_df.to_string(index=False))

    print("\nUnified pairwise table ...")
    unified = unified_pairwise_table(nemenyi, wilcoxon_df, mean_ranks, CD)
    unified.to_csv(os.path.join(OUT_DIR, "table7_unified_pairwise.csv"), index=False)
    print(unified.to_string(index=False))


if __name__ == "__main__":
    main()
