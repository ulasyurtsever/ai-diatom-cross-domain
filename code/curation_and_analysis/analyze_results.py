"""
Aggregate the per-experiment final_metrics.csv files under results/per_experiment/
into a single master spreadsheet and a few summary plots.
"""
import argparse
import glob
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")
plt.rcParams.update({'font.size': 12, 'figure.dpi': 300})


def load_master(results_dir):
    files = glob.glob(os.path.join(results_dir, "**/final_metrics.csv"),
                      recursive=True)
    if not files:
        return pd.DataFrame()

    frames = []
    for f in files:
        try:
            frames.append(pd.read_csv(f))
        except (IOError, pd.errors.ParserError) as e:
            print(f"  could not read {f}: {e}")
    return pd.concat(frames, ignore_index=True)


def plot_standard_bars(df, out_dir):
    sub = df[df['Mode'] == 'STANDARD'].sort_values('Acc', ascending=False)
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    melted = sub.melt(id_vars='Model', value_vars=['Acc', 'F1'],
                      var_name='Metric', value_name='Score')
    sns.barplot(data=melted, x='Model', y='Score', hue='Metric',
                palette='viridis', ax=ax)
    ax.set(title='Standard Mixed scenario', ylim=(0, 1.0),
           xlabel='Architecture', ylabel='Score')
    ax.legend(loc='lower right')
    plt.setp(ax.get_xticklabels(), rotation=15)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "S1_performance_bars.png"))
    plt.close(fig)


def plot_efficiency(df, out_dir):
    sub = df[df['Mode'] == 'STANDARD']
    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=sub, x='Time_Sec', y='Acc', hue='Model',
                    style='Model', s=200, palette='deep', ax=ax)
    for _, row in sub.iterrows():
        ax.text(row['Time_Sec'] + 30, row['Acc'], row['Model'], fontsize=9)
    ax.set(title='Accuracy vs. training time',
           xlabel='Training time (s) — lower is better',
           ylabel='Accuracy — higher is better')
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "S1_efficiency_scatter.png"))
    plt.close(fig)


def plot_loso_heatmap(df, out_dir):
    sub = df[df['Mode'] == 'LOSO']
    if sub.empty:
        return

    heatmap = sub.pivot_table(index='Model', columns='Domain', values='Acc')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(heatmap, annot=True, cmap='YlGnBu', fmt='.3f', linewidths=0.5, ax=ax)
    ax.set(title='LOSO accuracy by architecture and held-out domain',
           xlabel='Unseen test domain', ylabel='Architecture')
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "S2_LOSO_heatmap.png"))
    plt.close(fig)

    avg = sub.groupby('Model')['Acc'].mean().sort_values(ascending=False)
    print("\nLOSO ranking by mean accuracy:")
    print(avg.to_string())
    avg.to_csv(os.path.join(out_dir, "S2_LOSO_average_ranking.csv"),
               header=['mean_acc'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-dir', default='../results/per_experiment')
    parser.add_argument('--output-dir', default='Analysis_Report')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    df = load_master(args.results_dir)
    if df.empty:
        print(f"No results found under {args.results_dir!r}.")
        return

    print(f"Loaded {len(df)} experiments.")
    df.to_excel(os.path.join(args.output_dir, "master_results.xlsx"), index=False)

    plot_standard_bars(df, args.output_dir)
    plot_efficiency(df, args.output_dir)
    plot_loso_heatmap(df, args.output_dir)
    print(f"Plots written to {args.output_dir}/")


if __name__ == "__main__":
    main()
