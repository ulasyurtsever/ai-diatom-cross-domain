
import argparse
from pathlib import Path

import pandas as pd


DATA_ROOT = "./Processed_Data_PNG"


def build_matrix(root_path):
    root = Path(root_path)
    if not root.exists():
        raise FileNotFoundError(
            f"Image tree not found at {root_path!r}. "
            f"Run diatom_dataset_convert.py first, or pass --data-root.")

    rows = []
    for domain in (d for d in root.iterdir()
                   if d.is_dir() and not d.name.startswith('.')):
        for cls in (c for c in domain.iterdir()
                    if c.is_dir() and not c.name.startswith('.')):
            n = sum(1 for f in cls.iterdir() if f.suffix.lower() == '.png')
            if n > 0:
                rows.append({'Domain': domain.name,
                             'Class': cls.name,
                             'Count': n})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=DATA_ROOT)
    parser.add_argument('--output', default='domain_overlap_matrix.csv')
    args = parser.parse_args()

    df = build_matrix(args.data_root)
    if df.empty:
        print(f"No images found under {args.data_root!r}")
        return

    pivot = df.pivot_table(index='Class', columns='Domain', values='Count',
                           fill_value=0).astype(int)
    pivot['Domain_Count'] = (pivot.drop(columns=[], errors='ignore') > 0).sum(axis=1)

    print(f"Unique classes : {len(pivot)}")
    print(f"Source databases: {len(pivot.columns) - 1}")

    # Distribution of class-by-domain coverage
    dist = pivot['Domain_Count'].value_counts().sort_index()
    print("\nClasses by number of domains present in:")
    for n_domains, n_classes in dist.items():
        print(f"  {n_domains} domain(s): {n_classes} classes")

    # Most-shared classes — useful sanity check
    common = pivot[pivot['Domain_Count'] >= 2].copy()
    image_cols = [c for c in common.columns if c != 'Domain_Count']
    common['Total_Images'] = common[image_cols].sum(axis=1)
    top = common.sort_values(['Domain_Count', 'Total_Images'],
                             ascending=[False, False]).head(20)
    print("\nTop 20 most-shared classes:")
    print(top[['Domain_Count', 'Total_Images']].to_string())

    pivot.to_csv(args.output)
    print(f"\nWritten: {args.output}")


if __name__ == "__main__":
    main()
