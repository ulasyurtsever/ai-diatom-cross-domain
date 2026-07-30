
import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_ROOT = "./Processed_Data_PNG"
OUTPUT_CSV = "diatom_metadata.csv"
MIN_DOMAINS_FOR_DG = 5     # class must appear in >= this many domains for LOSO
SEED = 42


# Dominant microscope illumination technique per source database, recorded
# at the database level. BF = bright-field; DIC = differential interference
# contrast; Phase = phase-contrast. Laboratory identity, microscope hardware
# and illumination technique co-vary across the six contributing databases
# and cannot be deconfounded from the present data, so this attribute is
# database-level rather than per-image.
MODALITY = {
    "ADIAC_Database":    "BF",
    "AFD_Database":      "BF+DIC",
    "DIA_Database":      "BF+DIC",
    "DONA_Database":     "BF",
    "FCE_LTER_Database": "BF+DIC",
    "LOIR_Database":     "Phase+DIC",
}


def scan_dataset(root):
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(
            f"Image tree not found at {root!r}. "
            f"Run diatom_dataset_convert.py first, or pass --data-root.")

    rows = []
    for domain in (d for d in root_path.iterdir()
                   if d.is_dir() and not d.name.startswith('.')):
        for cls in (c for c in domain.iterdir()
                    if c.is_dir() and not c.name.startswith('.')):
            for img in cls.glob("*.png"):
                rows.append({'filepath': str(img),
                             'domain': domain.name,
                             'label': cls.name,
                             'modality': MODALITY.get(domain.name, 'UNKNOWN')})
    return pd.DataFrame(rows)


def assign_splits(df, seed=SEED):
    """Stratify by (domain, class) so that every class is represented in
    every split. Classes with fewer than 5 images are kept entirely in
    training (no point holding out one sample as a test set)."""
    df = df.copy()
    df['split'] = 'train'

    for domain in df['domain'].unique():
        for label in df['label'].unique():
            idx = df[(df['domain'] == domain) & (df['label'] == label)].index.tolist()
            if len(idx) < 5:
                continue

            train_idx, hold_idx = train_test_split(
                idx, test_size=0.2, random_state=seed)
            if len(hold_idx) >= 2:
                val_idx, test_idx = train_test_split(
                    hold_idx, test_size=0.5, random_state=seed)
                df.loc[val_idx, 'split'] = 'val'
                df.loc[test_idx, 'split'] = 'test'
            else:
                df.loc[hold_idx, 'split'] = 'val'

    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=DATA_ROOT,
                        help='Path to the curated image tree (default: %(default)s)')
    parser.add_argument('--output', default=OUTPUT_CSV,
                        help='Output CSV path (default: %(default)s)')
    parser.add_argument('--min-domains-dg', type=int, default=MIN_DOMAINS_FOR_DG,
                        help='Minimum number of source domains a class must appear in '
                             'to qualify for the LOSO scenario (default: %(default)s)')
    args = parser.parse_args()

    print(f"Scanning {args.data_root} ...")
    df = scan_dataset(args.data_root)
    print(f"  {len(df)} images, {df['label'].nunique()} classes, "
          f"{df['domain'].nunique()} domains")

    # DG eligibility: present in enough domains
    domain_counts = df.groupby('label')['domain'].nunique()
    dg_classes = domain_counts[domain_counts >= args.min_domains_dg].index.tolist()
    df['is_dg_class'] = df['label'].isin(dg_classes)
    print(f"  {len(dg_classes)} classes qualify for the LOSO scenario "
          f"(>= {args.min_domains_dg} domains)")

    # Dense label index
    classes = sorted(df['label'].unique())
    df['label_idx'] = df['label'].map({n: i for i, n in enumerate(classes)})

    # Splits
    df = assign_splits(df)
    counts = df['split'].value_counts().to_dict()
    print(f"  splits: train={counts.get('train', 0)}, "
          f"val={counts.get('val', 0)}, test={counts.get('test', 0)}")

    df.to_csv(args.output, index=False)
    print(f"Written: {args.output}")


if __name__ == "__main__":
    main()
