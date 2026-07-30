#!/usr/bin/env python3
"""
check_dup_split_leakage.py
--------------------------
Split-level leakage audit of the within-source exact-duplicate images.

Question answered: do any of the within-source exact pHash-duplicate groups
(the "35 groups / 73 images, ~0.6 %" reported in Section 3.3) fall into
DIFFERENT image-level splits (train vs. val/test) of the Standard-Mixed /
Single-Source partition? If a duplicate's two copies land in train and test,
that is genuine leakage for Scenarios A/B (LOSO is unaffected because it holds
out an entire source).

It reproduces the manuscript's exact-duplicate grouping (identical 64-bit pHash
within the same source database) and then cross-references each group against
the pre-computed `split` column in data/diatom_metadata.csv.

Run it where the curated images live:

    pip install pillow imagehash pandas numpy
    cd ".../02_Revision_R1/code"
    python check_dup_split_leakage.py --data-root "/path/to/+ Processed_Data_PNG"

If you have already produced data/phash_manifest.csv (e.g. from
phash_dedup_check.py), you can skip hashing:

    python check_dup_split_leakage.py --manifest ../data/phash_manifest.csv

Outputs (written next to the metadata, both small and safe to share):
  * data/phash_manifest.csv           (per-image pHash, if it had to hash)
  * data/dup_split_leakage_report.csv (only the duplicate groups that span >1 split)
and prints a one-line verdict.
"""
import argparse
import collections
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def _find_repo_root(start):
    """Walk up from the script (and CWD) until data/diatom_metadata.csv is found,
    so the script works regardless of how deeply it is nested in the repository."""
    for base in [start, *start.parents, Path.cwd(), *Path.cwd().parents]:
        if (base / "data" / "diatom_metadata.csv").exists():
            return base
    return start.parent


REPO = _find_repo_root(HERE)
META = REPO / "data" / "diatom_metadata.csv"


def load_or_hash(manifest_path, data_root):
    mp = Path(manifest_path) if manifest_path else None
    if mp and mp.exists():
        print(f"[i] using existing manifest: {mp}")
        return pd.read_csv(mp)
    if not data_root:
        raise SystemExit(
            "No pHash manifest found. Pass --data-root pointing to the curated "
            "image folder (e.g. the '+ Processed_Data_PNG' directory) so the "
            "script can compute the hashes, or pass --manifest to an existing one."
        )
    from PIL import Image
    import imagehash
    md = pd.read_csv(META)
    root = Path(data_root)
    print(f"[i] hashing {len(md)} images from {root} ...")
    rows, bad = [], 0
    for _, r in md.iterrows():
        p = root / r["domain"] / r["label"] / Path(r["filepath"]).name
        try:
            h = imagehash.phash(Image.open(p).convert("L"))
            rows.append((r["domain"], r["label"], p.name, int(str(h), 16)))
        except Exception:
            bad += 1
    print(f"[i] hashed={len(rows)} unreadable={bad}")
    ph = pd.DataFrame(rows, columns=["domain", "label", "name", "phash"])
    out = REPO / "data" / "phash_manifest.csv"
    ph.to_csv(out, index=False)
    print(f"[i] manifest written to {out}")
    return ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None,
                    help="Curated image folder (contains <domain>/<label>/<file>).")
    ap.add_argument("--manifest", default=str(REPO / "data" / "phash_manifest.csv"),
                    help="Existing per-image pHash manifest to reuse if present.")
    args = ap.parse_args()

    ph = load_or_hash(args.manifest, args.data_root)
    md = pd.read_csv(META)
    md["name"] = md["filepath"].map(lambda x: Path(str(x)).name)

    merged = ph.merge(md[["domain", "label", "name", "split"]],
                      on=["domain", "label", "name"], how="left")
    n_missing = int(merged["split"].isna().sum())
    if n_missing:
        print(f"[!] {n_missing} hashed images could not be matched to a split row "
              f"(name mismatch); they are ignored in the audit.")

    # exact-duplicate groups: identical pHash WITHIN the same source database
    merged["key"] = merged["domain"].astype(str) + "|" + merged["phash"].astype(str)
    groups = collections.defaultdict(list)
    for i, k in enumerate(merged["key"].values):
        groups[k].append(i)

    total_groups = total_imgs = 0
    span_multi = 0
    leak_train_test = 0
    leak_rows = []
    for k, idx in groups.items():
        if len(idx) < 2:
            continue
        sub = merged.iloc[idx]
        total_groups += 1
        total_imgs += len(sub)
        splits = set(sub["split"].dropna())
        if len(splits) > 1:
            span_multi += 1
            if "train" in splits and ({"test", "val"} & splits):
                leak_train_test += 1
                leak_rows.append(sub[["domain", "label", "name", "split"]])

    print("\n=== Within-source exact-duplicate audit (Scenario A/B image-level splits) ===")
    print(f"within-source exact-duplicate groups : {total_groups}  ({total_imgs} images)")
    print(f"groups spanning more than one split  : {span_multi}")
    print(f"groups with train <-> val/test split : {leak_train_test}   <-- potential A/B leakage")

    if leak_rows:
        rep = pd.concat(leak_rows)
        out = REPO / "data" / "dup_split_leakage_report.csv"
        rep.to_csv(out, index=False)
        print(f"\n[!] {leak_train_test} duplicate group(s) span train and val/test. "
              f"Details written to {out}:")
        print(rep.to_string(index=False))
    else:
        print("\n[OK] No within-source duplicate group is split across train and "
              "val/test: there is no image-level duplicate leakage in Scenarios A/B.")


if __name__ == "__main__":
    main()
