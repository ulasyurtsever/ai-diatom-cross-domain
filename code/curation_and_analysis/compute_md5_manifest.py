"""
Compute per-image MD5 checksums + per-domain file counts so a third party
can verify their independent re-scrape of the six source databases matches
the curated Core Dataset reported in the manuscript.

Writes two artefacts next to ``data/diatom_metadata.csv``:

  data/md5_manifest.csv         filepath, md5, file_size_bytes
  data/per_domain_counts.csv    domain, n_files_metadata, n_files_on_disk

Usage:

  python compute_md5_manifest.py --data-root ../Processed_Data_PNG \\
                                 --metadata ../data/diatom_metadata.csv \\
                                 --out-dir  ../data/

The script is intended to be run locally on a workstation that holds the
re-built image tree.  The image binaries themselves are NOT redistributed
in this repository; the manifest published in the release is what a
reproducer compares their own re-computation against.
"""
import argparse
import csv
import hashlib
import os
from collections import Counter
from pathlib import Path

import pandas as pd


def _ddir():
    """Return the repository data/ directory (with trailing slash), located by
    walking up from this script or the CWD; falls back to ../data/."""
    from pathlib import Path as _P
    for b in [_P(__file__).resolve().parent, *_P(__file__).resolve().parents, _P.cwd(), *_P.cwd().parents]:
        if (b / "data" / "diatom_metadata.csv").exists():
            return str(b / "data") + "/"
    return "../data/"



def md5sum(path, chunk=65536):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", default="./Processed_Data_PNG",
                        help="Root of the rebuilt image tree (default: %(default)s)")
    parser.add_argument("--metadata", default=_ddir()+"diatom_metadata.csv",
                        help="Path to diatom_metadata.csv (default: %(default)s)")
    parser.add_argument("--out-dir", default=_ddir(),
                        help="Where to write manifests (default: %(default)s)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading metadata from {args.metadata}")
    df = pd.read_csv(args.metadata)
    print(f"  {len(df)} rows, {df['domain'].nunique()} domains, {df['label'].nunique()} classes")

    # ------------------------------------------------------------------
    # MD5 manifest
    # ------------------------------------------------------------------
    manifest_path = out_dir / "md5_manifest.csv"
    print(f"\nComputing MD5 sums to {manifest_path} ...")

    data_root = Path(args.data_root)
    missing = 0
    written = 0
    with open(manifest_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["filepath", "md5", "file_size_bytes"])
        for i, row in df.iterrows():
            rel = row["filepath"]
            # metadata stores e.g. 'Processed_Data_PNG/<Domain>/<Genus>/<id>.png'
            # so we strip the leading 'Processed_Data_PNG/' if data_root already
            # points at that directory.
            rel_stripped = rel
            if rel_stripped.startswith("Processed_Data_PNG/"):
                rel_stripped = rel_stripped[len("Processed_Data_PNG/"):]
            full = data_root / rel_stripped
            if not full.exists():
                missing += 1
                if missing <= 10:
                    print(f"  MISSING: {full}")
                continue
            md5 = md5sum(full)
            size = full.stat().st_size
            w.writerow([rel, md5, size])
            written += 1
            if written % 1000 == 0:
                print(f"  {written} files hashed ...")

    print(f"  wrote {written} entries, {missing} missing")

    # ------------------------------------------------------------------
    # Per-domain counts.
    #
    # The relevant question for a reproducer is *not* "do disk and metadata
    # contain the same number of files?" -- the on-disk tree may legitimately
    # also hold the pre-curation raw images that the manuscript's
    # `<100 image threshold` and synonym-merge filters dropped from the
    # released Core Dataset.  The question that matters is "are all
    # metadata-listed files actually present on disk?".  That is answered by
    # the manifest itself: any metadata row whose file was missing was
    # printed above and was *not* written to md5_manifest.csv.
    # ------------------------------------------------------------------
    counts_path = out_dir / "per_domain_counts.csv"
    print(f"\nWriting per-domain counts to {counts_path} ...")

    # n_files_in_manifest comes from re-reading the manifest just written;
    # this is "how many metadata-listed files were actually hashed", i.e.
    # found on disk.
    manifest_df = pd.read_csv(manifest_path)
    manifest_df["domain"] = manifest_df["filepath"].str.split("/").str[1]
    manifest_counts = manifest_df["domain"].value_counts().to_dict()

    meta_counts = df["domain"].value_counts().to_dict()

    # n_files_on_disk: raw PNG scan (includes pre-curation images that
    # diatom_metadata.py would have filtered out).  Recorded for transparency
    # but not used as the pass/fail signal.
    disk_counts = Counter()
    for d in sorted(data_root.iterdir()) if data_root.exists() else []:
        if d.is_dir() and not d.name.startswith("."):
            disk_counts[d.name] = sum(
                1 for _ in d.rglob("*.png") if not _.name.startswith(".")
            )

    with open(counts_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow([
            "domain",
            "n_files_metadata",
            "n_files_in_manifest",
            "n_metadata_files_missing_on_disk",
            "n_files_on_disk_total_raw",
            "status",
        ])
        for dom in sorted(set(meta_counts) | set(disk_counts) | set(manifest_counts)):
            m = meta_counts.get(dom, 0)
            f = manifest_counts.get(dom, 0)
            d = disk_counts.get(dom, 0)
            missing_from_disk = m - f
            status = "OK" if missing_from_disk == 0 else "MISSING"
            w.writerow([dom, m, f, missing_from_disk, d, status])
            tag = "" if status == "OK" else f"  <-- {missing_from_disk} metadata files NOT on disk"
            print(f"  {dom:20s}  meta={m:5d}  manifest={f:5d}  "
                  f"disk_raw={d:5d}  missing={missing_from_disk}{tag}")

    print("\nDone.")
    print(f"  Release {manifest_path.name} and {counts_path.name} alongside diatom_metadata.csv.")
    print("  A third party reproducing this dataset can run this script with --data-root")
    print("  pointing at their re-scrape; identical md5_manifest.csv = identical image bytes.")


if __name__ == "__main__":
    main()
