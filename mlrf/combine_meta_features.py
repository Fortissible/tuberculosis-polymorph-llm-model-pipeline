#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
import pandas as pd


LABEL_COLS = ["label_INH", "label_RIF", "label_EMB", "label_PZA"]
KEY = "sample_id"


def read_unique(path: Path, key: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if key not in df.columns:
        raise ValueError(f"{path} must contain column '{key}'")
    # Drop duplicate keys (keep first)
    dup_count = df.duplicated(subset=[key]).sum()
    if dup_count:
        print(f"[WARN] {dup_count} duplicate {key} rows removed in {path}")
        df = df.drop_duplicates(subset=[key], keep="first")
    return df


def combine(features_csv: Path, meta_csv: Path, out_csv: Path) -> None:
    # Read
    feats = read_unique(features_csv, KEY)
    meta = read_unique(meta_csv, KEY)

    # Validate required label columns
    missing = [c for c in LABEL_COLS if c not in meta.columns]
    if missing:
        raise ValueError(f"{meta_csv} is missing label columns: {missing}")

    # We’ll drop lineage from meta (since lineage is already OHE in features)
    if "lineage" in meta.columns:
        meta = meta.drop(columns=["lineage"])

    # Keep only key + labels from meta
    meta = meta[[KEY] + LABEL_COLS]

    # Intersect on sample_id
    f_ids = set(feats[KEY])
    m_ids = set(meta[KEY])
    inter = f_ids & m_ids
    if not inter:
        raise ValueError("No overlapping sample_id between the two CSVs.")

    if (len(f_ids - inter) or len(m_ids - inter)):
        print(f"[WARN] Filtering to intersection of {len(inter)} samples "
              f"(dropped {len(f_ids - inter)} from features, {len(m_ids - inter)} from meta)")

    feats = feats[feats[KEY].isin(inter)].set_index(KEY).sort_index()
    meta = meta[meta[KEY].isin(inter)].set_index(KEY).sort_index()

    # Merge: left = features (keeps all feature columns), right = labels only
    out = feats.join(meta, how="inner")

    # Put sample_id as first column
    out = out.reset_index()

    # Basic sanity: ensure no name collisions aside from sample_id
    collisions = [c for c in LABEL_COLS if c in feats.columns]
    if collisions:
        print(f"[WARN] Feature columns collide with label names: {collisions}. "
              f"Consider renaming to avoid confusion.")

    # Save
    out.to_csv(out_csv, index=False)
    print(f"[OK] Wrote combined dataset -> {out_csv}")
    print(f"[INFO] Rows: {out.shape[0]} | Features: {feats.shape[1]} | Labels: {len(LABEL_COLS)}")


def main():
    ap = argparse.ArgumentParser(
        description="Combine features_with_lineage.csv and meta_labels.csv by sample_id, "
                    "dropping raw lineage from meta and keeping label columns."
    )
    ap.add_argument("--features_csv", required=True, type=Path,
                    help="CSV: sample_id + SNP features (+ lineage OHE already)")
    ap.add_argument("--meta_csv", required=True, type=Path,
                    help="CSV: sample_id, lineage, label_INH,label_RIF,label_EMB,label_PZA")
    ap.add_argument("--out_csv", required=True, type=Path,
                    help="Output CSV path")
    args = ap.parse_args()
    combine(args.features_csv, args.meta_csv, args.out_csv)


if __name__ == "__main__":
    main()
