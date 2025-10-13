#!/usr/bin/env python3
# split_features_meta_80_20.py
import numpy as np
import pandas as pd
from pathlib import Path

# ====== CONFIG ======
FEATURES_CSV = Path("E:/Project/tuberculosis-polymorph-llm-model-pipeline/preprocess/dataset-test/mutation-tables/features_with_lineage.csv")
META_CSV     = Path("E:/Project/tuberculosis-polymorph-llm-model-pipeline/preprocess/dataset-test/mutation-tables/meta_labels.csv")
OUT_TRAIN    = Path("./training")
OUT_PRED     = Path("./predict")
TRAIN_RATIO  = 0.80
SEED         = 42
# ====================

def main():
    if not FEATURES_CSV.exists():
        raise SystemExit(f"Missing features file: {FEATURES_CSV}")
    if not META_CSV.exists():
        raise SystemExit(f"Missing meta file: {META_CSV}")

    # Load (index = sample_id)
    X = pd.read_csv(FEATURES_CSV, index_col=0)
    meta = pd.read_csv(META_CSV, index_col=0)

    # Align to overlapping sample_ids
    common = X.index.intersection(meta.index)
    if len(common) == 0:
        raise SystemExit("No overlapping sample IDs between features and meta.")
    X = X.loc[common]
    meta = meta.loc[common]

    label_cols = [c for c in meta.columns if c.startswith("label_")]
    if not label_cols:
        raise SystemExit("meta file must contain at least one 'label_' column (prefix 'label_').")

    # Retain only label columns; treat lineage as features (already in X)
    meta = meta[label_cols]

    rng = np.random.RandomState(SEED)
    n_total = len(common)
    n_train = int(n_total * TRAIN_RATIO)
    if n_train == 0 or n_train == n_total:
        raise SystemExit("Train ratio leaves no samples for train or predict split.")

    max_attempts = 500
    for attempt in range(1, max_attempts + 1):
        perm = rng.permutation(n_total)
        X_shuffled = X.iloc[perm]
        meta_shuffled = meta.iloc[perm]

        X_train = X_shuffled.iloc[:n_train].copy()
        X_pred = X_shuffled.iloc[n_train:].copy()
        meta_train = meta_shuffled.iloc[:n_train].copy()
        meta_pred = meta_shuffled.iloc[n_train:].copy()

        def _has_label_support(df):
            for col in label_cols:
                col_values = df[col].dropna().astype(int)
                if col_values.empty:
                    return False
                positives = int(col_values.sum())
                if positives == 0 or positives == len(col_values):
                    return False
            return True

        labels_ok = _has_label_support(meta_train) and _has_label_support(meta_pred)
        if labels_ok:
            break
        if attempt == max_attempts:
            raise SystemExit("Unable to generate a split with positive coverage for each label after multiple attempts.")
    else:
        raise SystemExit("Unexpected error preparing split.")

    # Make dirs
    OUT_TRAIN.mkdir(parents=True, exist_ok=True)
    OUT_PRED.mkdir(parents=True, exist_ok=True)

    # Save (simple names inside each folder)
    (OUT_TRAIN / "features.csv").write_text(X_train.to_csv())
    (OUT_TRAIN / "meta.csv").write_text(meta_train.to_csv())
    (OUT_PRED / "features.csv").write_text(X_pred.to_csv())
    (OUT_PRED / "meta.csv").write_text(meta_pred.to_csv())

    print(f"Total samples: {len(common)} (train={len(X_train)}, predict={len(X_pred)})")
    print(f"Written:\n  {OUT_TRAIN/'features.csv'}\n  {OUT_TRAIN/'meta.csv'}\n  {OUT_PRED/'features.csv'}\n  {OUT_PRED/'meta.csv'}")

if __name__ == "__main__":
    main()
