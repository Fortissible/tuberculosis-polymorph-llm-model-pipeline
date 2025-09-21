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

    # Shuffle consistently
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(common))
    X = X.iloc[perm]
    meta = meta.iloc[perm]

    # Split 80/20
    n_train = int(len(common) * TRAIN_RATIO)
    X_train, X_pred = X.iloc[:n_train].copy(), X.iloc[n_train:].copy()
    meta_train, meta_pred = meta.iloc[:n_train].copy(), meta.iloc[n_train:].copy()

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
