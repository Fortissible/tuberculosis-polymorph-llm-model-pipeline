#!/usr/bin/env python3
# xgboost_train_per_drug.py (fold-safe metrics: Accuracy, AUROC, AUPRC)
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Robust import for XGBClassifier
try:
    from xgboost import XGBClassifier
except ImportError:
    from xgboost.sklearn import XGBClassifier  # fallback

from sklearn.model_selection import GroupKFold
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, accuracy_score
from joblib import dump

# ========= CONFIGURE HERE =========
FEATURES_CSV = Path("E:/Project/tuberculosis-polymorph-llm-model-pipeline/xgboost/training/features.csv")
META_CSV     = Path("E:/Project/tuberculosis-polymorph-llm-model-pipeline/xgboost/training/meta.csv")
OUT_DIR      = Path("./models_xgb")
DRUGS        = [
#    "INH",
#   "RIF",
    "EMB",
    "PZA"
]
N_SPLITS     = 5
CAL_METHOD   = "isotonic"   # "isotonic" or "sigmoid"
RANDOM_STATE = 42
# XGBoost base parameters (tune later if needed)
XGB_PARAMS = dict(
    n_estimators=600,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    learning_rate=0.05,
    n_jobs=-1,
    eval_metric="logloss",
    tree_method="hist",    # fast on modern CPUs
    random_state=RANDOM_STATE
)
# ==================================

def _safe_auroc(y_true, y_score):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))

def _safe_auprc(y_true, y_score):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))

def pick_threshold_min_vme(y_true, p):
    """Pick threshold that minimizes Very Major Errors (false S: FN)."""
    best_thr, best_vme = 0.5, np.inf
    for thr in np.linspace(0.05, 0.95, 181):
        yhat = (p >= thr).astype(int)
        # Always force 2x2 layout to avoid ravel errors
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0, 1]).ravel()
        vme = fn
        if vme < best_vme:
            best_thr, best_vme = thr, vme
    return float(best_thr)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_data():
    X = pd.read_csv(FEATURES_CSV, index_col=0)
    meta = pd.read_csv(META_CSV, index_col=0)
    if "lineage" not in meta.columns:
        raise ValueError("meta_labels.csv must have a 'lineage' column.")
    common = X.index.intersection(meta.index)
    if len(common) == 0:
        raise ValueError("No overlapping sample IDs between features and meta.")
    X = X.loc[common]
    meta = meta.loc[common]
    return X, meta

def train_one_drug(drug: str, X: pd.DataFrame, meta: pd.DataFrame):
    ycol = f"label_{drug}"
    if ycol not in meta.columns:
        print(f"[{drug}] missing {ycol} in meta — skipping.")
        return None

    y = meta[ycol]
    keep = ~y.isna()
    Xd  = X.loc[keep]
    yd  = y.loc[keep].astype(int).values
    gd  = meta.loc[keep, "lineage"].astype(str).values
    ids = Xd.index.to_numpy()

    if yd.sum() == 0 or yd.sum() == len(yd):
        print(f"[{drug}] skipped (single-class labels).")
        return None

    drug_dir = OUT_DIR / drug
    ensure_dir(drug_dir)

    gkf = GroupKFold(n_splits=N_SPLITS)
    oof_p = np.zeros_like(yd, dtype=float)
    oof_fold = np.full_like(yd, -1, dtype=int)
    fold_metrics = []

    print(f"[{drug}] Training with GroupKFold(n_splits={N_SPLITS}) over lineage groups...")
    for fold, (tr, te) in enumerate(gkf.split(Xd, yd, groups=gd), start=1):
        X_tr, y_tr = Xd.iloc[tr], yd[tr]
        X_te, y_te = Xd.iloc[te], yd[te]

        # Handle imbalance per-fold
        pos = max(1, (y_tr == 1).sum())
        neg = max(1, (y_tr == 0).sum())
        spw = max(1.0, neg / pos)

        base = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw)
        model = CalibratedClassifierCV(base, cv=3, method=CAL_METHOD)
        model.fit(X_tr, y_tr)

        p_te = model.predict_proba(X_te)[:, 1]
        oof_p[te] = p_te
        oof_fold[te] = fold

        # Fold metrics (safe for single-class y_te)
        acc  = float(accuracy_score(y_te, (p_te >= 0.5).astype(int)))
        auc  = _safe_auroc(y_te, p_te)
        aupr = _safe_auprc(y_te, p_te)

        # Save calibrated model for this fold
        fold_dir = drug_dir / f"fold{fold}"
        ensure_dir(fold_dir)
        dump(model, fold_dir / "calibrated_model.joblib")

        m = {
            "fold": fold,
            "n_train": int(len(tr)),
            "n_test": int(len(te)),
            "pos_train": int((y_tr==1).sum()),
            "pos_test": int((y_te==1).sum()),
            "scale_pos_weight": float(spw),
            "accuracy": acc,
            "auroc": auc,
            "auprc": aupr,
        }
        with open(fold_dir / "metrics.json", "w") as f:
            json.dump(m, f, indent=2)
        fold_metrics.append(m)

        print(f"  Fold {fold}: ACC={acc:.3f}  AUROC={np.nan if np.isnan(auc) else auc:.3f}  AUPRC={np.nan if np.isnan(aupr) else aupr:.3f}")

    # OOF metrics (safe)
    oof_acc  = float(accuracy_score(yd, (oof_p >= 0.5).astype(int)))
    oof_auc  = _safe_auroc(yd, oof_p)
    oof_aupr = _safe_auprc(yd, oof_p)

    # Pick threshold to minimize VME on OOF probs (works even if class-imbalanced)
    thr = pick_threshold_min_vme(yd, oof_p)
    yhat = (oof_p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(yd, yhat, labels=[0, 1]).ravel()
    sens = tp / max(1, (tp + fn))
    spec = tn / max(1, (tn + fp))

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "sample_id": ids,
        "fold": oof_fold,
        "y_true": yd,
        "p": oof_p
    })
    oof_df.to_csv(drug_dir / "oof_predictions.csv", index=False)

    # Save summary (only ACC, AUROC, AUPRC as requested)
    summary = {
        "drug": drug,
        "n_samples": int(len(yd)),
        "n_pos": int(yd.sum()),
        "n_neg": int((yd==0).sum()),
        "oof_accuracy@0.5": float(oof_acc),
        "oof_auroc": float(oof_auc) if not np.isnan(oof_auc) else None,
        "oof_auprc": float(oof_aupr) if not np.isnan(oof_aupr) else None,
        "chosen_threshold": float(thr),
        "confusion_at_threshold": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "sensitivity_at_threshold": float(sens),
        "specificity_at_threshold": float(spec),
        "fold_metrics": fold_metrics
    }
    with open(drug_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Train a final base (uncalibrated) model on ALL data for feature importances
    pos_all = max(1, (yd==1).sum())
    neg_all = max(1, (yd==0).sum())
    spw_all = max(1.0, neg_all / pos_all)
    final_base = XGBClassifier(**XGB_PARAMS, scale_pos_weight=spw_all)
    final_base.fit(Xd, yd)
    dump(final_base, drug_dir / "final_base_model.joblib")

    try:
        fi = pd.Series(final_base.feature_importances_, index=Xd.columns, name="gain")
        fi.sort_values(ascending=False).to_csv(drug_dir / "feature_importances.csv")
    except Exception as e:
        with open(drug_dir / "feature_importances_error.txt", "w") as f:
            f.write(str(e))

    print(f"[{drug}] OOF ACC={oof_acc:.3f}  OOF AUROC={np.nan if np.isnan(oof_auc) else oof_auc:.3f}  "
          f"OOF AUPRC={np.nan if np.isnan(oof_aupr) else oof_aupr:.3f}  thr={thr:.3f}")
    return summary

def main():
    ensure_dir(OUT_DIR)
    X, meta = load_data()

    missing_labels = [f"label_{d}" for d in DRUGS if f"label_{d}" not in meta.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns in meta: {missing_labels}")

    all_summaries = []
    for drug in DRUGS:
        s = train_one_drug(drug, X, meta)
        if s is not None:
            all_summaries.append(s)

    if all_summaries:
        pd.DataFrame(all_summaries).to_csv(OUT_DIR / "summary_per_drug.csv", index=False)
        print(f"\nWrote {OUT_DIR/'summary_per_drug.csv'}")
    print("\nDone.")

if __name__ == "__main__":
    main()
