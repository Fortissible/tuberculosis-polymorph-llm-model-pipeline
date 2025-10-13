#!/usr/bin/env python3
import json
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import load

from sklearn.metrics import (
    roc_auc_score, average_precision_score, roc_curve, precision_recall_curve,
    confusion_matrix
)
import matplotlib.pyplot as plt

# ========= CONFIGURE HERE (no CLI) =========
FEATURES_CSV = Path("E:/Project/tuberculosis-polymorph-llm-model-pipeline/xgboost/predict/features.csv")   # samples x features (index=sample_id)
META_CSV     = Path("E:/Project/tuberculosis-polymorph-llm-model-pipeline/xgboost/predict/meta.csv")             # has lineage + label_INH/RIF/EMB/PZA
MODELS_DIR   = Path("./models_xgb")                  # where training saved models
DRUGS        = ["INH", "RIF", "EMB", "PZA"]          # which drugs to evaluate
OUT_CSV      = Path("./predictions_per_drug.csv")    # main predictions table
# ===========================================

def _safe_roc_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    return float(roc_auc_score(y_true, y_score))

def _safe_auprc(y_true, y_score):
    y_true = np.asarray(y_true)
    if np.unique(y_true).size < 2:
        return None
    return float(average_precision_score(y_true, y_score))

def load_expected_features(drug_dir: Path):
    final_model = load(drug_dir / "final_base_model.joblib")
    try:
        feats = final_model.get_booster().feature_names
        if feats is None:
            raise AttributeError
        return list(feats)
    except Exception:
        if hasattr(final_model, "feature_names_in_"):
            return list(final_model.feature_names_in_)
        raise RuntimeError(f"Cannot determine expected features for {drug_dir.name}")

def load_fold_models(drug_dir: Path):
    folds = []
    for fold_dir in sorted(drug_dir.glob("fold*")):
        mpath = fold_dir / "calibrated_model.joblib"
        if mpath.exists():
            folds.append(load(mpath))
    if not folds:
        raise FileNotFoundError(f"No calibrated fold models found in {drug_dir}")
    return folds

def load_threshold(drug_dir: Path, default_thr=0.5):
    summ = drug_dir / "summary.json"
    if summ.exists():
        with open(summ, "r") as f:
            js = json.load(f)
        return float(js.get("chosen_threshold", default_thr))
    return float(default_thr)

def align_features(X: pd.DataFrame, expected_cols):
    X = X.copy()
    for c in expected_cols:
        if c not in X.columns:
            X[c] = 0
    return X[expected_cols]

def predict_for_drug(drug: str, X: pd.DataFrame, models_dir: Path):
    drug_dir = models_dir / drug
    if not drug_dir.exists():
        raise FileNotFoundError(f"no trained models found in {drug_dir}")
    expected_cols = load_expected_features(drug_dir)
    Xd = align_features(X, expected_cols)
    fold_models = load_fold_models(drug_dir)

    probs = np.zeros(len(Xd), dtype=float)
    for m in fold_models:
        probs += m.predict_proba(Xd)[:, 1]
    probs /= len(fold_models)

    thr = load_threshold(drug_dir, default_thr=0.5)
    calls = (probs >= thr).astype(int)
    return probs, calls, thr

def plot_roc(y_true, y_prob, out_png, title="ROC Curve"):
    auc = _safe_roc_auc(y_true, y_prob)
    if auc is None:
        return None
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, label=f"AUROC = {auc:.3f}")
    plt.plot([0,1], [0,1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return auc

def plot_pr(y_true, y_prob, out_png, title="Precision-Recall Curve"):
    auprc = _safe_auprc(y_true, y_prob)
    if auprc is None:
        return None
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    plt.figure()
    plt.plot(recall, precision, label=f"AUPRC = {auprc:.3f}")
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (PPV)")
    plt.title(title)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return auprc

def plot_confusion(y_true, y_pred, out_png, title="Confusion Matrix"):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(title)
    plt.colorbar()
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center", fontsize=12)
    plt.xticks([0,1], ["Pred S", "Pred R"])
    plt.yticks([0,1], ["True S", "True R"])
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return tn, fp, fn, tp

def main():
    eval_dir = OUT_CSV.with_name(OUT_CSV.stem + "_eval")
    eval_dir.mkdir(parents=True, exist_ok=True)

    if not FEATURES_CSV.exists():
        raise SystemExit(f"Missing features file: {FEATURES_CSV}")
    if not META_CSV.exists():
        raise SystemExit(f"Missing meta file: {META_CSV}")

    X = pd.read_csv(FEATURES_CSV, index_col=0)
    meta = pd.read_csv(META_CSV, index_col=0)

    common = X.index.intersection(meta.index)
    if len(common) == 0:
        raise SystemExit("No overlapping sample IDs between features and meta.")
    X = X.loc[common]
    meta = meta.loc[common]

    all_preds = pd.DataFrame(index=X.index)
    thresholds = {}
    per_drug_summaries = []

    for drug in DRUGS:
        print(f"[{drug}] predicting & evaluating...")
        try:
            probs, calls, thr = predict_for_drug(drug, X, MODELS_DIR)
        except (FileNotFoundError, RuntimeError) as exc:
            print(f"  Skipping {drug}: {exc}")
            continue

        all_preds[f"prob_{drug}"] = probs
        all_preds[f"call_{drug}"] = calls
        thresholds[drug] = thr

        ycol = f"label_{drug}"
        drug_dir = eval_dir / drug
        drug_dir.mkdir(parents=True, exist_ok=True)

        if ycol in meta.columns:
            y_true_full = meta[ycol]
            y_true = y_true_full.dropna().astype(int)
            idx = X.index.intersection(y_true.index)
            y_true = y_true.loc[idx]
            y_prob = all_preds.loc[idx, f"prob_{drug}"].values
            y_pred = (y_prob >= thr).astype(int)

            if len(y_true) == 0:
                print(f"  Warning: no ground-truth labels available for {drug}; skipping evaluation.")
                continue

            auroc = _safe_roc_auc(y_true.values, y_prob)
            auprc = _safe_auprc(y_true.values, y_prob)
            tn, fp, fn, tp = plot_confusion(y_true.values, y_pred, drug_dir / "confusion_matrix.png", title=f"{drug} Confusion")
            sens = tp / max(1, (tp + fn))
            spec = tn / max(1, (tn + fp))

            per_sample = pd.DataFrame({
                "sample_id": idx,
                "y_true": y_true.values,
                "prob": y_prob,
                "call": y_pred
            }).set_index("sample_id")
            per_sample.to_csv(drug_dir / "per_sample_predictions.csv")

            if auroc is not None:
                plot_roc(y_true.values, y_prob, drug_dir / "roc_curve.png", title=f"{drug} ROC")
            if auprc is not None:
                plot_pr(y_true.values, y_prob, drug_dir / "pr_curve.png", title=f"{drug} PR")

            summary = {
                "drug": drug,
                "n": int(len(y_true)),
                "n_pos": int((y_true == 1).sum()),
                "n_neg": int((y_true == 0).sum()),
                "threshold": float(thr),
                "auroc": auroc,
                "auprc": auprc,
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
                "sensitivity": float(sens),
                "specificity": float(spec),
            }
            with open(drug_dir / "summary.json", "w") as f:
                json.dump(summary, f, indent=2)
            per_drug_summaries.append(summary)

            fmt_auc = "N/A" if auroc is None else f"{auroc:.3f}"
            fmt_aupr = "N/A" if auprc is None else f"{auprc:.3f}"
            print(f"  AUROC={fmt_auc} AUPRC={fmt_aupr} thr={thr:.3f}  TP={tp} FP={fp} FN={fn} TN={tn}")
        else:
            print(f"  Warning: {ycol} not in meta; skipping evaluation for {drug}")


    # Combined predictions CSV and summaries
    all_preds.to_csv(OUT_CSV)
    if per_drug_summaries:
        pd.DataFrame(per_drug_summaries).to_csv(eval_dir / "summary_per_drug.csv", index=False)
    with open(eval_dir / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print(f"\nWrote predictions to: {OUT_CSV.resolve()}")
    print(f"Artifacts (plots, summaries) in: {eval_dir.resolve()}")

if __name__ == "__main__":
    main()
