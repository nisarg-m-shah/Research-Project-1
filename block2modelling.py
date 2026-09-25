"""
Block 2 — Modeling & Full Evaluation Battery
Uniswap V3 LP Profitability Classifier

Requires Block 1 outputs: X_train.csv, X_test.csv, y_train.csv, y_test.csv

pip install xgboost shap --break-system-packages   (if not already installed)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, roc_curve,
    average_precision_score, brier_score_loss, f1_score
)
from eval_utils import full_report, ks_gini, decile_lift_table, best_threshold_macro_f1

pd.set_option('display.width', 120)

# ============================================================
# LOAD
# ============================================================
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]

# drop the redundant raw deposit_value_usd — keep log_deposit_value_usd only
for df_ in (X_train, X_test):
    if 'deposit_value_usd' in df_.columns:
        df_.drop(columns=['deposit_value_usd'], inplace=True)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# MODEL 1 — Baseline Logistic Regression
# ============================================================
lr = LogisticRegression(max_iter=2000, C=1.0)
lr.fit(X_train, y_train)
lr_probs_test = lr.predict_proba(X_test)[:, 1]
lr_probs_train = lr.predict_proba(X_train)[:, 1]

# ============================================================
# MODEL 2 — XGBoost
# ============================================================
try:
    from xgboost import XGBClassifier
except ImportError:
    raise SystemExit("Run: pip install xgboost --break-system-packages")

# carve the last 15% of TRAIN chronologically as an early-stopping validation
# set — keeps the no-shuffle discipline instead of a random carve
val_cut = int(len(X_train) * 0.85)
X_fit, X_val = X_train.iloc[:val_cut], X_train.iloc[val_cut:]
y_fit, y_val = y_train.iloc[:val_cut], y_train.iloc[val_cut:]

xgb = XGBClassifier(
    n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    eval_metric='auc', early_stopping_rounds=30,
    random_state=0, n_jobs=-1
)
xgb.fit(X_fit, y_fit, eval_set=[(X_val, y_val)], verbose=False)
print(f"XGBoost stopped at {xgb.best_iteration} trees")

xgb_probs_test = xgb.predict_proba(X_test)[:, 1]
xgb_probs_train = xgb.predict_proba(X_train)[:, 1]

# ============================================================
# THRESHOLD — choose via train-set F1 sweep, not a blind 0.5
# ============================================================
# Session-2 WP0 (brief Section 3b): the legacy positive-class F1 selector
# was reproduced first (it gives 0.45 on the original features). The project
# rule is TRAIN MACRO-F1 (handoff Section 4) - applied here. It must
# reproduce the reported 0.56 threshold.

xgb_threshold = best_threshold_macro_f1(y_train, xgb_probs_train)
print(f"\nSelected XGBoost threshold (train-set macro-F1-optimal): {xgb_threshold:.2f}")

# ============================================================
# EVALUATION BATTERY (run once per model)
# ============================================================
# ks_gini, decile_lift_table and full_report now live in eval_utils.py
# (identical behaviour; full_report's return dict is backward-compatibly
# extended with ks_pct, threshold, cm, accuracy, precision, recall, f1,
# macro_f1 and decile_table).

print("\n" + "#"*60)
print("# LOGISTIC REGRESSION (baseline)")
print("#"*60)
lr_test_metrics = full_report("LR — TEST", y_test, lr_probs_test, 0.5)
lr_train_metrics = full_report("LR — TRAIN", y_train, lr_probs_train, 0.5)

print("\n" + "#"*60)
print("# XGBOOST (challenger)")
print("#"*60)
xgb_test_metrics = full_report("XGBoost — TEST", y_test, xgb_probs_test, xgb_threshold)
xgb_train_metrics = full_report("XGBoost — TRAIN", y_train, xgb_probs_train, xgb_threshold)

# ============================================================
# OVERFITTING DIAGNOSTIC — train vs test gap, side by side
# ============================================================
print("\n" + "="*60)
print("TRAIN vs TEST GAP (XGBoost)")
print("="*60)
gap_tbl = pd.DataFrame({
    'train': xgb_train_metrics,
    'test': xgb_test_metrics,
})
gap_tbl['gap'] = gap_tbl['train'] - gap_tbl['test']
print(gap_tbl.round(4))
print("Rule of thumb: >0.05 AUC gap is worth investigating; this dataset also has a")
print("known train(58%)/test(67%) class-balance shift from the time split, which will")
print("inflate some of this gap independent of overfitting — note both in the report.")

# ============================================================
# LEARNING CURVE — chronological training-size slices
# ============================================================
print("\n" + "="*60)
print("LEARNING CURVE (chronological slices, XGBoost)")
print("="*60)
fractions = [0.2, 0.4, 0.6, 0.8, 1.0]
for frac in fractions:
    n = int(len(X_fit) * frac)
    m = XGBClassifier(n_estimators=xgb.best_iteration or 200, max_depth=4,
                       learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                       random_state=0, n_jobs=-1)
    m.fit(X_fit.iloc[:n], y_fit.iloc[:n])
    tr_auc = roc_auc_score(y_fit.iloc[:n], m.predict_proba(X_fit.iloc[:n])[:, 1])
    te_auc = roc_auc_score(y_test, m.predict_proba(X_test)[:, 1])
    print(f"  train_size={n:6d}  train_auc={tr_auc:.4f}  test_auc={te_auc:.4f}  gap={tr_auc-te_auc:.4f}")

# ============================================================
# SHAP — top features, XGBoost
# ============================================================
try:
    import shap
    explainer = shap.TreeExplainer(xgb)
    shap_values = explainer.shap_values(X_test.sample(min(3000, len(X_test)), random_state=0))
    print("\nSHAP computed on a 3000-row test sample. Use shap.summary_plot(shap_values, ...) "
          "in a notebook/Streamlit to render the plot for your report/dashboard.")
except ImportError:
    print("\nSHAP not installed — run: pip install shap --break-system-packages")

# ============================================================
# SAVE
# ============================================================
# Self-contained comparison rows: test battery + train AUC + train-test gap
# + 5-fold TimeSeriesSplit CV AUC + the confusion-matrix cells, so WP5 can
# aggregate without re-running any fit (every cell traced to a CSV on disk).
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

tscv = TimeSeriesSplit(n_splits=5)


def _cv(model, X, y):
    s = cross_val_score(model, X, y, cv=tscv, scoring='roc_auc', n_jobs=-1)
    return float(s.mean()), float(s.std())


# cross_val_score does not pass an eval_set, so the fitted early-stopping
# XGBoost is CV-cloned at its best_iteration trees without the callback
# (same data, same seed -> the CV AUC measures that exact model).
xgb_for_cv = XGBClassifier(
    n_estimators=xgb.get_booster().num_boosted_rounds(), max_depth=4,
    learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='auc',
    random_state=0, n_jobs=-1
)


def _row(name, te, m, train_auc, cv_mean, cv_std):
    r = {k: te[k] for k in scalar_keys}
    r.update({'train_auc': train_auc, 'gap': train_auc - te['auc'],
              'train_cv_auc_mean': cv_mean, 'train_cv_auc_std': cv_std,
              'cm_tn': int(te['cm'][0, 0]), 'cm_fp': int(te['cm'][0, 1]),
              'cm_fn': int(te['cm'][1, 0]), 'cm_tp': int(te['cm'][1, 1])})
    return r


scalar_keys = ['auc', 'pr_auc', 'ks', 'gini', 'brier', 'ks_pct', 'threshold',
               'accuracy', 'precision', 'recall', 'f1', 'macro_f1']

lr_cv, xgb_cv = _cv(lr, X_train, y_train), _cv(xgb_for_cv, X_train, y_train)
rows = [
    _row('LogReg', lr_test_metrics, lr,
         float(ks_gini(y_train, lr_probs_train)[2]), *lr_cv),
    _row('XGBoost', xgb_test_metrics, xgb,
         float(ks_gini(y_train, xgb_probs_train)[2]), *xgb_cv),
]
# This is the Block 2 checkpoint. WP5's make_comparison_table.py reads this
# file and writes the canonical model_comparison_test_metrics_v2.csv with the
# full battery; keeping them separate avoids the aggregator reading the very
# file it is about to overwrite.
pd.DataFrame(rows, index=['LogReg', 'XGBoost']).to_csv("block2_base_metrics.csv")
print("\nSaved block2_base_metrics.csv")
print("\nDone. XGBoost threshold chosen:", xgb_threshold)