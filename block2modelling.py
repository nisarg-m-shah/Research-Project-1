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
    average_precision_score, brier_score_loss
)

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
from sklearn.metrics import f1_score
def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.71, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t

xgb_threshold = best_threshold(y_train, xgb_probs_train)
print(f"\nSelected XGBoost threshold (train-set F1-optimal): {xgb_threshold:.2f}")

# ============================================================
# EVALUATION BATTERY (run once per model)
# ============================================================
def ks_gini(y_true, probs):
    df_ = pd.DataFrame({'y': y_true.values, 'p': probs})
    df_['decile'] = pd.qcut(df_['p'].rank(method='first'), 10, labels=False)
    grp = df_.groupby('decile')['y'].agg(['sum', 'count'])
    grp['cum_bad'] = (grp['sum'] / grp['sum'].sum()).cumsum()          # profitable = "good" here; kept name from course convention
    grp['cum_good'] = ((grp['count'] - grp['sum']) / (grp['count'] - grp['sum']).sum()).cumsum()
    ks = (grp['cum_bad'] - grp['cum_good']).abs().max()
    auc = roc_auc_score(y_true, probs)
    gini = 2 * auc - 1
    return ks, gini, auc

def decile_lift_table(y_true, probs):
    df_ = pd.DataFrame({'y': y_true.values, 'p': probs})
    df_['decile'] = pd.qcut(df_['p'].rank(method='first', ascending=False), 10, labels=range(1, 11))
    tbl = df_.groupby('decile').agg(n=('y', 'count'), positive=('y', 'sum'))
    tbl['positive_rate'] = tbl['positive'] / tbl['n']
    overall_rate = df_['y'].mean()
    tbl['lift'] = tbl['positive_rate'] / overall_rate
    tbl['cum_positive_pct'] = (tbl['positive'].cumsum() / tbl['positive'].sum() * 100)
    return tbl

def full_report(name, y_true, probs, threshold):
    preds = (probs >= threshold).astype(int)
    print(f"\n{'='*60}\n{name}  (threshold={threshold:.2f})\n{'='*60}")
    print(confusion_matrix(y_true, preds))
    print(classification_report(y_true, preds, digits=3))
    ks, gini, auc = ks_gini(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    print(f"ROC-AUC: {auc:.4f}  PR-AUC: {pr_auc:.4f}  KS: {ks*100:.1f}  Gini: {gini:.4f}  Brier: {brier:.4f}")
    if gini > 0.75 or auc > 0.9:
        print("!! Gini/AUC unusually high — re-check for leakage before trusting this.")
    print("\nDecile/Lift table:")
    print(decile_lift_table(y_true, probs).round(3))
    return {'auc': auc, 'pr_auc': pr_auc, 'ks': ks, 'gini': gini, 'brier': brier}

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
pd.DataFrame([lr_test_metrics, xgb_test_metrics], index=['LogReg', 'XGBoost']).to_csv("model_comparison_test_metrics.csv")
print("\nSaved model_comparison_test_metrics.csv")
print("\nDone. XGBoost threshold chosen:", xgb_threshold)