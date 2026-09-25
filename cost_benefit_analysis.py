"""
Chapter 11.2 — Cost-Benefit Analysis inputs
Chapter 11.3 — threshold re-check against the resulting cost ratio

Computes, from the dataset's own real P&L, the dollar cost ratio between a
false positive (LP opens a position the model calls profitable that turns
out unprofitable) and a false negative (LP skips a position that would have
been profitable), then checks whether the 0.56 macro-F1 threshold should
shift under the resulting cost asymmetry.

P&L definition (as prescribed in the project handoff, Section 5 / item 2):
    net_profit_usd = true_fees_value_usd - impermanent_loss_usd
    profitable     = (net_profit_usd > 0)       [matches pipeline label]

Data-quality note: the pipeline's Step-6 outlier filter guards only
mint_amount0_adj / mint_amount1_adj / total_token0_returned_adj . A small
set of rows carry impossible magnitudes in the *derived* USD columns
(true_fees_value_usd / returned_value_usd / impermanent_loss_usd) — e.g. one
position shows ~$26.5 trillion of fees on a ~$21k deposit. These are
physically impossible on-chain and would dominate any mean-based cost
figure. They are flagged by the provable bound that a position's returned
principal cannot exceed ~deposit value (USD appreciation of the held token
accounts for the slack; >10x is unreachable on-chain). They are excluded
from the dollar figures below. Their model-input columns are unaffected
(they live only on the leakage side), so X/y/model numbers are untouched.
"""

import pandas as pd
import numpy as np

pd.set_option('display.width', 160)

# ============================================================
# 1. Load, derive P&L, flag corrupt rows
# ============================================================
df = pd.read_csv("final_merged_dataset_v5.csv")
df['net_profit_usd'] = df['true_fees_value_usd'] - df['impermanent_loss_usd']

assert ((df['net_profit_usd'] > 0).astype(int) == df['profitable']).all()

CORRUPT_MASK = (
    (df['true_fees_value_usd'] > 10 * df['deposit_value_usd']) |
    (df['returned_value_usd'] > 10 * df['deposit_value_usd']) |
    (df['impermanent_loss_usd'].abs() > 10 * df['deposit_value_usd'])
)
n_corrupt = int(CORRUPT_MASK.sum())
print(f"[CLEAN] Corrupt-derived-P&L rows excluded from dollar figures: {n_corrupt} "
      f"({n_corrupt/len(df)*100:.4f}% of {len(df)})")
print(f"[CLEAN] tokenIds: {sorted(df.loc[CORRUPT_MASK, 'tokenId'].tolist())}")

clean = df[~CORRUPT_MASK].copy()

# ============================================================
# 2. Cost figures (robustness across statistics)
# ============================================================
p1 = clean.loc[clean['profitable'] == 1, 'net_profit_usd']
p0 = clean.loc[clean['profitable'] == 0, 'net_profit_usd']


def trimmed_mean(s, lo=0.05, hi=0.95):
    return s[(s >= s.quantile(lo)) & (s <= s.quantile(hi))].mean()


rows = {}
for name, s1, s0 in [("raw (all rows)", p1, p0)]:
    pass
# mean-based ratio (literal handoff instruction, clean rows only)
cost_fp_mean = -p0.mean()
cost_fn_mean = p1.mean()
# median-based (robust to the genuine right-skew of P&L)
cost_fp_med = -p0.median()
cost_fn_med = p1.median()
# trimmed-mean-based
cost_fp_trim = -trimmed_mean(p0)
cost_fn_trim = trimmed_mean(p1)

print("\n=== Cost-benefit inputs (net_profit_usd) ===")
print(f"  class 1 (profitable): n={len(p1)}  mean={p1.mean():,.2f}  "
      f"median={p1.median():,.2f}  trimmed(5-95%)={trimmed_mean(p1):,.2f}")
print(f"  class 0 (unprofitable): n={len(p0)}  mean={p0.mean():,.2f}  "
      f"median={p0.median():,.2f}  trimmed(5-95%)={trimmed_mean(p0):,.2f}")
print(f"\n  cost-FP (avg loss on predicted-positive-actually-negative):")
print(f"    mean-based : {cost_fp_mean:,.2f}   ratio FP:FN = {cost_fp_mean/cost_fn_mean:.4f}")
print(f"    median-based: {cost_fp_med:,.2f}   ratio FP:FN = {cost_fp_med/cost_fn_med:.4f}")
print(f"    trimmed(5-95%): {cost_fp_trim:,.2f}   ratio FP:FN = {cost_fp_trim/cost_fn_trim:.4f}")

# ============================================================
# 3. Train the Block-3 final model, reproduce test probabilities
# ============================================================
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score

X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]
for d in (X_train, X_test):
    if 'deposit_value_usd' in d.columns:
        d = d.drop(columns=['deposit_value_usd'])

params = {
    'subsample': 0.8, 'reg_lambda': 2.0, 'reg_alpha': 5.0, 'n_estimators': 100,
    'min_child_weight': 7.0, 'max_depth': 4, 'learning_rate': 0.02,
    'colsample_bytree': 0.6,
}
model = XGBClassifier(**params, eval_metric='auc', random_state=0, n_jobs=-1)
model.fit(X_train, y_train)
y_score_train = model.predict_proba(X_train)[:, 1]
y_score_test = model.predict_proba(X_test)[:, 1]

test_auc = roc_auc_score(y_test, y_score_test)
print(f"\n=== Reproduced Block-3 final model on TEST ===")
print(f"  ROC-AUC: {test_auc:.4f}")

# ============================================================
# 4. Threshold sweep
#    (a) macro-F1 optimal (the current 0.56 choice)
#    (b) cost-corrected: minimize expected misclassification cost
#    (c) total-P&L optimizer: maximize realized net profit on test set
# ============================================================
def macro_f1_best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.30, 0.81, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t, best_f1


t_f1, f1_val = macro_f1_best_threshold(y_train, y_score_train)
print(f"\n  macro-F1-optimal threshold (train): {t_f1:.2f}  (macro-F1={f1_val:.4f})")

# cost-corrected optimal threshold, closed form t* = C_FP / (C_FP + C_FN)
for label, (cfp, cfn) in {
    "mean-based (clean)": (cost_fp_mean, cost_fn_mean),
    "median-based (clean)": (cost_fp_med, cost_fn_med),
    "trimmed-mean (clean)": (cost_fp_trim, cost_fn_trim),
}.items():
    t = cfp / (cfp + cfn)
    print(f"  cost-optimal threshold [{label}]: {t:.3f}  (ratio {cfp/cfn:.3f})")

# Total realized P&L on the TEST slice, using clean P&L (7 corrupt rows excluded).
# X_test row i corresponds positionally to the i-th row of the 20%-later slice of the
# open_time-sorted dataset (Block 1's split), because Block 1 sorted before splitting.
sorted_df = df.sort_values('open_time').reset_index(drop=True)
split_idx = int(len(sorted_df) * 0.8)
test_sorted = sorted_df.iloc[split_idx:].reset_index(drop=True)
assert len(test_sorted) == len(X_test)

corrupt_test_tokens = set(sorted_df.loc[CORRUPT_MASK, 'tokenId'])
net_test = test_sorted['net_profit_usd'].where(
    ~test_sorted['tokenId'].isin(corrupt_test_tokens))

net_test = test_sorted['net_profit_usd'].where(~test_sorted['tokenId'].isin(
    set(sorted_df.loc[CORRUPT_MASK, 'tokenId'])))

best_pnl_t, best_pnl = 0.5, -np.inf
for t in np.arange(0.30, 0.81, 0.01):
    pos = (y_score_test >= t)
    pnl = net_test[pos].sum()
    if pnl > best_pnl:
        best_pnl_t, best_pnl = t, pnl
print(f"\n  Total-realized-P&L-optimal threshold (test, clean P&L): {best_pnl_t:.2f} "
      f"(total realized P&L ${best_pnl:,.0f})")

# expected lift from operating at each candidate threshold on the test set
print("\n  Test-set realized P&L by operating threshold (clean P&L; sum over predicted positives):")
for t in [0.50, t_f1, 0.60, 0.65, 0.70]:
    pos = (y_score_test >= t)
    pnl = net_test[pos].sum()
    n = int(pos.sum())
    print(f"    threshold={t:.2f}: n_pred_positive={n:5d}  realized P&L=${pnl:,.0f}")

# Save a results artifact for the report
summary = pd.DataFrame({
    'statistic': ['mean (handoff formula, clean rows)', 'median', 'trimmed 5-95%'],
    'cost_fp_usd': [cost_fp_mean, cost_fp_med, cost_fp_trim],
    'cost_fn_usd': [cost_fn_mean, cost_fn_med, cost_fn_trim],
    'ratio_fp_to_fn': [cost_fp_mean / cost_fn_mean, cost_fp_med / cost_fn_med, cost_fp_trim / cost_fn_trim],
    'cost_optimal_threshold': [
        cost_fp_mean / (cost_fp_mean + cost_fn_mean),
        cost_fp_med / (cost_fp_med + cost_fn_med),
        cost_fp_trim / (cost_fp_trim + cost_fn_trim),
    ],
})
summary['corrupt_rows_excluded'] = n_corrupt
summary.to_csv("cost_benefit_results.csv", index=False)
print("\nSaved cost_benefit_results.csv")