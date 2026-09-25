"""WP1 supplementary results after a NULL ablation verdict:
  1. new_feature_importance.csv - mean |SHAP| for every feature under the V3
     (fullest) feature set, 3,000-row test sample (reuses the
     generate_shap_plot.py approach), so the new time/interaction features
     can be ranked against the originals.
  2. Bootstrap 95% CI on the TEST AUC difference (winner-policy check) - V1,
     V2, V3 vs V0 for XGBoost tuned. Resampling the test AUC metric (not a
     split) is compatible with the no-shuffling rule.
  3. Collinearity check for Logistic Regression: correlation of each
     interaction feature with its two source columns (train set).
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import block1prep

XGB_PARAMS = {'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 100,
              'subsample': 0.8, 'colsample_bytree': 0.6,
              'min_child_weight': 7, 'reg_alpha': 5.0, 'reg_lambda': 2.0}

# --- 3,000-row test sample with the same open_time split as Block 1 ---
df = pd.read_csv('final_merged_dataset_v5.csv')
df['open_time'] = pd.to_datetime(df['open_time'])
df = df.sort_values('open_time').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
split_time = df.iloc[split_idx]['open_time']
test_df = df[df['open_time'] >= split_time].reset_index(drop=True)

out = block1prep.build_datasets(add_time_features=True, add_interactions=True,
                                diagnostics=False, save=False)
X_tr, X_te, y_tr = out['X_train'], out['X_test'], out['y_train']
y_test = out['y_test']
for df_ in (X_tr, X_te):
    if 'deposit_value_usd' in df_.columns:
        df_.drop(columns=['deposit_value_usd'], inplace=True)

print("V3 feature count:", X_tr.shape[1])

# --- 1. SHAP importance (mean |SHAP|), 3,000-row test sample ---
import shap
model = XGBClassifier(**XGB_PARAMS, eval_metric='auc', random_state=0, n_jobs=-1)
model.fit(X_tr, y_tr)
sample = X_te.sample(min(3000, len(X_te)), random_state=0)
explainer = shap.TreeExplainer(model)
sv = explainer.shap_values(sample)
mean_abs = pd.Series(np.abs(sv).mean(axis=0), index=model.feature_names_in_).sort_values(ascending=False)
mean_abs.rename('mean_abs_shap').to_frame().to_csv('new_feature_importance.csv')
print("\nMean |SHAP| ranking (all features):")
print(mean_abs.round(5).head(15))
new_feats = [c for c in mean_abs.index if c.startswith('open_') or c.startswith('interaction_')]
print("\nNew features' mean |SHAP|:")
for c in new_feats:
    print(f"  {c}: {mean_abs[c]:.5f}")

# --- 2. Bootstrap 95% CI on test-AUC differences vs V0 ---
def auc_ci_delta(probs_a, probs_b, y, B=1000, seed=0):
    rng = np.random.RandomState(seed)
    diffs = []
    n = len(y)
    for _ in range(B):
        idx = rng.randint(0, n, n)
        diffs.append(roc_auc_score(y.iloc[idx], probs_b[idx]) -
                     roc_auc_score(y.iloc[idx], probs_a[idx]))
    diffs = np.array(diffs)
    return np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)

def build_variant(tflag, iflag):
    o = block1prep.build_datasets(add_time_features=tflag, add_interactions=iflag,
                                  diagnostics=False, save=False)
    Xt, Xe, yt, ye = o['X_train'], o['X_test'], o['y_train'], o['y_test']
    for df_ in (Xt, Xe):
        if 'deposit_value_usd' in df_.columns:
            df_.drop(columns=['deposit_value_usd'], inplace=True)
    return Xt, Xe, yt, ye

v0 = pd.read_csv('xgb_v0_test_predictions.csv')['prob'].values
probs = {'V0': v0}
for name, fl in [('V1', (True, False)), ('V2', (False, True)), ('V3', (True, True))]:
    Xt, Xe, yt, ye = build_variant(*fl)
    m = XGBClassifier(**XGB_PARAMS, eval_metric='auc', random_state=0, n_jobs=-1)
    m.fit(Xt, yt)
    probs[name] = m.predict_proba(Xe)[:, 1]

y_test_real = pd.read_csv('y_test.csv').iloc[:, 0]

print("\nBootstrap 95% CI on test-AUC difference vs V0 (B=1000, resampling the metric):")
for name in ['V1', 'V2', 'V3']:
    lo, hi = auc_ci_delta(probs['V0'], probs[name], y_test_real)
    delta = roc_auc_score(y_test_real, probs[name]) - roc_auc_score(y_test_real, probs['V0'])
    print(f"  {name} vs V0: delta={delta:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
          f"{'excludes 0' if lo > 0 or hi < 0 else 'includes 0 (within noise)'}")

# --- 3. Collinearity check (LR) for interaction features ---
vt = build_variant(False, True)[0]
print("\nCollinearity check (train, V2): interaction vs source columns")
base = ['range_width_normalized', 'token0_price_std_24h', 'fee_tier_pct',
        'pre_open_avg_daily_swap_count_3d']
print(vt[['interaction_range_width_x_price_std', 'interaction_fee_tier_x_swap_count'] + base].corr().round(3))