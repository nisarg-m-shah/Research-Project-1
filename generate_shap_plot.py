"""
Appendix C — SHAP summary plot
Uniswap V3 LP Profitability Classifier

Renders the SHAP summary plot for the tuned (Block 3) XGBoost model on a
3000-row TEST sample, using the exact final hyperparameters from
best_hyperparameters.csv. Saves to shap_summary_plot.png for embedding in
the report alongside the EDA figures (fig1..fig8).

Run: pip install shap  (if not already installed)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
import shap

# ============================================================
# LOAD (same X/y as Blocks 2/3, deposit_value_usd dropped as redundant)
# ============================================================
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0]
for d in (X_train, X_test):
    if 'deposit_value_usd' in d.columns:
        d.drop(columns=['deposit_value_usd'], inplace=True)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# FINAL MODEL — the tuned Block 3 hyperparameters
# ============================================================
params = pd.read_csv("best_hyperparameters.csv", index_col=0).iloc[:, 0].to_dict()
params = {k: (int(v) if float(v) == int(v) else v) for k, v in params.items()}
print("Final hyperparameters:", params)

model = XGBClassifier(**params, eval_metric='auc', random_state=0, n_jobs=-1)
model.fit(X_train, y_train)

# ============================================================
# SHAP ON A 3000-ROW TEST SAMPLE (random_state=0, same size as Block 2)
# ============================================================
rng = np.random.RandomState(0)
sample = X_test.sample(min(3000, len(X_test)), random_state=0)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)
print(f"SHAP values: {shap_values.shape}")

plt.figure(figsize=(10, 9))
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig("shap_summary_plot.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved shap_summary_plot.png")

# also dump the mean-|shap| ranking so the report's Appendix C caption can
# cite the top drivers with real numbers
mean_abs_shap = np.abs(shap_values).mean(axis=0)
ranking = pd.Series(mean_abs_shap, index=sample.columns).sort_values(ascending=False)
ranking.to_csv("shap_feature_ranking.csv")
print("\nTop-10 features by mean |SHAP|:")
print(ranking.head(10).round(5).to_string())
print("\nSaved shap_feature_ranking.csv")