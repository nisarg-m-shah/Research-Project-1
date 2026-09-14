"""
Block 3 — Hyperparameter Tuning & Chronological Cross-Validation
Uniswap V3 LP Profitability Classifier

Requires Block 1 outputs: X_train.csv, X_test.csv, y_train.csv, y_test.csv
Fills in Chapter 7.4 (CV Design) and Chapter 9 (Hyperparameter Tuning) with
real, run results — no defaults-only shortcut this time.

pip install xgboost --break-system-packages
pip install optuna --break-system-packages   (optional, for the Optuna section)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss,
    classification_report, confusion_matrix, f1_score
)
from xgboost import XGBClassifier

pd.set_option('display.width', 120)

# ============================================================
# LOAD (same X/y as Block 2, deposit_value_usd already dropped there —
# redo the drop here too in case you're running this fresh)
# ============================================================
X_train = pd.read_csv("X_train.csv")
X_test = pd.read_csv("X_test.csv")
y_train = pd.read_csv("y_train.csv").iloc[:, 0]
y_test = pd.read_csv("y_test.csv").iloc[:, 0]
for df_ in (X_train, X_test):
    if 'deposit_value_usd' in df_.columns:
        df_.drop(columns=['deposit_value_usd'], inplace=True)

print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")

# ============================================================
# STEP 1: Chronological cross-validation design (Chapter 7.4)
# ============================================================
# TimeSeriesSplit respects the no-shuffle rule used throughout this project:
# fold k always trains on an earlier chronological block and validates on a
# later one, never the reverse. X_train is already sorted by open_time from
# Block 1, so positional folds ARE chronological folds here.
N_SPLITS = 5
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

print(f"\nCross-validation scheme: TimeSeriesSplit, {N_SPLITS} folds (chronological, no shuffling)")
for i, (tr_idx, val_idx) in enumerate(tscv.split(X_train)):
    print(f"  fold {i+1}: train rows 0-{tr_idx[-1]} ({len(tr_idx)} rows), "
          f"validate rows {val_idx[0]}-{val_idx[-1]} ({len(val_idx)} rows)")

# ============================================================
# STEP 2: RandomizedSearchCV — primary search method
# ============================================================
# Random search over Grid, given the parameter space here is continuous/wide
# enough that a grid would need far more iterations for the same coverage —
# this is the course's stated rationale for preferring Random over Grid when
# compute is finite but not razor-thin.
param_distributions = {
    'max_depth': [3, 4, 5, 6, 7],
    'learning_rate': [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15],
    'n_estimators': [100, 200, 300, 400, 500],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'min_child_weight': [1, 3, 5, 7, 10],
    'reg_alpha': [0, 0.01, 0.1, 1, 5],
    'reg_lambda': [0.5, 1, 2, 5, 10],
}

base_model = XGBClassifier(eval_metric='auc', random_state=0, n_jobs=-1)

search = RandomizedSearchCV(
    base_model,
    param_distributions=param_distributions,
    n_iter=60,
    scoring='roc_auc',
    cv=tscv,
    random_state=0,
    n_jobs=-1,
    verbose=1,
    return_train_score=True,
)

print("\nRunning RandomizedSearchCV (60 iterations x 5 folds = 300 fits — this will take a while)...")
search.fit(X_train, y_train)

print(f"\nBest CV ROC-AUC (mean across {N_SPLITS} folds): {search.best_score_:.4f}")
print(f"Best params: {search.best_params_}")

# Fold-level spread for the params that won — this is the "CV fold variance"
# check the course requires: large spread = don't trust a single split.
best_idx = search.best_index_
fold_score_cols = [c for c in search.cv_results_.keys() if c.startswith('split') and c.endswith('_test_score')]
fold_scores = [search.cv_results_[c][best_idx] for c in fold_score_cols]
print(f"\nPer-fold ROC-AUC for the best params: {[round(s,4) for s in fold_scores]}")
print(f"Fold mean: {np.mean(fold_scores):.4f}   Fold std: {np.std(fold_scores):.4f}")
if np.std(fold_scores) > 0.05:
    print("!! Fold std > 0.05 — this model's performance is unstable across time periods, "
          "not just weak on average. Say this plainly in Chapter 7.4/9.3.")

# ============================================================
# STEP 3: Optional — Optuna Bayesian search (extra rigor, now that time allows)
# ============================================================
try:
    import optuna
    from sklearn.model_selection import cross_val_score

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 600),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 12),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 20, log=True),
        }
        model = XGBClassifier(**params, eval_metric='auc', random_state=0, n_jobs=-1)
        scores = cross_val_score(model, X_train, y_train, cv=tscv, scoring='roc_auc', n_jobs=-1)
        return scores.mean()

    print("\nRunning Optuna (60 trials, TPE sampler, same TimeSeriesSplit folds)...")
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=0))
    study.optimize(objective, n_trials=60, show_progress_bar=False)

    print(f"\nOptuna best CV ROC-AUC: {study.best_value:.4f}")
    print(f"Optuna best params: {study.best_params}")
    print(f"(compare directly against RandomizedSearchCV's {search.best_score_:.4f} above — "
          f"report whichever generalizes better on TEST in step 4, not whichever has the higher CV score)")

    final_params = study.best_params if study.best_value > search.best_score_ else search.best_params_
    search_method_used = "Optuna (TPE)" if study.best_value > search.best_score_ else "RandomizedSearchCV"
except ImportError:
    print("\nOptuna not installed — skipping. Run: pip install optuna --break-system-packages "
          "if you want the Bayesian comparison (optional, RandomizedSearchCV result above is sufficient on its own).")
    final_params = search.best_params_
    search_method_used = "RandomizedSearchCV"

print(f"\nFinal hyperparameters selected via: {search_method_used}")
print(final_params)

# ============================================================
# STEP 4: Refit on full train with best params, evaluate on TEST
# ============================================================
final_model = XGBClassifier(**final_params, eval_metric='auc', random_state=0, n_jobs=-1)
final_model.fit(X_train, y_train)

train_probs = final_model.predict_proba(X_train)[:, 1]
test_probs = final_model.predict_proba(X_test)[:, 1]

def best_threshold(y_true, probs):
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.71, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t

threshold = best_threshold(y_train, train_probs)
print(f"\nThreshold (train macro-F1 optimal): {threshold:.2f}")

def ks_gini(y_true, probs):
    d = pd.DataFrame({'y': y_true.values, 'p': probs})
    d['decile'] = pd.qcut(d['p'].rank(method='first'), 10, labels=False)
    grp = d.groupby('decile')['y'].agg(['sum', 'count'])
    grp['cum_bad'] = (grp['sum'] / grp['sum'].sum()).cumsum()
    grp['cum_good'] = ((grp['count'] - grp['sum']) / (grp['count'] - grp['sum']).sum()).cumsum()
    ks = (grp['cum_bad'] - grp['cum_good']).abs().max()
    auc = roc_auc_score(y_true, probs)
    return ks, 2 * auc - 1, auc

for name, y_true, probs in [("TRAIN", y_train, train_probs), ("TEST", y_test, test_probs)]:
    preds = (probs >= threshold).astype(int)
    ks, gini, auc = ks_gini(y_true, probs)
    print(f"\n{name}: AUC={auc:.4f}  Gini={gini:.4f}  KS={ks*100:.1f}  "
          f"PR-AUC={average_precision_score(y_true, probs):.4f}  Brier={brier_score_loss(y_true, probs):.4f}")
    print(confusion_matrix(y_true, preds))
    print(classification_report(y_true, preds, digits=3))

print("\nCompare these TEST numbers against Block 2's untuned XGBoost (AUC 0.596, Gini 0.191).")
print("If the improvement is small, that itself is a legitimate, reportable finding: it means Block 2's")
print("diagnosis (feature-set/regime-shift ceiling, not a tuning gap) holds up under real tuning, not just")
print("under a defaults-only run — which is a STRONGER claim for Chapter 10.5/12.1 than the untuned version.")

pd.Series(final_params).to_csv("best_hyperparameters.csv")
print("\nSaved best_hyperparameters.csv")