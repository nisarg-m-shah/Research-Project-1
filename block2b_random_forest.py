"""WP2 - Random Forest baseline and tuning on the winning variant's features.

Winner of WP1 is V0 (the original 27 features); WP1's ablation showed no
variant beats it on train-CV AUC, so RF is run on V0. RF bootstrap resampling
is intrinsic to the estimator and does not violate the no-shuffling rule
(rows are never shuffled; TimeSeriesSplit is used for CV).

Baseline: n_estimators=300, min_samples_leaf=5, max_features='sqrt',
          random_state=0, n_jobs=-1.
Tuned:    RandomizedSearchCV (TimeSeriesSplit 5-fold, scoring='roc_auc',
          n_iter=24, random_state=0) over a compact grid.

Reported per model (RF baseline, RF tuned): train 5-fold TSCV AUC (mean,std),
test full_report() with threshold re-derived from train macro-F1, train AUC
and train-test gap. Saved to rf_results.csv using the same schema as
ablation_results.csv so WP5 can aggregate cleanly.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV

from eval_utils import full_report, ks_gini, best_threshold_macro_f1

pd.set_option('display.width', 160)

tscv = TimeSeriesSplit(n_splits=5)

def load_v0():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    for df_ in (X_train, X_test):
        if 'deposit_value_usd' in df_.columns:
            df_.drop(columns=['deposit_value_usd'], inplace=True)
    return X_train, X_test, y_train, y_test

def evaluate(name, model, X_train, X_test, y_train, y_test, cv_scores=None):
    model.fit(X_train, y_train)
    tr_probs = model.predict_proba(X_train)[:, 1]
    te_probs = model.predict_proba(X_test)[:, 1]
    threshold = best_threshold_macro_f1(y_train, tr_probs)
    te = full_report(f"{name} - TEST", y_test, te_probs, threshold)
    tr_auc = ks_gini(y_train, tr_probs)[2]
    gap = tr_auc - te['auc']
    if cv_scores is None:
        cv_scores = cross_val_score(model, X_train, y_train, cv=tscv,
                                    scoring='roc_auc', n_jobs=-1)
    cv_mean, cv_std = float(cv_scores.mean()), float(cv_scores.std())
    print(f"[{name}] train_cv_auc={cv_mean:.4f} (std {cv_std:.4f}) "
          f"test_auc={te['auc']:.4f} train_auc={tr_auc:.4f} gap={gap:.4f} "
          f"threshold={threshold:.2f}")
    if name == 'RFtuned':
        pd.DataFrame({'y_true': y_test.values,
                      'prob': te_probs}).to_csv('rf_tuned_test_predictions.csv', index=False)
    return {
        'variant': 'V0', 'model': name, 'n_features': X_train.shape[1],
        'test_auc': te['auc'], 'test_gini': te['gini'],
        'test_ks_pct': te['ks_pct'], 'test_pr_auc': te['pr_auc'],
        'test_brier': te['brier'], 'threshold': threshold,
        'test_accuracy': te['accuracy'], 'test_macro_f1': te['macro_f1'],
        'test_precision': te['precision'], 'test_recall': te['recall'],
        'test_f1': te['f1'],
        'cm_tn': int(te['cm'][0, 0]), 'cm_fp': int(te['cm'][0, 1]),
        'cm_fn': int(te['cm'][1, 0]), 'cm_tp': int(te['cm'][1, 1]),
        'train_auc': tr_auc, 'gap': gap,
        'train_cv_auc_mean': cv_mean, 'train_cv_auc_std': cv_std,
    }

def main():
    X_train, X_test, y_train, y_test = load_v0()
    print(f"V0 features: {X_train.shape[1]} columns")

    print("\n--- RF baseline ---")
    rf_base = RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     max_features='sqrt', random_state=0, n_jobs=-1)
    res_baseline = evaluate('RFbaseline', rf_base, X_train, X_test,
                            y_train, y_test)

    print("\n--- RF tuned (RandomizedSearchCV, 24 iters, TSCV roc_auc) ---")
    grid = {
        'n_estimators': [150, 300, 450],
        'max_depth': [None, 10, 15],
        'min_samples_leaf': [2, 5, 10],
        'max_features': ['sqrt', 'log2', 0.3],
        'min_samples_split': [2, 5, 10],
    }
    rf_tuned = RandomForestClassifier(random_state=0, n_jobs=-1)
    search = RandomizedSearchCV(rf_tuned, grid, n_iter=24, scoring='roc_auc',
                                cv=tscv, random_state=0, n_jobs=-1, verbose=1)
    search.fit(X_train, y_train)
    print("Best params:", search.best_params_)
    print("Best CV score:", round(search.best_score_, 4))
    # CV mean/std for the BEST hyperparameter set from the search itself.
    scores = [search.cv_results_[f'split{i}_test_score'][search.best_index_]
              for i in range(5)]
    res_tuned = evaluate('RFtuned', search.best_estimator_, X_train, X_test,
                         y_train, y_test, cv_scores=np.array(scores))

    pd.DataFrame([res_baseline, res_tuned]).to_csv('rf_results.csv', index=False)
    print("\nSaved rf_results.csv")
    print(pd.DataFrame([res_baseline, res_tuned]).round(4).to_string(index=False))

if __name__ == '__main__':
    main()