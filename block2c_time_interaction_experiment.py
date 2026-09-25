"""WP1 - Open-time and interaction feature ablation (V0..V3).

Ablation ladder (identical chronological split for all variants):
  V0: original 27 features (existing X_train.csv/X_test.csv, byte-verified)
  V1: V0 + cyclical open-time features (hour + weekday, UTC)
  V2: V0 + two domain-grounded interaction features (product of train-fitted
      z-scores)
  V3: V0 + time + interactions

For each variant, for Logistic Regression and XGBoost (FIXED tuned Block 3
params; no re-tuning, so the comparison isolates the features):
  - 5-fold TimeSeriesSplit CV ROC-AUC on TRAIN only (mean, std)
  - test-set full_report() with threshold re-derived from train macro-F1
  - train AUC and train-test gap

DECISION RULE: the winning variant is chosen on TRAIN-CV AUC only. Test
metrics are reported for all variants for transparency but the selection is
never made on test. A null result (within the ~0.001 noise floor and CV fold
std) is a legitimate, reportable finding.

Leakage guard: any variant pushing test ROC-AUC more than ~0.02 above the
0.5955 baseline trips a hard stop / re-audit before anything is written.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from xgboost import XGBClassifier

import block1prep
from eval_utils import (
    full_report, ks_gini, best_threshold_macro_f1,
)

pd.set_option('display.width', 160)

# Tuned Block 3 hyperparameters (best_hyperparameters.csv), read the same way
# the report artifacts were created.
_hp = pd.read_csv("best_hyperparameters.csv", index_col=0).iloc[:, 0]
XGB_PARAMS = {
    'max_depth': int(_hp['max_depth']),
    'learning_rate': float(_hp['learning_rate']),
    'n_estimators': int(_hp['n_estimators']),
    'subsample': float(_hp['subsample']),
    'colsample_bytree': float(_hp['colsample_bytree']),
    'min_child_weight': int(_hp['min_child_weight']),
    'reg_alpha': float(_hp['reg_alpha']),
    'reg_lambda': float(_hp['reg_lambda']),
}
print("Tuned XGBoost params:", XGB_PARAMS)

N_FOLDS = 5
tscv = TimeSeriesSplit(n_splits=N_FOLDS)
LEAK_BASE = 0.5955
LEAK_TOL = 0.02


def load_v0():
    """Load the original, byte-verified Block 1 outputs."""
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    for df_ in (X_train, X_test):
        if 'deposit_value_usd' in df_.columns:
            df_.drop(columns=['deposit_value_usd'], inplace=True)
    return X_train, X_test, y_train, y_test


VARIANTS = {
    'V0': (False, False),
    'V1': (True, False),
    'V2': (False, True),
    'V3': (True, True),
}


def train_cv_auc(model, X_train, y_train):
    scores = cross_val_score(model, X_train, y_train, cv=tscv,
                             scoring='roc_auc', n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def run_variant(name, X_train, X_test, y_train, y_test, results, preds_out):
    print(f"\n{'#'*70}\nVARIANT {name}\n{'#'*70}")
    print(f"Features: {X_train.shape[1]} columns")

    for model_name, model in [
        ('LogReg', LogisticRegression(max_iter=2000, C=1.0)),
        ('XGBtuned', XGBClassifier(**XGB_PARAMS, eval_metric='auc',
                                   random_state=0, n_jobs=-1)),
    ]:
        cv_mean, cv_std = train_cv_auc(model, X_train, y_train)
        model.fit(X_train, y_train)
        tr_probs = model.predict_proba(X_train)[:, 1]
        te_probs = model.predict_proba(X_test)[:, 1]
        threshold = best_threshold_macro_f1(y_train, tr_probs)
        te = full_report(f"{name} {model_name} - TEST", y_test, te_probs, threshold)
        tr_auc = ks_gini(y_train, tr_probs)[2]
        gap = tr_auc - te['auc']

        if te['auc'] > LEAK_BASE + LEAK_TOL:
            print(f"!! LEAKAGE GUARD: {name} {model_name} test AUC {te['auc']:.4f} "
                  f"is > {LEAK_BASE + LEAK_TOL:.3f}. STOP - re-audit before writing anything.")
        print(f"[{name} {model_name}] train_cv_auc={cv_mean:.4f} (std {cv_std:.4f}) "
              f"test_auc={te['auc']:.4f} train_auc={tr_auc:.4f} gap={gap:.4f} "
              f"threshold={threshold:.2f}")

        if name == 'V0' and model_name == 'XGBtuned':
            pd.DataFrame({'y_true': y_test.values,
                          'prob': te_probs}).to_csv('xgb_v0_test_predictions.csv', index=False)
            pd.DataFrame({'y_true': y_train.values,
                          'prob': tr_probs}).to_csv('xgb_v0_train_predictions.csv', index=False)

        results.append({
            'variant': name, 'model': model_name,
            'n_features': X_train.shape[1],
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
        })


def coverage_and_spans():
    df = pd.read_csv('final_merged_dataset_v5.csv')
    df['open_time'] = pd.to_datetime(df['open_time'])
    df = df.sort_values('open_time').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    split_time = df.iloc[split_idx]['open_time']
    train_df = df[df['open_time'] < split_time]
    test_df = df[df['open_time'] >= split_time]
    for name, d in [('train', train_df), ('test', test_df)]:
        ot = d['open_time']
        print(f"\n{name}: rows={len(d)} span_days={(ot.max()-ot.min()).days} "
              f"hours_present={ot.dt.hour.nunique()} dows_present={ot.dt.dayofweek.unique().tolist()} "
              f"day0..6 counts={ot.dt.dayofweek.value_counts().sort_index().to_dict()}")
    return train_df, test_df


def main():
    results = []
    coverage_and_spans()

    V0 = load_v0()
    for name, (tflag, iflag) in VARIANTS.items():
        if name == 'V0':
            X_tr, X_te, y_tr, y_te = V0
        else:
            out = block1prep.build_datasets(add_time_features=tflag,
                                            add_interactions=iflag,
                                            diagnostics=False, save=False)
            X_tr, X_te, y_tr, y_te = (out['X_train'], out['X_test'],
                                      out['y_train'], out['y_test'])
            for df_ in (X_tr, X_te):
                if 'deposit_value_usd' in df_.columns:
                    df_.drop(columns=['deposit_value_usd'], inplace=True)
        run_variant(name, X_tr, X_te, y_tr, y_te, results, None)

    res_df = pd.DataFrame(results)
    res_df.to_csv('ablation_results.csv', index=False)
    print("\nSaved ablation_results.csv")
    print("\n" + "="*70)
    print("ABLATION SUMMARY (train-CV AUC decides the winner)")
    print("="*70)
    print(res_df.round(4).to_string(index=False))


if __name__ == '__main__':
    main()