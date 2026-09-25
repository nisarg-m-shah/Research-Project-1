"""WP3 - Recursive Feature Elimination with Cross Validation (RFECV).

Runs RFECV (step=1, scoring='roc_auc', TimeSeriesSplit(5)) with two
estimators on the winning V0 feature set:
  - LogisticRegression(C=1.0, max_iter=2000)
  - XGBoost with the FIXED tuned Block 3 params (evaluation only).

Outputs:
  rfecv_cv_curve.csv   - CV ROC-AUC mean/std vs number of features (per
                         estimator, so the Ch.6.3 / Fig 13 curve can be drawn)
  rfecv_results.csv    - outcome summary per estimator: n_features selected,
                         selected feature list, best CV AUC (mean/std),
                         and the full-report test metrics for that subset
  fig13_rfecv_curve.png - CV AUC vs feature count for both estimators,
                         optimal point marked.

RFECV never shuffles data; TimeSeriesSplit is used throughout.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from eval_utils import full_report, ks_gini, best_threshold_macro_f1

pd.set_option('display.width', 160)

tscv = TimeSeriesSplit(n_splits=5)

_hp = pd.read_csv("best_hyperparameters.csv", index_col=0).iloc[:, 0]
XGB_PARAMS = {
    'max_depth': int(_hp['max_depth']), 'learning_rate': float(_hp['learning_rate']),
    'n_estimators': int(_hp['n_estimators']), 'subsample': float(_hp['subsample']),
    'colsample_bytree': float(_hp['colsample_bytree']),
    'min_child_weight': int(_hp['min_child_weight']),
    'reg_alpha': float(_hp['reg_alpha']), 'reg_lambda': float(_hp['reg_lambda']),
}

def load_v0():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    for df_ in (X_train, X_test):
        if 'deposit_value_usd' in df_.columns:
            df_.drop(columns=['deposit_value_usd'], inplace=True)
    return X_train, X_test, y_train, y_test

ESTIMATORS = [
    ('LogReg', LogisticRegression(C=1.0, max_iter=2000)),
    ('XGBtuned', XGBClassifier(**XGB_PARAMS, eval_metric='auc', random_state=0)),
]

def main():
    X_train, X_test, y_train, y_test = load_v0()
    feats = X_train.columns.tolist()
    print(f"V0 features: {len(feats)}")

    curve_rows, result_rows, cv_curves = [], [], {}

    for name, est in ESTIMATORS:
        print(f"\n--- RFECV with {name} ---")
        rfecv = RFECV(estimator=est, step=1, cv=tscv, scoring='roc_auc',
                      n_jobs=-1, min_features_to_select=1)
        rfecv.fit(X_train.values, y_train.values)

        mask = rfecv.support_
        selected = [f for f, m in zip(feats, mask) if m]
        n_sel = int(mask.sum())

        cr = rfecv.cv_results_
        curve = pd.DataFrame({
            'n_features': cr['n_features'].astype(int),
            'cv_auc_mean': cr['mean_test_score'],
            'cv_auc_std': cr['std_test_score'],
            'estimator': name,
        })
        curve.to_csv('rfecv_cv_curve.csv', mode='a' if name == 'XGBtuned' else 'w',
                     index=False, header=(name == 'LogReg'))
        cv_curves[name] = curve

        best_idx = int(np.argmax(cr['mean_test_score']))
        best_cv = float(cr['mean_test_score'][best_idx])
        best_std = float(cr['std_test_score'][best_idx])

        # Test evaluation on the selected subset, threshold from train macro-F1
        est2 = XGBClassifier(**XGB_PARAMS, eval_metric='auc', random_state=0) \
            if name == 'XGBtuned' else LogisticRegression(C=1.0, max_iter=2000)
        est2.fit(X_train.values[:, mask], y_train.values)
        tr_probs = est2.predict_proba(X_train.values[:, mask])[:, 1]
        te_probs = est2.predict_proba(X_test.values[:, mask])[:, 1]
        th = best_threshold_macro_f1(y_train, tr_probs)
        te = full_report(f"{name} RFECV subset ({n_sel} feats) - TEST",
                         y_test, te_probs, th)
        tr_auc = ks_gini(y_train, tr_probs)[2]

        print(f"[{name}] selected {n_sel}/{len(feats)} features, "
              f"CV AUC {best_cv:.4f} (std {best_std:.4f}), "
              f"test AUC {te['auc']:.4f}, threshold {th:.2f}")

        result_rows.append({
            'estimator': name, 'n_features_total': len(feats),
            'n_features_selected': n_sel,
            'selected_features': '; '.join(selected),
            'best_cv_auc_mean': best_cv, 'best_cv_auc_std': best_std,
            'test_auc': te['auc'], 'test_gini': te['gini'],
            'test_ks_pct': te['ks_pct'], 'test_pr_auc': te['pr_auc'],
            'test_brier': te['brier'], 'threshold': th,
            'test_accuracy': te['accuracy'], 'test_macro_f1': te['macro_f1'],
            'cm_tn': int(te['cm'][0, 0]), 'cm_fp': int(te['cm'][0, 1]),
            'cm_fn': int(te['cm'][1, 0]), 'cm_tp': int(te['cm'][1, 1]),
            'train_auc': tr_auc, 'gap': tr_auc - te['auc'],
        })

    pd.DataFrame(result_rows).to_csv('rfecv_results.csv', index=False)
    print("\nSaved rfecv_results.csv and rfecv_cv_curve.csv")

    # Fig 13: CV AUC vs feature count, both estimators, optimal marked
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'LogReg': '#1f77b4', 'XGBtuned': '#ff7f0e'}
    for name, curve in cv_curves.items():
        ax.plot(curve['n_features'], curve['cv_auc_mean'],
                marker='o', ms=4, lw=1.5, color=colors[name], label=name)
        opt = int(curve.loc[curve['cv_auc_mean'].idxmax(), 'n_features'])
        val = curve['cv_auc_mean'].max()
        ax.axvline(opt, color=colors[name], ls=':', alpha=0.6)
        ax.annotate(f'{name}: {opt} feat., CV {val:.4f}', (opt, val),
                    textcoords='offset points', xytext=(8, 10), fontsize=8,
                    color=colors[name])
    ax.set_xlabel('Number of features (random feature elimination, step=1)')
    ax.set_ylabel('5-fold TimeSeriesSplit ROC-AUC')
    ax.set_title('RFECV: CV ROC-AUC vs number of features')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig13_rfecv_curve.png', dpi=150)
    print("Saved fig13_rfecv_curve.png")

    print("\n" + "="*60)
    print(pd.DataFrame(result_rows).drop(columns=['selected_features']).round(4).to_string(index=False))

if __name__ == '__main__':
    main()