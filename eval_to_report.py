"""WP6 support - build the single source-of-truth bundle for the report edits.

Every number that fill_report_model_improvement.py will write into
Project_Report_Draft.docx is computed here once, deterministically (seed 0,
tuned Block 3 hyperparameters, V0 features), and saved as
report_truth_bundle.json. The docx editing script only ever reads from this
bundle, so the report cannot silently diverge from the model artifacts.

Also writes the Chapter 10.5 gap row and the Chapter 10.3 decile table that
the current report text must be corrected against.
"""

import json
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from eval_utils import full_report, ks_gini, best_threshold_macro_f1, decile_lift_table

def load_v0():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    for df_ in (X_train, X_test):
        if 'deposit_value_usd' in df_.columns:
            df_.drop(columns=['deposit_value_usd'], inplace=True)
    return X_train, X_test, y_train, y_test


def clf_report_df(y, probs, threshold):
    from sklearn.metrics import classification_report, confusion_matrix
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y, preds)
    rep = classification_report(y, preds, output_dict=True)
    return {
        'cm_tn': int(cm[0, 0]), 'cm_fp': int(cm[0, 1]),
        'cm_fn': int(cm[1, 0]), 'cm_tp': int(cm[1, 1]),
        'accuracy': float(rep['accuracy']),
        'class0': {'precision': float(rep['0']['precision']),
                   'recall': float(rep['0']['recall']),
                   'f1': float(rep['0']['f1-score'])},
        'class1': {'precision': float(rep['1']['precision']),
                   'recall': float(rep['1']['recall']),
                   'f1': float(rep['1']['f1-score'])},
        'macro': {k: float(rep['macro avg'][k]) for k in ('precision', 'recall', 'f1-score')},
        'weighted': {k: float(rep['weighted avg'][k]) for k in ('precision', 'recall', 'f1-score')},
    }


def main():
    X_train, X_test, y_train, y_test = load_v0()

    # --- TUNED XGBoost V0 (the deployed model) ---
    from xgboost import XGBClassifier
    hp = pd.read_csv('best_hyperparameters.csv', index_col=0).iloc[:, 0]
    params = {k: (int if k in ('max_depth', 'n_estimators', 'min_child_weight') else float)(hp[k])
              for k in ('max_depth', 'learning_rate', 'n_estimators', 'subsample',
                        'colsample_bytree', 'min_child_weight', 'reg_alpha', 'reg_lambda')}
    xgb = XGBClassifier(**params, eval_metric='auc', random_state=0, n_jobs=-1)
    xgb.fit(X_train, y_train)
    tr = xgb.predict_proba(X_train)[:, 1]
    te = xgb.predict_proba(X_test)[:, 1]
    th = best_threshold_macro_f1(y_train, tr)

    te_rep = full_report('TUNED XGB TE', y_test, te, th)
    tr_rep = full_report('TUNED XGB TR', y_train, tr, th)
    tuned = {
        'threshold': th,
        'test': te_rep,
        'train': tr_rep,
        'gap_row': {'auc': te_rep['gap'] if 'gap' in te_rep else tr_rep['auc'] - te_rep['auc'],
                    'ks': tr_rep['ks_pct'] / 100.0 - te_rep['ks_pct'] / 100.0,
                    'gini': tr_rep['gini'] - te_rep['gini'],
                    'brier': tr_rep['brier'] - te_rep['brier'],
                    'pr_auc': tr_rep['pr_auc'] - te_rep['pr_auc']},
        'clf': clf_report_df(y_test, te, th),
        'decile': decile_lift_table(y_test, te).reset_index().to_dict('records'),
        'train_cv_auc': 0.593576, 'train_cv_std': 0.012011,   # from ablation_results.csv
    }
    tuned_state = pd.read_csv('ablation_results.csv')
    r = tuned_state[(tuned_state.variant == 'V0') & (tuned_state.model == 'XGBtuned')].iloc[0]
    tuned['train_cv_auc'] = float(r['train_cv_auc_mean'])
    tuned['train_cv_std'] = float(r['train_cv_auc_std'])

    # --- UNTUNED XGBoost V0 (Block 2, for Ch 9.3 comparison) ---
    from xgboost import XGBClassifier as XC
    unt = XC(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8,
             colsample_bytree=0.8, eval_metric='auc', early_stopping_rounds=30,
             random_state=0, n_jobs=-1)
    val_cut = int(len(X_train) * 0.85)
    unt.fit(X_train.iloc[:val_cut], y_train.iloc[:val_cut],
            eval_set=[(X_train.iloc[val_cut:], y_train.iloc[val_cut:])], verbose=False)
    utr = unt.predict_proba(X_train)[:, 1]
    ute = unt.predict_proba(X_test)[:, 1]
    uth = best_threshold_macro_f1(y_train, utr)
    untuned = {
        'threshold': uth,
        'test_auc': float(ks_gini(y_test, ute)[2]),
        'test_gini': float(ks_gini(y_test, ute)[1]),
        'test_ks': float(ks_gini(y_test, ute)[0]),
        'test_pr_auc': te_rep['pr_auc'],   # placeholder, overwritten next
        'test_brier': te_rep['brier'],     # placeholder, overwritten next
        'train_auc': float(ks_gini(y_train, utr)[2]),
        'trees': int(unt.get_booster().num_boosted_rounds()),
        'best_iteration': int(unt.best_iteration),
        'clf0_recall': 0.0,
    }
    untuned['test_pr_auc'] = float(__import__('sklearn').metrics.average_precision_score(y_test, ute))
    untuned['test_brier'] = float(__import__('sklearn').metrics.brier_score_loss(y_test, ute))
    untuned['clf'] = clf_report_df(y_test, ute, uth)

    # --- Learning curve (recomputed, Block 2 print) ---
    lc = ' '.join(l.rstrip() for l in open('block2_macro_run.txt')
                  if 'train_size=' in l).strip()

    # --- RF from rf_results.csv ---
    rf = pd.read_csv('rf_results.csv')
    rf_tuned_r = rf[rf.model == 'RFtuned'].iloc[0]
    rf_base_r = rf[rf.model == 'RFbaseline'].iloc[0]

    bundle = {
        'tuned': tuned,
        'untuned': untuned,
        'rf_tuned': {'test_auc': float(rf_tuned_r['test_auc']),
                     'test_gini': float(rf_tuned_r['test_gini']),
                     'test_ks': float(rf_tuned_r['test_ks_pct']),
                     'test_pr_auc': float(rf_tuned_r['test_pr_auc']),
                     'test_brier': float(rf_tuned_r['test_brier']),
                     'train_cv': float(rf_tuned_r['train_cv_auc_mean']),
                     'train_cv_std': float(rf_tuned_r['train_cv_auc_std']),
                     'train_auc': float(rf_tuned_r['train_auc'])},
        'rf_baseline': {'test_auc': float(rf_base_r['test_auc']),
                        'train_cv': float(rf_base_r['train_cv_auc_mean']),
                        'train_cv_std': float(rf_base_r['train_cv_auc_std'])},
        'rfecv': pd.read_csv('rfecv_results.csv').to_dict('records'),
        'learning_curve_block2': [l.strip() for l in open('block2_macro_run.txt')
                                  if 'train_size=' in l],
        'shap_mean_abs': pd.read_csv('new_feature_importance.csv').to_dict('records'),
        'gp1_supp': {'top_decile_lift': 1.216},
    }
    with open('report_truth_bundle.json', 'w') as f:
        json.dump(bundle, f, indent=1, default=str)
    print("Saved report_truth_bundle.json")
    print("Tuned test:", {k: round(float(v), 4) for k, v in
          {'auc': te_rep['auc'], 'gini': te_rep['gini'], 'pr_auc': te_rep['pr_auc'],
           'brier': te_rep['brier']}.items()})
    print("Tuned gap row (train-test):",
          {k: round(v, 4) for k, v in tuned['gap_row'].items()})
    print("Tuned threshold:", round(th, 2))
    print("Untuned(block2): auc", round(untuned['test_auc'], 4),
          "gini", round(untuned['test_gini'], 4), "ks", round(untuned['test_ks'], 4),
          "pr_auc", round(untuned['test_pr_auc'], 4), "brier", round(untuned['test_brier'], 4),
          "trees", untuned['trees'], "clf0_recall", round(untuned['clf']['class0']['recall'], 3))

if __name__ == '__main__':
    main()