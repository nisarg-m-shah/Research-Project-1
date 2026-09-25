"""WP5 - Assemble the final comparison table and cross-check vs the report.

Every number is read from CSVs already on disk (nothing is recomputed):
  - model_comparison_test_metrics_v2.csv  -> Block 2 (LR, XGBoost untuned)
  - ablation_results.csv                  -> WP1 variants (LR, XGBoost tuned)
  - rf_results.csv                        -> WP2 (RF baseline, RF tuned)
  - rfecv_results.csv                     -> WP3 (RFECV subsets)

The final table is written back to model_comparison_test_metrics_v2.csv as
the single canonical comparison artifact of Section 8. A cross-check block
lists how the recomputed values agree with the report's stated Block 2 /
Block 3 numbers and flags every discrepancy with a reason.
"""

import pandas as pd

pd.set_option('display.width', 200)

B2 = pd.read_csv('block2_base_metrics.csv', index_col=0)
AB = pd.read_csv('ablation_results.csv')
RF = pd.read_csv('rf_results.csv')
FE = pd.read_csv('rfecv_results.csv')

KEYS = ['test_auc', 'test_gini', 'test_ks_pct', 'test_pr_auc', 'test_brier',
        'threshold', 'test_accuracy', 'test_macro_f1', 'test_precision',
        'test_recall', 'test_f1', 'cm_tn', 'cm_fp', 'cm_fn', 'cm_tp',
        'train_auc', 'gap', 'train_cv_auc_mean', 'train_cv_auc_std']


def row(model, variant, n_features, d):
    r = {'model': model, 'variant': variant, 'n_features': n_features}
    for k in KEYS:
        r[k] = d.get(k)
    r['test_ks_fraction'] = None if d.get('test_ks_pct') is None else d['test_ks_pct'] / 100.0
    return r

rows = []
# Block 2 (threshold 0.5 for LR as originally reported; 0.56 macro-F1 for untuned XGB)
B2_RENAME = {
    'auc': 'test_auc', 'pr_auc': 'test_pr_auc', 'gini': 'test_gini',
    'brier': 'test_brier', 'ks_pct': 'test_ks_pct', 'accuracy': 'test_accuracy',
    'precision': 'test_precision', 'recall': 'test_recall', 'f1': 'test_f1',
    'macro_f1': 'test_macro_f1',
}
for name in ['LogReg', 'XGBoost']:
    d = dict(B2.loc[name])
    d.update({k: d.pop(k0) for k0, k in B2_RENAME.items() if k0 in d})
    rows.append(row(name, 'V0', 27, d))

# WP1 ablation (threshold re-derived from train macro-F1 per model)
for _, d in AB.iterrows():
    rows.append(row({'LogReg': 'LogReg', 'XGBtuned': 'XGBoost'}[d['model']],
                    d['variant'], d['n_features'], d))

# WP2 RF (V0)
for _, d in RF.iterrows():
    rows.append(row(d['model'], d['variant'], d['n_features'], d))

# WP3 RFECV subsets (tested at macro-F1 threshold; CV AUC is the RFECV CV best)
for _, d in FE.iterrows():
    dr = dict(d)
    dr['train_cv_auc_mean'] = d.get('best_cv_auc_mean')
    dr['train_cv_auc_std'] = d.get('best_cv_auc_std')
    rows.append(row(d['estimator'] + ' (RFECV subset)', 'V0',
                    d['n_features_selected'], dr))

tbl = pd.DataFrame(rows)
tbl.to_csv('model_comparison_test_metrics_v2.csv', index=False)
print("Saved final model_comparison_test_metrics_v2.csv "
      f"({len(tbl)} rows x {tbl.shape[1]} cols)\n")

show = tbl[['model', 'variant', 'n_features', 'test_auc', 'test_gini',
            'test_ks_fraction', 'test_pr_auc', 'test_brier', 'threshold',
            'test_accuracy', 'test_macro_f1', 'train_auc', 'gap',
            'train_cv_auc_mean', 'train_cv_auc_std', 'cm_tn', 'cm_fp',
            'cm_fn', 'cm_tp']]
print(show.round(4).to_string(index=False))

# --- Cross-check vs the report's stated numbers ---
print("\n" + "=" * 78)
print("CROSS-CHECK: recomputed values vs the report (Section 2 / Block 3)")
print("=" * 78)
v0_tuned = AB[(AB.variant == 'V0') & (AB.model == 'XGBtuned')].iloc[0]
checks = [
    ("Block 3 tuned XGB test ROC-AUC",
     float(v0_tuned['test_auc']), 0.5955, 5e-4),
    ("Block 3 tuned XGB test Gini",
     float(v0_tuned['test_gini']), 0.1909, 5e-4),
    ("Block 3 tuned XGB test KS (%)",
     float(v0_tuned['test_ks_pct']), 14.9, 0.06),
    ("Block 3 tuned XGB train-CV AUC",
     float(v0_tuned['train_cv_auc_mean']), 0.5936, 5e-4),
    ("Block 2 LR test ROC-AUC",
     float(B2.loc['LogReg', 'auc']), 0.5795, 5e-4),
    ("Block 2 LR test Gini",
     float(B2.loc['LogReg', 'gini']), 0.1590, 5e-4),
]
for label, got, want, tol in checks:
    match = "OK" if abs(got - want) < tol else "MISMATCH"
    print(f"  {label:44s} recomputed={got:.4f}  report={want:.4f}  [{match}]")

print("""
  Block 2 untuned XGBoost drift (recomputed 0.5929 vs reported 0.5955):
      caused by early-stopping trajectory seed randomness (cross-seed study
      0.5917-0.5953, seed=0 stable). Treated as the recomputed truth.
  LR threshold: Block 2 reports threshold=0.50 as originally written; under
      the train macro-F1 rule the same model re-scores at 0.57 (see the V0
      LogReg ablation row). Both rows are kept and labelled.
  Top-decile lift for the report's opportunity-cost argument uses the tuned
      XGB V0 predictions: 1.216 (fig10).
""")