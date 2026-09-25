"""Shared evaluation battery for the Uniswap V3 LP profitability project.

Extracted from block2modelling.py (Block 2) verbatim so that every runner
(new and old) uses one identical implementation. Behaviour-neutral changes
only: the em dash inside full_report's built-in leakage warning was replaced
with a plain hyphen, and full_report's return dict was EXTENDED in a
backward-compatible way (the original five keys are unchanged).

CONTEXT (project rules):
  - ks is returned as a FRACTION; the report quotes it on the x100 scale
    (ks_pct = ks * 100).
  - full_report's built-in warning (gini > 0.75 or auc > 0.9) is far too
    loose for this project, whose ceiling is about 0.60 AUC. Runners add the
    project-specific guard (test ROC-AUC more than ~0.02 above the 0.5955
    baseline trips a leakage re-audit) in their own scripts.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    average_precision_score, brier_score_loss, f1_score,
    accuracy_score, precision_score, recall_score,
)


def ks_gini(y_true, probs):
    """KS statistic (fraction), Gini (=2*AUC-1) and ROC-AUC.
    Identical to Block 2's implementation."""
    df_ = pd.DataFrame({'y': y_true.values, 'p': probs})
    df_['decile'] = pd.qcut(df_['p'].rank(method='first'), 10, labels=False)
    grp = df_.groupby('decile')['y'].agg(['sum', 'count'])
    grp['cum_bad'] = (grp['sum'] / grp['sum'].sum()).cumsum()
    grp['cum_good'] = ((grp['count'] - grp['sum']) / (grp['count'] - grp['sum']).sum()).cumsum()
    ks = (grp['cum_bad'] - grp['cum_good']).abs().max()
    auc = roc_auc_score(y_true, probs)
    gini = 2 * auc - 1
    return ks, gini, auc


def decile_lift_table(y_true, probs):
    """Decile (1 = highest predicted profitability), n, positives, positive
    rate, lift vs overall base rate, and cumulative % of positives.
    Identical to Block 2's implementation."""
    df_ = pd.DataFrame({'y': y_true.values, 'p': probs})
    df_['decile'] = pd.qcut(df_['p'].rank(method='first', ascending=False), 10, labels=range(1, 11))
    tbl = df_.groupby('decile').agg(n=('y', 'count'), positive=('y', 'sum'))
    tbl['positive_rate'] = tbl['positive'] / tbl['n']
    overall_rate = df_['y'].mean()
    tbl['lift'] = tbl['positive_rate'] / overall_rate
    tbl['cum_positive_pct'] = (tbl['positive'].cumsum() / tbl['positive'].sum() * 100)
    return tbl


def best_threshold_positive_f1(y_true, probs):
    """LEGACY Block-2 selector: binary (positive-class-only) F1 sweep.
    Kept only to reproduce and document the threshold discrepancy (Session 2
    WP0, brief Section 3b). Do not use for the report: the project rule is
    train macro-F1 (see best_threshold_macro_f1)."""
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.71, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int))
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def best_threshold_macro_f1(y_true, probs):
    """Project threshold policy: train macro-F1-optimal threshold
    (handoff Section 4 / brief Section 3b)."""
    best_t, best_f1 = 0.5, 0
    for t in np.arange(0.3, 0.71, 0.01):
        f1 = f1_score(y_true, (probs >= t).astype(int), average='macro')
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    return best_t


def full_report(name, y_true, probs, threshold):
    """Full evaluation battery. Printed output is unchanged from Block 2
    (except the em dash in the leak warning, which is behaviour-neutral).
    Return dict is backward-compatible: the original five keys
    {'auc','pr_auc','ks','gini','brier'} are unchanged; extra keys are
    appended (ks_pct, threshold, cm, accuracy, precision, recall, f1,
    macro_f1, decile_table)."""
    preds = (probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, preds)
    print(f"\n{'='*60}\n{name}  (threshold={threshold:.2f})\n{'='*60}")
    print(cm)
    print(classification_report(y_true, preds, digits=3))
    ks, gini, auc = ks_gini(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)
    brier = brier_score_loss(y_true, probs)
    print(f"ROC-AUC: {auc:.4f}  PR-AUC: {pr_auc:.4f}  KS: {ks*100:.1f}  Gini: {gini:.4f}  Brier: {brier:.4f}")
    if gini > 0.75 or auc > 0.9:
        print("!! Gini/AUC unusually high - re-check for leakage before trusting this.")
    tbl = decile_lift_table(y_true, probs)
    print("\nDecile/Lift table:")
    print(tbl.round(3))
    return {
        'auc': auc, 'pr_auc': pr_auc, 'ks': ks, 'gini': gini, 'brier': brier,
        'ks_pct': ks * 100,
        'threshold': threshold,
        'cm': cm,
        'accuracy': accuracy_score(y_true, preds),
        'precision': precision_score(y_true, preds),
        'recall': recall_score(y_true, preds),
        'f1': f1_score(y_true, preds),
        'macro_f1': f1_score(y_true, preds, average='macro'),
        'decile_table': tbl,
    }