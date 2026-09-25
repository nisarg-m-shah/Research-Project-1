"""WP4 - Evaluation charts for the report (Figs 10, 11, 12).

All charts use deterministic-fitted models on the winning V0 features:
  - Fig 10 fig10_decile_lift.png: decile lift bar chart (tuned XGBoost V0),
    built directly from eval_utils.decile_lift_table so the Ch.10.3 chart and
    the Ch.10.3 table always agree.
  - Fig 11 fig11_cumulative_gain.png: cumulative gain curve (tuned XGB V0)
    vs the random-chance diagonal.
  - Fig 12 fig12_calibration_curve.png: reliability diagram
    (calibration_curve, n_bins=10) for tuned XGBoost, Logistic Regression
    and the tuned Random Forest.

Predictions are the saved test prediction CSVs from block2c / block2b
(deterministic, seed 0). LR V0 is refit here with seed-independent `C=1.0`.
Data behind the figures is also dumped to CSV for a manual audit.
dpi=150. No shuffling anywhere; probabilities are test-time only.
"""

import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression

from eval_utils import decile_lift_table

def load_v0():
    X_train = pd.read_csv("X_train.csv")
    X_test = pd.read_csv("X_test.csv")
    y_train = pd.read_csv("y_train.csv").iloc[:, 0]
    y_test = pd.read_csv("y_test.csv").iloc[:, 0]
    for df_ in (X_train, X_test):
        if 'deposit_value_usd' in df_.columns:
            df_.drop(columns=['deposit_value_usd'], inplace=True)
    return X_train, X_test, y_train, y_test

def main():
    X_train, X_test, y_train, y_test = load_v0()

    xgb_pred = pd.read_csv('xgb_v0_test_predictions.csv')
    rf_pred = pd.read_csv('rf_tuned_test_predictions.csv')
    assert len(xgb_pred) == len(y_test) == len(rf_pred) == 12450
    assert (xgb_pred['y_true'].values == y_test.values).all()
    assert (rf_pred['y_true'].values == y_test.values).all()

    lr = LogisticRegression(C=1.0, max_iter=2000)
    lr.fit(X_train, y_train)
    lr_probs = lr.predict_proba(X_test)[:, 1]

    probs = {'XGBoost tuned': xgb_pred['prob'].values,
             'Logistic Regression': lr_probs,
             'Random Forest tuned': rf_pred['prob'].values}

    # --- Fig 10: decile lift (tuned XGBoost V0) ---
    lift = decile_lift_table(y_test, probs['XGBoost tuned'])
    lift.to_csv('decile_lift_xgb_v0.csv')
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(lift.index.astype(int), lift['lift'], color='#1f77b4', alpha=0.85,
           edgecolor='black', lw=0.4)
    ax.axhline(1.0, color='grey', lw=1, ls='--', label='No lift (random = 1.0)')
    ax.set_xlabel('Test-decile of predicted profitability probability')
    ax.set_ylabel('Lift (share of profitable vs baseline)')
    ax.set_title('Decile lift - tuned XGBoost (V0 features, test)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig10_decile_lift.png', dpi=150)
    plt.close(fig)

    # --- Fig 11: cumulative gain (tuned XGBoost V0) ---
    cumpct = lift['cum_positive_pct'].values
    dec = np.arange(1, 11)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(dec, cumpct, marker='o', ms=5, color='#ff7f0e', lw=2,
            label='Tuned XGBoost')
    ax.plot(dec, np.linspace(12.0, 100.0, 10), ls='--', color='grey',
            label='Random baseline')
    ax.set_xlabel('Top-k test deciles by predicted probability')
    ax.set_ylabel('Cumulative share of profitable positions (%)')
    ax.set_title('Cumulative gain - tuned XGBoost (V0 features, test)')
    ax.set_xlim(1, 10)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig11_cumulative_gain.png', dpi=150)
    plt.close(fig)

    # --- Fig 12: calibration curve (n_bins=10, all three models) ---
    cal_rows = []
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {'XGBoost tuned': '#1f77b4', 'Logistic Regression': '#2ca02c',
              'Random Forest tuned': '#d62728'}
    for name, p in probs.items():
        frac_pos, mean_pred = calibration_curve(y_test, p, n_bins=10,
                                                strategy='uniform')
        ax.plot(mean_pred, frac_pos, marker='o', ms=5, lw=2,
                color=colors[name], label=name)
        cal_rows.append(pd.DataFrame({'model': name, 'mean_predicted': mean_pred,
                                      'fraction_positive': frac_pos}))
    ax.plot([0, 1], [0, 1], ls='--', color='grey', label='Perfectly calibrated')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Fraction of positives')
    ax.set_title('Calibration curve (test, 10 uniform bins)')
    ax.text(0.02, 0.97, 'Note: bins with no observations are omitted.',
            transform=ax.transAxes, fontsize=8, va='top')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('fig12_calibration_curve.png', dpi=150)
    plt.close(fig)
    pd.concat(cal_rows, ignore_index=True).to_csv('calibration_data.csv', index=False)

    # --- Programmatic verification of what was drawn ---
    assert len(lift) == 10 and lift['lift'].between(0, 3).all()
    assert cumpct[0] >= 12.0 and abs(cumpct[-1] - 100.0) < 1e-9
    bin_counts = []
    for name, p in probs.items():
        fp, mp = calibration_curve(y_test, p, n_bins=10, strategy='uniform')
        assert len(fp) > 0 and len(fp) == len(mp)
        assert float(fp.min()) >= 0.0 and float(fp.max()) <= 1.0
        assert float(mp.min()) >= 0.0 and float(mp.max()) <= 1.0
        bin_counts.append(len(fp))
    print("Fig 10/11/12 verification: OK "
          "(10 deciles, gain reaches 100%, calibration points within [0,1] "
          f"per model, non-empty uniform bins: {bin_counts})")
    print("Top-decile lift (tuned XGBoost):", round(float(lift['lift'].iloc[0]), 3))
    print("Saved fig10_decile_lift.png, fig11_cumulative_gain.png, fig12_calibration_curve.png")

if __name__ == '__main__':
    main()