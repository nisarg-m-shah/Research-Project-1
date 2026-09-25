"""
Block 1 - Modeling Prep & Variable Selection
Uniswap V3 LP Profitability Classifier

Input:  final_merged_dataset_v5.csv (62,250 rows, 35 cols)
Output: X_train, X_test, y_train, y_test (scaled, leakage-safe,
        one-hot + StandardScaler fit on TRAIN only)
        + a ranked variable-selection report

Run this BEFORE any model training.

Session-2 WP0 changes (behaviour-preserving for the default path):
  - Refactored into importable functions; scripts executed at import time
    previously (block2modelling.py set the bug in 3b), so block1prep.py is
    now import-safe for the WP1 ablation runner.
  - Stale docstring corrected (was 55,279 rows / 34 cols; real is
    62,250 / 35).
  - TWO NEW FLAGS, both default False:
      ADD_TIME_FEATURES    adds cyclical open-time features (hour + weekday,
                           UTC) to the model matrix.
      ADD_INTERACTIONS     adds two domain-grounded interactions built as
                           products of TRAIN-fitted z-scores.
    With both False the produced files must be byte-identical to the
    pre-session X_train.csv / X_test.csv / y_train.csv / y_test.csv /
    rf_feature_importance.csv / lasso_selected_features.csv (verified in
    WP0 by SHA-256). With either True, only NEW file names are written
    (X_train_v2.csv, X_test_v2.csv) - never the originals.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.width', 120)

DATASET = "final_merged_dataset_v5.csv"

# Columns knowable at open time only. Everything outcome-side is excluded.
LEAKAGE_COLS = [
    'close_time',
    'token0_price_close', 'token1_price_close',
    'held_value_usd', 'returned_value_usd',
    'true_fees_value_usd', 'impermanent_loss_usd',
    'position_duration_hours',   # = close_time - open_time, not known at open
    'decrease_event_count',      # counts withdrawal events across the WHOLE life
]
ID_COLS = ['tokenId', 'open_time']
TARGET_COL = 'profitable'
categorical_to_encode = ['pool_address']
drop_raw_address_cols = ['token0_address', 'token1_address']

# --- WP1 time features (cyclical encoding; raw integers NOT kept in X) ---
TIME_FEATURES = ['open_hour_sin', 'open_hour_cos', 'open_dow_sin', 'open_dow_cos']

# --- WP1 interaction features (product of TRAIN-fitted z-scores) ---
INTERACTION_PAIRS = [
    ('range_width_normalized', 'token0_price_std_24h'),
    ('fee_tier_pct', 'pre_open_avg_daily_swap_count_3d'),
]
INTERACTION_COLS = [
    'interaction_range_width_x_price_std',
    'interaction_fee_tier_x_swap_count',
]


def load_data():
    """Load, parse datetimes (open_time is tz-aware UTC in the raw CSV),
    add cyclical time features if requested, return the full frame."""
    df = pd.read_csv(DATASET)
    df['open_time'] = pd.to_datetime(df['open_time'])
    df['close_time'] = pd.to_datetime(df['close_time'])
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    print(f"open_time dtype: {df['open_time'].dtype}  (tz: {df['open_time'].dt.tz})")
    print(f"Date range: {df['open_time'].min()} -> {df['open_time'].max()}")
    print(f"Class balance: {df['profitable'].value_counts(normalize=True).to_dict()}")
    return df


def apply_time_features(df):
    """Cyclical sin/cos encoding of open hour (period 24) and open weekday
    (period 7). Derived from the UTC open_time. Raw integers are not kept in
    the model matrix, per WP1 (b)."""
    hour = df['open_time'].dt.hour
    dow = df['open_time'].dt.dayofweek
    df['open_hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['open_hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['open_dow_sin'] = np.sin(2 * np.pi * dow / 7)
    df['open_dow_cos'] = np.cos(2 * np.pi * dow / 7)
    print(f"Added cyclical time features: {TIME_FEATURES}")
    return df


def apply_interactions(train_df, test_df):
    """Two domain-grounded interactions as products of TRAIN-fitted z-scores
    (so one pool's absolute scale does not dominate):
      (1) range_width_normalized x token0_price_std_24h
          - does a tight range get riskier when the token is more volatile?
      (2) fee_tier_pct x pre_open_avg_daily_swap_count_3d
          - fee-tier/volume relationship flagged at r = -0.70 in the
            Chapter 3.3 heatmap; does it interact with profitability?
    All four source columns are knowable at open time (pre-open/at-open
    values), checked against the Block-1 leakage list."""
    for (a, b), name in zip(INTERACTION_PAIRS, INTERACTION_COLS):
        scaler = StandardScaler().fit(train_df[[a, b]])
        train_df[name] = np.prod(scaler.transform(train_df[[a, b]]), axis=1)
        test_df[name] = np.prod(scaler.transform(test_df[[a, b]]), axis=1)
    print(f"Added interaction features: {INTERACTION_COLS}")
    return train_df, test_df


def train_test_split_chronological(df):
    """STEP 1 in the original script: time-based 80/20 split by open_time,
    no shuffling. A random split would leak future market regimes."""
    df = df.sort_values('open_time').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)
    split_time = df.iloc[split_idx]['open_time']
    train_df = df[df['open_time'] < split_time].reset_index(drop=True)
    test_df = df[df['open_time'] >= split_time].reset_index(drop=True)
    print(f"\nSplit at {split_time}")
    print(f"Train: {len(train_df)} rows ({train_df['open_time'].min()} -> {train_df['open_time'].max()})")
    print(f"Test:  {len(test_df)} rows ({test_df['open_time'].min()} -> {test_df['open_time'].max()})")
    print(f"Train class balance: {train_df['profitable'].mean():.3f} profitable")
    print(f"Test class balance:  {test_df['profitable'].mean():.3f} profitable")
    return train_df, test_df


def build_xy(source_df, feature_cols, categorical_cols, pool_categories=None):
    X = source_df[feature_cols].copy()
    for c in categorical_cols:
        dummies = pd.get_dummies(X[c], prefix=c, drop_first=True)
        X = X.drop(columns=[c]).join(dummies)
    if pool_categories is not None:
        for col in pool_categories:
            if col not in X.columns:
                X[col] = 0
        X = X[pool_categories]
    y = source_df[TARGET_COL]
    return X, y


def build_datasets(add_time_features=False, add_interactions=False,
                   diagnostics=True, save=False):
    """Full Block 1 pipeline. Returns a dict with the scaled matrices, y
    vectors and (when diagnostics=True) rf_importance / l1_kept.
    save=True writes files: the original names when both flags are False,
    X_train_v2.csv / X_test_v2.csv otherwise."""
    df = load_data()
    if add_time_features:
        df = apply_time_features(df)
    train_df, test_df = train_test_split_chronological(df)

    print(f"\nExcluded as leakage: {LEAKAGE_COLS}")
    print(f"Excluded as ID/redundant: {ID_COLS + drop_raw_address_cols}")

    exclude_all = LEAKAGE_COLS + ID_COLS + drop_raw_address_cols + [TARGET_COL]
    feature_cols = [c for c in df.columns if c not in exclude_all]
    if add_interactions:
        train_df, test_df = apply_interactions(train_df, test_df)
        feature_cols = feature_cols + INTERACTION_COLS
    # cyclical time cols are already in df.columns (added before split), so
    # they enter feature_cols naturally; this line keeps the order explicit.
    if add_time_features:
        feature_cols = [c for c in feature_cols if c not in TIME_FEATURES] + TIME_FEATURES
    print(f"Remaining feature columns ({len(feature_cols)}): {feature_cols}")

    X_train, y_train = build_xy(train_df, feature_cols, categorical_to_encode)
    train_dummy_cols = [c for c in X_train.columns]
    X_test, y_test = build_xy(test_df, feature_cols, categorical_to_encode,
                              pool_categories=train_dummy_cols)
    print(f"\nX_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    binary_like = [c for c in numeric_cols if X_train[c].dropna().isin([0, 1]).all()]
    cols_to_scale = [c for c in numeric_cols if c not in binary_like]

    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()
    X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])
    print(f"Scaled {len(cols_to_scale)} continuous columns; left {len(binary_like)} binary/dummy columns unscaled")

    rf_importance, l1_kept = None, None
    if diagnostics:
        print("\n--- Filter: near-zero variance ---")
        variances = X_train[cols_to_scale].var()
        low_var = variances[variances < 1e-6]
        print(f"Near-zero-variance columns (candidates to drop): {list(low_var.index) if len(low_var) else 'none'}")

        print("\n--- Filter: pairwise correlation > 0.9 (redundancy check) ---")
        corr = X_train[cols_to_scale].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr_pairs = [(col, row, upper.loc[row, col])
                           for col in upper.columns for row in upper.index
                           if pd.notna(upper.loc[row, col]) and upper.loc[row, col] > 0.9]
        if high_corr_pairs:
            for a, b, v in high_corr_pairs:
                print(f"  {a} <-> {b}: r={v:.3f}")
        else:
            print("  none above 0.9")

        print("\n--- Embedded: L1 logistic (sparse selection) ---")
        l1 = LogisticRegression(penalty='l1', solver='liblinear', C=0.1, max_iter=2000)
        l1.fit(X_train_scaled, y_train)
        l1_coefs = pd.Series(l1.coef_[0], index=X_train_scaled.columns)
        l1_kept = l1_coefs[l1_coefs != 0].sort_values(key=abs, ascending=False)
        l1_dropped = l1_coefs[l1_coefs == 0].index.tolist()
        print(f"Lasso kept {len(l1_kept)} / {len(l1_coefs)} features")
        print(l1_kept)
        print(f"Lasso zeroed out: {l1_dropped}")

        print("\n--- Embedded: Random Forest importance ---")
        rf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                    min_samples_leaf=5, random_state=0, n_jobs=-1)
        rf.fit(X_train, y_train)  # trees don't need scaling
        rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
        print(rf_importance.head(20))

    out = dict(X_train=X_train_scaled, X_test=X_test_scaled,
               y_train=y_train, y_test=y_test,
               cols_to_scale=cols_to_scale,
               rf_importance=rf_importance, l1_kept=l1_kept)
    if save:
        if not add_time_features and not add_interactions:
            X_train_scaled.to_csv("X_train.csv", index=False)
            X_test_scaled.to_csv("X_test.csv", index=False)
            y_train.to_csv("y_train.csv", index=False)
            y_test.to_csv("y_test.csv", index=False)
            if diagnostics:
                rf_importance.to_csv("rf_feature_importance.csv")
                l1_kept.to_csv("lasso_selected_features.csv")
            print("\nSaved: X_train.csv, X_test.csv, y_train.csv, y_test.csv, "
                  "rf_feature_importance.csv, lasso_selected_features.csv")
        else:
            X_train_scaled.to_csv("X_train_v2.csv", index=False)
            X_test_scaled.to_csv("X_test_v2.csv", index=False)
            print("\nSaved: X_train_v2.csv, X_test_v2.csv (new names; originals untouched)")
    print("\nDone. Review the correlation flags and Lasso/RF agreement before finalizing "
          "the feature list you report in Chapter 6 of the write-up.")
    return out


if __name__ == '__main__':
    build_datasets(add_time_features=False, add_interactions=False,
                   diagnostics=True, save=True)