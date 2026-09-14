"""
Block 1 — Modeling Prep & Variable Selection
Uniswap V3 LP Profitability Classifier

Input:  final_merged_dataset_v5.csv (55,279 rows, 34 cols)
Output: X_train, X_test, y_train, y_test (scaled, leakage-safe)
        + a ranked variable-selection report

Run this BEFORE any model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

pd.set_option('display.width', 120)

df = pd.read_csv("final_merged_dataset_v5.csv")
df['open_time'] = pd.to_datetime(df['open_time'])
df['close_time'] = pd.to_datetime(df['close_time'])

print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
print(f"Date range: {df['open_time'].min()} -> {df['open_time'].max()}")
print(f"Class balance: {df['profitable'].value_counts(normalize=True).to_dict()}")

# ============================================================
# STEP 1: Time-based train/test split (80/20 by open_time)
# ============================================================
# No shuffling. This is chronological data — a random split would let the
# model train on positions that opened AFTER some test positions closed,
# which is a leak (the model would implicitly "see the future" of market
# regimes that hadn't happened yet at some training points' close time).
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

# ============================================================
# STEP 2: Leakage-safe feature list
# ============================================================
# Excluded — outcome-side / only knowable at or after close:
LEAKAGE_COLS = [
    'close_time',
    'token0_price_close', 'token1_price_close',
    'held_value_usd', 'returned_value_usd',
    'true_fees_value_usd', 'impermanent_loss_usd',
    'position_duration_hours',   # = close_time - open_time, not known at open
    'decrease_event_count',      # counts withdrawal events across the WHOLE life
]

# Excluded — identifiers / join keys, not model features:
ID_COLS = ['tokenId', 'open_time']

# token0_address / token1_address / pool_address: only 10 pools total, low
# cardinality, and pool identity carries real signal (liquidity depth, pair
# volatility profile) not fully captured by fee_tier_pct + stablecoin flags.
# One-hot encoding pool_address is cheap here — kept as a categorical feature.
TARGET_COL = 'profitable'

categorical_to_encode = ['pool_address']
drop_raw_address_cols = ['token0_address', 'token1_address']  # redundant with
                                                                 # stablecoin flags + pool_address

exclude_all = LEAKAGE_COLS + ID_COLS + drop_raw_address_cols + [TARGET_COL]
feature_cols = [c for c in df.columns if c not in exclude_all]

print(f"\nExcluded as leakage: {LEAKAGE_COLS}")
print(f"Excluded as ID/redundant: {ID_COLS + drop_raw_address_cols}")
print(f"Remaining feature columns ({len(feature_cols)}): {feature_cols}")

# ============================================================
# STEP 3: Build X / y, one-hot encode pool_address (fit categories on train)
# ============================================================
def build_xy(source_df, feature_cols, categorical_cols, pool_categories=None):
    X = source_df[feature_cols].copy()
    for c in categorical_cols:
        dummies = pd.get_dummies(X[c], prefix=c, drop_first=True)
        X = X.drop(columns=[c]).join(dummies)
    if pool_categories is not None:
        # align test set to train's dummy columns (in case a pool is missing in test)
        for col in pool_categories:
            if col not in X.columns:
                X[col] = 0
        X = X[pool_categories]
    y = source_df[TARGET_COL]
    return X, y

X_train, y_train = build_xy(train_df, feature_cols, categorical_to_encode)
train_dummy_cols = [c for c in X_train.columns]
X_test, y_test = build_xy(test_df, feature_cols, categorical_to_encode, pool_categories=train_dummy_cols)

print(f"\nX_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

# ============================================================
# STEP 4: Scale numeric features (fit on TRAIN only)
# ============================================================
numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
# don't scale one-hot dummies or binary flags
binary_like = [c for c in numeric_cols if X_train[c].dropna().isin([0, 1]).all()]
cols_to_scale = [c for c in numeric_cols if c not in binary_like]

scaler = StandardScaler()
X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()
X_train_scaled[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
X_test_scaled[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

print(f"Scaled {len(cols_to_scale)} continuous columns; left {len(binary_like)} binary/dummy columns unscaled")

# ============================================================
# STEP 5: Variable selection
# ============================================================
# (a) FILTER — drop near-zero variance, flag high pairwise correlation
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

# (b) EMBEDDED — Lasso logistic (scaled features required) + RF importance
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
rf = RandomForestClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5,
                             random_state=0, n_jobs=-1)
rf.fit(X_train, y_train)  # trees don't need scaling
rf_importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
print(rf_importance.head(20))

# ============================================================
# STEP 6: Save prepped artifacts for the modeling script
# ============================================================
X_train_scaled.to_csv("X_train.csv", index=False)
X_test_scaled.to_csv("X_test.csv", index=False)
y_train.to_csv("y_train.csv", index=False)
y_test.to_csv("y_test.csv", index=False)
rf_importance.to_csv("rf_feature_importance.csv")
l1_kept.to_csv("lasso_selected_features.csv")

print("\nSaved: X_train.csv, X_test.csv, y_train.csv, y_test.csv, "
      "rf_feature_importance.csv, lasso_selected_features.csv")
print("\nDone. Review the correlation flags and Lasso/RF agreement before finalizing "
      "the feature list you report in Chapter 6 of the write-up.")