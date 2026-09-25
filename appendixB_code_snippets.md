# Appendix B — Code Extracts (leakage exclusion, time split, final training)

Monospace extracts for the report appendix, taken verbatim from the
repository scripts. Re-run the source scripts rather than treating these as
living code.

## B.1 Leakage exclusion (Block 1, block1prep.py)

All columns knowable only at or after position close are excluded from the
feature matrix `X`:

```python
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

exclude_all = LEAKAGE_COLS + ID_COLS + drop_raw_address_cols + [TARGET_COL]
feature_cols = [c for c in df.columns if c not in exclude_all]
```

## B.2 Chronological 80/20 train/test split (Block 1, block1prep.py)

No shuffling: this is chronological data, and a random split would let the
model train on positions that opened after some test positions closed.

```python
# STEP 1: Time-based train/test split (80/20 by open_time)
df = df.sort_values('open_time').reset_index(drop=True)
split_idx = int(len(df) * 0.8)
split_time = df.iloc[split_idx]['open_time']

train_df = df[df['open_time'] < split_time].reset_index(drop=True)
test_df  = df[df['open_time'] >= split_time].reset_index(drop=True)
```

## B.3 Final XGBoost training call (Block 3, block3hyperparametertuning.py)

Refit on the full chronological training split with the winning tuned
hyperparameters, evaluated on the untouched 20%-later test split:

```python
final_params = {   # RandomizedSearchCV winner, 60 trials x TimeSeriesSplit(5),
                   # cross-checked by Optuna TPE (60 trials) to within 0.0002 AUC
    'max_depth': 4, 'learning_rate': 0.02, 'n_estimators': 100,
    'subsample': 0.8, 'colsample_bytree': 0.6, 'min_child_weight': 7,
    'reg_alpha': 5, 'reg_lambda': 2,
}
final_model = XGBClassifier(**final_params, eval_metric='auc',
                            random_state=0, n_jobs=-1)
final_model.fit(X_train, y_train)

test_probs = final_model.predict_proba(X_test)[:, 1]   # threshold 0.56
```

## B.4 Chronological cross-validation design (Block 3, block3hyperparametertuning.py)

The tuning search uses the same no-shuffle discipline: `TimeSeriesSplit`
folds that always train on an earlier chronological block and validate on a
later one.

```python
tscv = TimeSeriesSplit(n_splits=5)          # X_train is already open_time-sorted
search = RandomizedSearchCV(
    base_model, param_distributions, n_iter=60, scoring='roc_auc',
    cv=tscv, random_state=0, n_jobs=-1, return_train_score=True,
)
```