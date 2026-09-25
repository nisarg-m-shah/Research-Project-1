"""
Appendix A — full data dictionary generator
Uniswap V3 LP Profitability Classifier

Expands Chapter 2.2 (raw tables) and Chapter 6.2 (model inputs) into a single
one-row-per-column dictionary covering all 35 raw columns of
final_merged_dataset_v5.csv plus the 27 final model-input columns (Block 1's
leakage-safe feature set, i.e. X columns after the Block 2/3
deposit_value_usd drop). Writes appendixA_data_dictionary.csv.
"""

import pandas as pd

df = pd.read_csv("final_merged_dataset_v5.csv")
dtypes = df.dtypes.astype(str).to_dict()

# ------------------------------------------------------------
# source identifies the raw extraction table a column originates from.
# "Derived" = computed in uniswap_pipeline_v5.py from one or more raw sources.
# knowable_at_open / role mirror Block 1's LEAKAGE_COLS / ID_COLS removals.
# ------------------------------------------------------------
DICT = [
    # id / join-key columns
    ('tokenId', 'Mint', 'Y', 'id', 'Numeric NFTSL position identifier; join key across Mint/Open/Close/Collect events. Dropped from X.'),
    ('open_time', 'Open events', 'N/A', 'id', 'ISO timestamp of the opening IncreaseLiquidity transaction; defines the prediction instant and the chronological split point. Dropped from X.'),
    ('pool_address', 'Pool metadata (PoolCreated)', 'Y', 'categorical-feature', 'Pool contract address (top-10-by-volume universe). One-hot encoded (drop_first) in Block 1 and kept as a feature: pool identity proxies liquidity depth and pair volatility not otherwise captured.'),
    ('token0_address', 'Pool metadata (PoolCreated)', 'Y', 'id', 'Address of the pool token0. Redundant with pool_address + stablecoin flags; dropped from X.'),
    ('token1_address', 'Pool metadata (PoolCreated)', 'Y', 'id', 'Address of the pool token1. Redundant with pool_address + stablecoin flags; dropped from X.'),
    # leakage columns (knowable only at/after close)
    ('close_time', 'Close events', 'N', 'leakage', 'ISO timestamp of the final DecreaseLiquidity event. Knowable at close only; excluded from X.'),
    ('position_duration_hours', 'Derived', 'N', 'leakage', 'close_time - open_time in hours. Not knowable at open; excluded from X.'),
    ('token0_price_close', 'Prices (Dune, minute)', 'N', 'leakage', 'USD price of token0 at the close minute. Excluded from X.'),
    ('token1_price_close', 'Prices (Dune, minute)', 'N', 'leakage', 'USD price of token1 at the close minute. Excluded from X.'),
    ('held_value_usd', 'Derived', 'N', 'leakage', 'Deposit amounts valued at close prices (what the LP would hold if they had just kept the tokens). Excluded from X.'),
    ('returned_value_usd', 'Derived (Close events)', 'N', 'leakage', 'Sum of all DecreaseLiquidity principal returned, valued at close prices. Excluded from X.'),
    ('true_fees_value_usd', 'Derived (Collect + Close)', 'N', 'leakage', 'Fees genuinely earned = fees collected minus principal returned (Uniswap bundles principal+fees in one Collect), valued at close; clipped at 0. Drives the label. Excluded from X.'),
    ('impermanent_loss_usd', 'Derived', 'N', 'leakage', 'hed_value_usd - returned_value_usd; the USD cost of concentrated range vs just holding. Drives the label. Excluded from X.'),
    ('decrease_event_count', 'Close events', 'N', 'leakage', 'Number of partial/full withdrawal events across the whole position life. Not knowable at open; excluded from X.'),
    # derived USD (block 2/3 redundant drop)
    ('deposit_value_usd', 'Derived', 'Y', 'redundant-dropped', 'Deposit value priced at open = mint_amount0_adj * token0_price_open + mint_amount1_adj * token1_price_open. Block 1 exposes it but Blocks 2/3 drop it as redundant with log_deposit_value_usd.'),
    # features
    ('range_width_normalized', 'Mint', 'Y', 'feature', 'tickUpper - tickLower normalized by tick_spacing; the headline feature — how wide a range the LP chose, comparable across pools.'),
    ('fee_tier_pct', 'Pool metadata (PoolCreated)', 'Y', 'feature', 'Pool fee tier as a rate (fee_tier/1,000,000); e.g. 0.003 = 0.30%.'),
    ('mint_amount0_adj', 'Open events', 'Y', 'feature', 'Decimal-adjusted token0 deposited at open (base units / 10^decimals).'),
    ('mint_amount1_adj', 'Open events', 'Y', 'feature', 'Decimal-adjusted token1 deposited at open.'),
    ('log_deposit_value_usd', 'Derived', 'Y', 'feature', 'log1p(deposit_value_usd); skew-corrected deposit size.'),
    ('token0_price_open', 'Prices (Dune, minute)', 'Y', 'feature', 'USD price of token0 at the open minute.'),
    ('token1_price_open', 'Prices (Dune, minute)', 'Y', 'feature', 'USD price of token1 at the open minute.'),
    ('token0_is_stablecoin', 'Derived (token registry)', 'Y', 'feature', '1 if token0 is USDC/DAI/USDT.'),
    ('token1_is_stablecoin', 'Derived (token registry)', 'Y', 'feature', '1 if token1 is USDC/DAI/USDT.'),
    ('token0_price_std_24h', 'Prices (Dune, minute)', 'Y', 'feature', 'Std dev of token0 USD price over the 24h window before open.'),
    ('token0_price_pct_change_24h', 'Prices (Dune, minute)', 'Y', 'feature', 'token0 24h momentum: (price_at_open - price_24h_before)/price_24h_before; NaN-filled with 0.'),
    ('token0_price_range_pct_24h', 'Prices (Dune, minute)', 'Y', 'feature', 'token0 24h range: (max-min)/mean over rolling 24h before open.'),
    ('token1_price_std_24h', 'Prices (Dune, minute)', 'Y', 'feature', 'Std dev of token1 USD price over the 24h window before open.'),
    ('token1_price_pct_change_24h', 'Prices (Dune, minute)', 'Y', 'feature', 'token1 24h momentum; NaN-filled with 0.'),
    ('token1_price_range_pct_24h', 'Prices (Dune, minute)', 'Y', 'feature', 'token1 24h range: (max-min)/mean over rolling 24h before open.'),
    ('pre_open_avg_daily_swap_count_3d', 'Swap daily aggregates', 'Y', 'feature', 'Average daily swap count in the pool over the 3 days before open.'),
    ('pre_open_avg_daily_token0_volume_3d', 'Swap daily aggregates', 'Y', 'feature', 'Average daily token0 swap volume over the 3 days before open, in raw token0 base units (consistent within pool; scale varies across pools by token0 decimals).'),
    ('range_overlaps_recent_trading', 'Derived (Swap + Mint)', 'Y', 'feature', '1 if [tickLower, tickUpper] overlaps the swap-observed tick range over the pre-open window.'),
    ('lp_prior_position_count', 'Derived (Mint fee_recipient)', 'Y', 'feature', 'Number of positions this wallet had already opened before this one; computed chronologically so it never looks ahead.'),
    # label
    ('profitable', 'Derived', 'N/A', 'label', 'Target: 1 if true_fees_value_usd > impermanent_loss_usd, else 0 (60.0% / 40.0% balanced).'),
]

rows = []
for col, source, knowable, role, desc in DICT:
    rows.append({
        'column': col, 'dtype_in_v5': dtypes.get(col, 'n/a'),
        'role': role, 'source': source, 'knowable_at_open': knowable, 'description': desc,
    })

dict_df = pd.DataFrame(rows)
dict_df.to_csv("appendixA_data_dictionary.csv", index=False)

# ------------------------------------------------------------
# the 27 final model-input columns (X after Block 2/3 deposit_value_usd drop)
# ------------------------------------------------------------
X = pd.read_csv("X_train.csv")
if 'deposit_value_usd' in X.columns:
    X = X.drop(columns=['deposit_value_usd'])
model_cols = list(X.columns)

print(f"Raw columns documented: {len(dict_df)}")
print(f"Final model-input columns: {len(model_cols)}")
print("\nModel-input columns not already in the raw dictionary (pool dummies):")
new_cols = [c for c in model_cols if c not in set(dict_df['column'])]
for c in new_cols:
    print(" ", c)

# append the 8 pool one-hot dummies to complete the 27
for c in new_cols:
    pool = c.replace('pool_address_', '')
    rows.append({
        'column': c, 'dtype_in_v5': 'float64 (one-hot)',
        'role': 'feature (one-hot)', 'source': 'Block 1 (from pool_address)',
        'knowable_at_open': 'Y',
        'description': f'One-hot indicator: position in pool {pool}. drop_first encoding leaves 8 of 10 pools as explicit dummies; the dropped reference pool is the omitted category.',
    })

full = pd.DataFrame(rows)
full.to_csv("appendixA_data_dictionary.csv", index=False)

raw_only = full[full['role'].isin(['id', 'leakage', 'label']) | (full['column'] == 'deposit_value_usd')]
feature_count = full[full['role'].str.startswith('feature')]
print(f"\nFinal dictionary rows (35 raw + 8 dummies): {len(full)}")
print(f"  raw columns: {len(dict_df)}  feature rows incl. dummies: {len(feature_count)}  label/leakage/id rows: {len(raw_only)}")
print(f"  model-input total documented: {len(model_cols)} "
      f"({len([c for c in model_cols if c in set(full['column'])])} covered by dictionary rows)")
print("\nSaved appendixA_data_dictionary.csv")