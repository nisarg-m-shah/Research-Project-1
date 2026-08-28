"""
Uniswap V3 LP Profitability Pipeline — Data Cleaning, Merging, Labeling
Assumes these raw tables are already loaded as pandas DataFrames:
  mint, open_events, open_delta, close_events, collect, pools, prices
"""

import pandas as pd

mint = pd.read_csv("mint_data.csv", low_memory=False)
open_events = pd.read_csv("open_events_raw.csv")
open_delta = pd.read_csv("open_events_topup.csv")
close_events = pd.read_csv("close_events.csv")
collect = pd.read_csv("collect.csv")
pools = pd.read_csv("pool_metadata.csv")
swap_agg = pd.read_csv("swap_daily_aggregates.csv")
transactions = pd.read_csv("transactions.csv")
prices_a = pd.read_csv("raw_prices_usdc_weth_wbtc.csv")
prices_b = pd.read_csv("raw_prices_dai_usdt.csv")
prices = pd.concat([prices_a, prices_b], ignore_index=True)

print("mint_data:",list(mint.columns))
print("open_events:",list(open_events.columns))
print("open_delta:",list(open_delta.columns))
print("close_events:",list(close_events.columns))
print("collect:",list(collect.columns))
print("pools:",list(pools.columns))
print("swap_agg:",list(swap_agg.columns))
print("transactions:",list(transactions.columns)) 
print("prices:",list(prices.columns))

# ============================================================
# STEP 1: Fix dtypes
# ============================================================
close_events['close_time'] = pd.to_datetime(close_events['close_time'])
prices['timestamp'] = pd.to_datetime(prices['timestamp'])

# ============================================================
# STEP 2: Join mint_data (ticks) to open_events (tokenId)
# ============================================================

# 2a. Single-mint transactions -> safe direct join
single_mint_txs = mint.groupby('evt_tx_hash').filter(lambda x: len(x) == 1)['evt_tx_hash'].unique()
single_mints = mint[mint['evt_tx_hash'].isin(single_mint_txs)]

clean_joins_df = single_mints.merge(
    open_events[['evt_tx_hash', 'tokenId']],
    on='evt_tx_hash', how='inner'
)
clean_joins_std = clean_joins_df[['evt_tx_hash', 'tokenId', 'tickLower', 'tickUpper',
                                    'mint_amount0', 'mint_amount1', 'liquidity_minted']]

# Single mints with NO match in open_events at all (these are single-tx top-ups)
unmatched_single = single_mints[~single_mints['evt_tx_hash'].isin(open_events['evt_tx_hash'])]

# 2b. Multi-mint transactions -> match by liquidity_added value (evt_tx_hash alone is ambiguous)
dupe_txs = mint.groupby('evt_tx_hash').filter(lambda x: len(x) > 1)['evt_tx_hash'].unique()

resolved_matches = []
still_unresolved_1 = []
for tx in dupe_txs:
    mints = mint[mint['evt_tx_hash'] == tx]
    opens = open_events[open_events['evt_tx_hash'] == tx]
    for _, mint_row in mints.iterrows():
        candidates = opens[opens['liquidity_added'] == mint_row['liquidity_minted']]
        if len(candidates) == 1:
            resolved_matches.append({
                'evt_tx_hash': tx, 'tokenId': candidates.iloc[0]['tokenId'],
                'tickLower': mint_row['tickLower'], 'tickUpper': mint_row['tickUpper'],
                'mint_amount0': mint_row['mint_amount0'], 'mint_amount1': mint_row['mint_amount1'],
                'liquidity_minted': mint_row['liquidity_minted'],
            })
        else:
            still_unresolved_1.append(tx)

# 2c. Leftovers from 2b -> try matching against open_delta (top-ups within multi-mint tx)
resolved_from_delta = []
still_unresolved_2 = []
for tx in set(still_unresolved_1):
    mints = mint[mint['evt_tx_hash'] == tx]
    for _, mint_row in mints.iterrows():
        candidates = open_delta[open_delta['liquidity_added'] == mint_row['liquidity_minted']]
        if len(candidates) == 1:
            resolved_from_delta.append({
                'evt_tx_hash': tx, 'tokenId': candidates.iloc[0]['tokenId'],
                'tickLower': mint_row['tickLower'], 'tickUpper': mint_row['tickUpper'],
                'mint_amount0': mint_row['mint_amount0'], 'mint_amount1': mint_row['mint_amount1'],
                'liquidity_minted': mint_row['liquidity_minted'],
            })
        else:
            still_unresolved_2.append(tx)

# 2d. Single-mint txs with no open_events match -> match against open_delta (single-tx top-ups)
unmatched_resolved = []
unmatched_still_unresolved = []
for _, mint_row in unmatched_single.iterrows():
    candidates = open_delta[open_delta['liquidity_added'] == mint_row['liquidity_minted']]
    if len(candidates) == 1:
        unmatched_resolved.append({
            'evt_tx_hash': mint_row['evt_tx_hash'], 'tokenId': candidates.iloc[0]['tokenId'],
            'tickLower': mint_row['tickLower'], 'tickUpper': mint_row['tickUpper'],
            'mint_amount0': mint_row['mint_amount0'], 'mint_amount1': mint_row['mint_amount1'],
            'liquidity_minted': mint_row['liquidity_minted'],
        })
    else:
        unmatched_still_unresolved.append(mint_row['evt_tx_hash'])

# 2e. Combine all resolved mint-level rows
all_resolved = pd.concat([
    clean_joins_std,
    pd.DataFrame(resolved_matches),
    pd.DataFrame(resolved_from_delta),
    pd.DataFrame(unmatched_resolved),
], ignore_index=True)

# 2f. Aggregate to one row per tokenId (top-ups share the same tick range -> sum amounts)
range_check = all_resolved.groupby('tokenId')[['tickLower', 'tickUpper']].nunique()
inconsistent_ids = range_check[(range_check['tickLower'] > 1) | (range_check['tickUpper'] > 1)].index.tolist()

positions = all_resolved.groupby('tokenId').agg({
    'tickLower': 'first', 'tickUpper': 'first',
    'mint_amount0': 'sum', 'mint_amount1': 'sum', 'liquidity_minted': 'sum',
}).reset_index()

positions_clean = positions[~positions['tokenId'].isin(inconsistent_ids)].reset_index(drop=True)

# ============================================================
# STEP 3: Join close events + fees collected
# ============================================================
collect_agg = collect.groupby('tokenId').agg({
    'fees_collected_token0': 'sum',
    'fees_collected_token1': 'sum',
    'collect_event_count': 'sum',
}).reset_index()

positions_full = positions_clean.merge(
    close_events[['tokenId', 'close_time', 'close_block', 'total_liquidity_removed',
                  'total_token0_returned', 'total_token1_returned']],
    on='tokenId', how='left'
)
positions_full = positions_full.merge(
    collect_agg[['tokenId', 'fees_collected_token0', 'fees_collected_token1', 'collect_event_count']],
    on='tokenId', how='left'
)

# Keep only closed positions (open positions have no realized fees/IL outcome)
positions_closed = positions_full[positions_full['close_time'].notna()].reset_index(drop=True)
positions_closed['fees_collected_token0'] = positions_closed['fees_collected_token0'].fillna(0)
positions_closed['fees_collected_token1'] = positions_closed['fees_collected_token1'].fillna(0)
positions_closed['close_time_rounded'] = positions_closed['close_time'].dt.round('min')

# ============================================================
# STEP 4: Attach pool + token addresses
# ============================================================
tokenId_pool_map = all_resolved.merge(
    mint[['evt_tx_hash', 'pool_address']].drop_duplicates(),
    on='evt_tx_hash', how='left'
)[['tokenId', 'pool_address']].drop_duplicates()

# Drop the rare tokenIds that map to more than one pool_address
dupe_pool_ids = tokenId_pool_map.groupby('tokenId')['pool_address'].nunique()
dupe_pool_ids = dupe_pool_ids[dupe_pool_ids > 1].index.tolist()

pos_with_pool = positions_closed[~positions_closed['tokenId'].isin(dupe_pool_ids)].merge(
    tokenId_pool_map[~tokenId_pool_map['tokenId'].isin(dupe_pool_ids)],
    on='tokenId', how='left'
)

pos_with_tokens = pos_with_pool.merge(
    pools[['pool_address', 'token0_address', 'token1_address', 'pool_fee_tier']],
    on='pool_address', how='left'
)
pos_with_tokens['close_time_rounded'] = pos_with_tokens['close_time'].dt.round('min')

# ============================================================
# STEP 5: Join prices at close time (drop positions outside price coverage)
# ============================================================
pos_priced = pos_with_tokens.merge(
    prices[['timestamp', 'contract_address', 'price']].rename(
        columns={'contract_address': 'token0_address', 'price': 'token0_price_close'}),
    left_on=['close_time_rounded', 'token0_address'],
    right_on=['timestamp', 'token0_address'], how='left'
)
pos_priced = pos_priced.merge(
    prices[['timestamp', 'contract_address', 'price']].rename(
        columns={'contract_address': 'token1_address', 'price': 'token1_price_close'}),
    left_on=['close_time_rounded', 'token1_address'],
    right_on=['timestamp', 'token1_address'], how='left'
)

pos_labeled_base = pos_priced[
    pos_priced['token0_price_close'].notna() & pos_priced['token1_price_close'].notna()
].reset_index(drop=True)

# ============================================================
# STEP 6: Apply decimals conversion to all raw amount columns
# ============================================================
decimals_map = prices[['contract_address', 'decimals']].drop_duplicates() \
    .set_index('contract_address')['decimals'].to_dict()

pos_labeled_base['token0_decimals'] = pos_labeled_base['token0_address'].map(decimals_map)
pos_labeled_base['token1_decimals'] = pos_labeled_base['token1_address'].map(decimals_map)

raw_amount_cols = {
    'mint_amount0': 'token0_decimals', 'mint_amount1': 'token1_decimals',
    'total_token0_returned': 'token0_decimals', 'total_token1_returned': 'token1_decimals',
    'fees_collected_token0': 'token0_decimals', 'fees_collected_token1': 'token1_decimals',
}
for col, dec_col in raw_amount_cols.items():
    pos_labeled_base[col + '_adj'] = pos_labeled_base[col].astype(float) / (10 ** pos_labeled_base[dec_col])

# Drop rows with corrupted/anomalous on-chain values (extreme outliers, ~6+ orders of magnitude beyond p99.9)
pos_labeled_base = pos_labeled_base[pos_labeled_base['total_token0_returned_adj'] <= 1e10].reset_index(drop=True)

# ============================================================
# STEP 7: Build the profitability label
# ============================================================
pos_labeled_base['returned_value_usd'] = (
    pos_labeled_base['total_token0_returned_adj'] * pos_labeled_base['token0_price_close'] +
    pos_labeled_base['total_token1_returned_adj'] * pos_labeled_base['token1_price_close']
)
pos_labeled_base['held_value_usd'] = (
    pos_labeled_base['mint_amount0_adj'] * pos_labeled_base['token0_price_close'] +
    pos_labeled_base['mint_amount1_adj'] * pos_labeled_base['token1_price_close']
)
pos_labeled_base['impermanent_loss_usd'] = (
    pos_labeled_base['held_value_usd'] - pos_labeled_base['returned_value_usd']
)

# NOTE: Uniswap's Collect event bundles principal + fees together when collection happens
# alongside/after a full withdrawal. True fees = collected amount minus the principal
# already accounted for via DecreaseLiquidity (total_token0/1_returned).
pos_labeled_base['true_fees_token0_adj'] = (
    pos_labeled_base['fees_collected_token0_adj'] - pos_labeled_base['total_token0_returned_adj']
).clip(lower=0)
pos_labeled_base['true_fees_token1_adj'] = (
    pos_labeled_base['fees_collected_token1_adj'] - pos_labeled_base['total_token1_returned_adj']
).clip(lower=0)

pos_labeled_base['true_fees_value_usd'] = (
    pos_labeled_base['true_fees_token0_adj'] * pos_labeled_base['token0_price_close'] +
    pos_labeled_base['true_fees_token1_adj'] * pos_labeled_base['token1_price_close']
)

pos_labeled_base['profitable'] = (
    pos_labeled_base['true_fees_value_usd'] > pos_labeled_base['impermanent_loss_usd']
).astype(int)

print(f"Final labeled dataset: {len(pos_labeled_base)} positions")
print(pos_labeled_base['profitable'].value_counts(normalize=True))

print("Saving Final Merged Dataset to csv")
pos_labeled_base.to_csv("final_merged_dataset.csv")
