"""
Uniswap V3 LP Profitability Pipeline — Data Cleaning, Merging, Labeling
v5: fixes corrupted/overflow deposit amounts (mint_amount0/1 showing inf in
    describe()) that the earlier outlier filter didn't catch, and fills the
    small number of NaN token1_price_pct_change_24h values. Otherwise identical
    to v4 (fee tier as numeric, tick-spacing-normalized range width, log deposit,
    LP prior-experience count, decrease_event_count, range/recent-trading overlap).
"""

import pandas as pd
import numpy as np

mint = pd.read_csv("mint_data.csv", low_memory=False)
open_events = pd.read_csv("open_events_raw.csv")
open_delta = pd.read_csv("open_events_topup.csv")
close_events = pd.read_csv("close_events.csv")
collect = pd.read_csv("collect.csv")
pools = pd.read_csv("pool_metadata.csv")
swap_agg = pd.read_csv("swap_daily_aggregates.csv")
transactions = pd.read_csv("transactions.csv")              # still unused; kept for reference
prices_a = pd.read_csv("raw_prices_usdc_weth_wbtc.csv")
prices_b = pd.read_csv("raw_prices_dai_usdt.csv")
prices = pd.concat([prices_a, prices_b], ignore_index=True)

# ============================================================
# STEP 1: Fix dtypes
# ============================================================
mint['evt_block_time'] = pd.to_datetime(mint['evt_block_time'])
close_events['close_time'] = pd.to_datetime(close_events['close_time'])
prices['timestamp'] = pd.to_datetime(prices['timestamp'])
swap_agg['swap_date'] = pd.to_datetime(swap_agg['swap_date'])

# ============================================================
# STEP 2: Join mint (ticks + open_time) to open_events (tokenId)
# ============================================================
single_mint_txs = mint.groupby('evt_tx_hash').filter(lambda x: len(x) == 1)['evt_tx_hash'].unique()
single_mints = mint[mint['evt_tx_hash'].isin(single_mint_txs)]

clean_joins_df = single_mints.merge(
    open_events[['evt_tx_hash', 'tokenId']],
    on='evt_tx_hash', how='inner'
)
clean_joins_std = clean_joins_df[['evt_tx_hash', 'tokenId', 'tickLower', 'tickUpper',
                                    'mint_amount0', 'mint_amount1', 'liquidity_minted',
                                    'evt_block_time']]

unmatched_single = single_mints[~single_mints['evt_tx_hash'].isin(open_events['evt_tx_hash'])]

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
                'evt_block_time': mint_row['evt_block_time'],
            })
        else:
            still_unresolved_1.append(tx)

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
                'evt_block_time': mint_row['evt_block_time'],
            })
        else:
            still_unresolved_2.append(tx)

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
            'evt_block_time': mint_row['evt_block_time'],
        })
    else:
        unmatched_still_unresolved.append(mint_row['evt_tx_hash'])

all_resolved = pd.concat([
    clean_joins_std,
    pd.DataFrame(resolved_matches),
    pd.DataFrame(resolved_from_delta),
    pd.DataFrame(unmatched_resolved),
], ignore_index=True)

range_check = all_resolved.groupby('tokenId')[['tickLower', 'tickUpper']].nunique()
inconsistent_ids = range_check[(range_check['tickLower'] > 1) | (range_check['tickUpper'] > 1)].index.tolist()

positions = all_resolved.groupby('tokenId').agg({
    'tickLower': 'first', 'tickUpper': 'first',
    'mint_amount0': 'sum', 'mint_amount1': 'sum', 'liquidity_minted': 'sum',
    'evt_block_time': 'min',
}).reset_index()
positions = positions.rename(columns={'evt_block_time': 'open_time'})

positions_clean = positions[~positions['tokenId'].isin(inconsistent_ids)].reset_index(drop=True)

# ============================================================
# STEP 3: Join close events + fees collected
# ============================================================
collect_agg = collect.groupby('tokenId').agg({
    'fees_collected_token0': 'sum',
    'fees_collected_token1': 'sum',
    'collect_event_count': 'sum',
    'fee_recipient': 'first',   # wallet that owns/collects this position — used for LP experience feature
}).reset_index()

positions_full = positions_clean.merge(
    close_events[['tokenId', 'close_time', 'close_block', 'total_liquidity_removed',
                  'total_token0_returned', 'total_token1_returned', 'decrease_event_count']],
    on='tokenId', how='left'
)
positions_full = positions_full.merge(
    collect_agg[['tokenId', 'fees_collected_token0', 'fees_collected_token1', 'collect_event_count', 'fee_recipient']],
    on='tokenId', how='left'
)

positions_closed = positions_full[positions_full['close_time'].notna()].reset_index(drop=True)
positions_closed['fees_collected_token0'] = positions_closed['fees_collected_token0'].fillna(0)
positions_closed['fees_collected_token1'] = positions_closed['fees_collected_token1'].fillna(0)
positions_closed['close_time_rounded'] = positions_closed['close_time'].dt.round('min')
positions_closed['open_time_rounded'] = positions_closed['open_time'].dt.round('min')

# ============================================================
# STEP 4: Attach pool + token addresses
# ============================================================
tokenId_pool_map = all_resolved.merge(
    mint[['evt_tx_hash', 'pool_address']].drop_duplicates(),
    on='evt_tx_hash', how='left'
)[['tokenId', 'pool_address']].drop_duplicates()

dupe_pool_ids = tokenId_pool_map.groupby('tokenId')['pool_address'].nunique()
dupe_pool_ids = dupe_pool_ids[dupe_pool_ids > 1].index.tolist()

pos_with_pool = positions_closed[~positions_closed['tokenId'].isin(dupe_pool_ids)].merge(
    tokenId_pool_map[~tokenId_pool_map['tokenId'].isin(dupe_pool_ids)],
    on='tokenId', how='left'
)

pos_with_tokens = pos_with_pool.merge(
    pools[['pool_address', 'token0_address', 'token1_address', 'pool_fee_tier', 'tick_spacing']],
    on='pool_address', how='left'
)

# ============================================================
# STEP 5: Join prices at OPEN and CLOSE (drop positions outside price coverage)
# ============================================================
def join_price(df, time_col, suffix):
    out = df.merge(
        prices[['timestamp', 'contract_address', 'price']].rename(
            columns={'contract_address': 'token0_address', 'price': f'token0_price_{suffix}'}),
        left_on=[time_col, 'token0_address'], right_on=['timestamp', 'token0_address'], how='left'
    ).drop(columns=['timestamp'])
    out = out.merge(
        prices[['timestamp', 'contract_address', 'price']].rename(
            columns={'contract_address': 'token1_address', 'price': f'token1_price_{suffix}'}),
        left_on=[time_col, 'token1_address'], right_on=['timestamp', 'token1_address'], how='left'
    ).drop(columns=['timestamp'])
    return out

pos_priced = join_price(pos_with_tokens, 'close_time_rounded', 'close')
pos_priced = join_price(pos_priced, 'open_time_rounded', 'open')

print(f"[DEBUG] Rows before price-availability filter: {len(pos_priced)}")
print(f"[DEBUG] Missing close prices: {(pos_priced['token0_price_close'].isna() | pos_priced['token1_price_close'].isna()).sum()}")
print(f"[DEBUG] Missing open prices: {(pos_priced['token0_price_open'].isna() | pos_priced['token1_price_open'].isna()).sum()}")

pos_labeled_base = pos_priced[
    pos_priced['token0_price_close'].notna() & pos_priced['token1_price_close'].notna() &
    pos_priced['token0_price_open'].notna() & pos_priced['token1_price_open'].notna()
].reset_index(drop=True)

print(f"[DEBUG] Rows after price-availability filter: {len(pos_labeled_base)}")

# ============================================================
# STEP 6: Decimals conversion + outlier removal
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

# Outlier removal: drop rows with corrupted/anomalous on-chain values, 6+ orders of
# magnitude beyond realistic position sizes. Originally this only checked
# total_token0_returned_adj — describe() later revealed mint_amount0_adj and
# mint_amount1_adj can ALSO carry the same corruption (values up to ~1e31, and
# inf after downstream arithmetic), so both deposit-side columns are checked too.
before_outliers = len(pos_labeled_base)
pos_labeled_base = pos_labeled_base[
    (pos_labeled_base['total_token0_returned_adj'] <= 1e10) &
    (pos_labeled_base['mint_amount0_adj'] <= 1e10) &
    (pos_labeled_base['mint_amount1_adj'] <= 1e10)
].reset_index(drop=True)
print(f"[DEBUG] Rows after outlier removal (Step 6): {len(pos_labeled_base)} "
      f"(dropped {before_outliers - len(pos_labeled_base)})")

# ============================================================
# STEP 7: Profitability label
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

# Uniswap's Collect event bundles principal + fees together when collection happens
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

# Value deposited, priced at OPEN (separate from held_value_usd, which is priced at close)
pos_labeled_base['deposit_value_usd'] = (
    pos_labeled_base['mint_amount0_adj'] * pos_labeled_base['token0_price_open'] +
    pos_labeled_base['mint_amount1_adj'] * pos_labeled_base['token1_price_open']
)

# ============================================================
# STEP 8: Duration
# ============================================================
pos_labeled_base['open_time'] = pd.to_datetime(pos_labeled_base['open_time'])
pos_labeled_base['position_duration_hours'] = (
    pos_labeled_base['close_time'] - pos_labeled_base['open_time']
).dt.total_seconds() / 3600

# ============================================================
# STEP 8b: Fee tier as numeric (ordinal, not categorical), range width, log deposit
# ============================================================
# pool_fee_tier is stored in hundredths of a bip (e.g. 3000 = 0.30%) — convert to a
# real numeric rate so magnitude/order is preserved, instead of one-hot categories.
pos_labeled_base['fee_tier_pct'] = pos_labeled_base['pool_fee_tier'] / 1_000_000

# Raw tick width isn't comparable across pools with different tick_spacing —
# normalize by tick_spacing to get width in "spacing units," which IS comparable.
pos_labeled_base['range_width_ticks'] = pos_labeled_base['tickUpper'] - pos_labeled_base['tickLower']
pos_labeled_base['range_width_normalized'] = pos_labeled_base['range_width_ticks'] / pos_labeled_base['tick_spacing']

# Log-transform deposit size before normalization downstream — compresses the heavy right
# tail (a few whale positions) so scaling afterward doesn't leave most values crushed near 0.
# Now safe from inf since Step 6 filters out the corrupted deposit-amount rows.
pos_labeled_base['log_deposit_value_usd'] = np.log1p(pos_labeled_base['deposit_value_usd'])

# LP experience: how many prior CLOSED positions has this wallet run before this one opened.
# Computed causally (only counts positions that opened earlier) to avoid leakage.
pos_labeled_base = pos_labeled_base.sort_values('open_time').reset_index(drop=True)
pos_labeled_base['lp_prior_position_count'] = pos_labeled_base.groupby('fee_recipient').cumcount()

# ============================================================
# STEP 9: PRE-OPEN MARKET CONDITION FEATURES
# For each position, compute volatility/momentum for BOTH token0 and token1
# over the 24h window BEFORE open_time — no assumption about which one is
# "the volatile side." Also flag whether each token is a stablecoin.
# Also pull recent swap volume from swap_agg over the 3 days before open.
# ============================================================

LOOKBACK_HOURS = 24
print(f"[DEBUG] Rows entering Step 9 (before volatility features): {len(pos_labeled_base)}")

# Known stablecoins in our 5-token universe (USDC, DAI, USDT). WETH/WBTC are not.
STABLECOIN_ADDRESSES = {
    '0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48',  # USDC
    '0x6b175474e89094c44da98b954eedeac495271d0f',  # DAI
    '0xdac17f958d2ee523a2206206994597c13d831ec7',  # USDT
}

pos_labeled_base['token0_is_stablecoin'] = pos_labeled_base['token0_address'].isin(STABLECOIN_ADDRESSES).astype(int)
pos_labeled_base['token1_is_stablecoin'] = pos_labeled_base['token1_address'].isin(STABLECOIN_ADDRESSES).astype(int)

def compute_rolling_price_stats(token_addr, prices_df, lookback_hours=24):
    """Precompute rolling volatility stats for ONE token's full price history, vectorized."""
    tp = prices_df[prices_df['contract_address'] == token_addr][['timestamp', 'price']].sort_values('timestamp')
    tp = tp.set_index('timestamp')

    roll = tp['price'].rolling(f'{lookback_hours}h')
    tp['price_std_24h'] = roll.std()
    tp['price_min_24h'] = roll.min()
    tp['price_max_24h'] = roll.max()
    tp['price_mean_24h'] = roll.mean()
    tp['price_range_pct_24h'] = (tp['price_max_24h'] - tp['price_min_24h']) / tp['price_mean_24h']

    tp = tp.reset_index()
    return tp[['timestamp', 'price', 'price_std_24h', 'price_range_pct_24h']]

def attach_token_price_features(positions_df, token_col, prices_df, prefix):
    """For each unique token in token_col, precompute rolling stats once, then asof-merge onto positions."""
    positions_df = positions_df.sort_values('open_time')
    results = []

    for token_addr in positions_df[token_col].unique():
        rolling_stats = compute_rolling_price_stats(token_addr, prices_df)
        subset = positions_df[positions_df[token_col] == token_addr].sort_values('open_time')

        # nearest prior price snapshot (and its trailing 24h stats) at/just before open_time
        merged = pd.merge_asof(
            subset, rolling_stats,
            left_on='open_time', right_on='timestamp',
            direction='backward', tolerance=pd.Timedelta(hours=9)
        )
        # price ~24h before open, for pct_change calc
        price_24h_ago = pd.merge_asof(
            subset[['tokenId', 'open_time']].assign(lookup_time=subset['open_time'] - pd.Timedelta(hours=24)),
            rolling_stats[['timestamp', 'price']].rename(columns={'price': 'price_24h_ago'}),
            left_on='lookup_time', right_on='timestamp',
            direction='backward', tolerance=pd.Timedelta(hours=9)
        )[['tokenId', 'price_24h_ago']]

        merged = merged.merge(price_24h_ago, on='tokenId', how='left')
        merged[f'{prefix}_price_pct_change_24h'] = (
            (merged['price'] - merged['price_24h_ago']) / merged['price_24h_ago']
        )
        merged = merged.rename(columns={
            'price_std_24h': f'{prefix}_price_std_24h',
            'price_range_pct_24h': f'{prefix}_price_range_pct_24h',
        })
        results.append(merged[['tokenId', f'{prefix}_price_std_24h',
                                f'{prefix}_price_pct_change_24h', f'{prefix}_price_range_pct_24h']])

    return pd.concat(results, ignore_index=True)

token0_feats = attach_token_price_features(pos_labeled_base, 'token0_address', prices, 'token0')
token1_feats = attach_token_price_features(pos_labeled_base, 'token1_address', prices, 'token1')

pos_labeled_base = pos_labeled_base.merge(token0_feats, on='tokenId', how='left')
pos_labeled_base = pos_labeled_base.merge(token1_feats, on='tokenId', how='left')

print(f"Positions missing token0 pre-open price window: {pos_labeled_base['token0_price_std_24h'].isna().sum()}")
print(f"Positions missing token1 pre-open price window: {pos_labeled_base['token1_price_std_24h'].isna().sum()}")

# --- Recent swap volume/activity from swap_agg (3 days before open), vectorized ---
VOLUME_LOOKBACK_DAYS = 3

def attach_volume_features(positions_df, swap_df, lookback_days=3):
    results = []
    positions_df = positions_df.sort_values('open_time')

    for pool in positions_df['pool_address'].unique():
        pool_swaps = swap_df[swap_df['pool_address'] == pool].sort_values('swap_date').set_index('swap_date')
        roll = pool_swaps[['swap_count', 'token0_volume']].rolling(f'{lookback_days}D')
        pool_swaps['avg_daily_swap_count'] = roll['swap_count'].mean()
        pool_swaps['avg_daily_token0_volume'] = roll['token0_volume'].mean()
        pool_swaps = pool_swaps.reset_index()

        subset = positions_df[positions_df['pool_address'] == pool]
        merged = pd.merge_asof(
            subset[['tokenId', 'open_time']].sort_values('open_time'),
            pool_swaps[['swap_date', 'avg_daily_swap_count', 'avg_daily_token0_volume']],
            left_on='open_time', right_on='swap_date',
            direction='backward', tolerance=pd.Timedelta(days=lookback_days + 1)
        )
        results.append(merged[['tokenId', 'avg_daily_swap_count', 'avg_daily_token0_volume']])

    out = pd.concat(results, ignore_index=True)
    out = out.rename(columns={
        'avg_daily_swap_count': 'pre_open_avg_daily_swap_count_3d',
        'avg_daily_token0_volume': 'pre_open_avg_daily_token0_volume_3d',
    })
    return out

volume_features = attach_volume_features(pos_labeled_base, swap_agg)
pos_labeled_base = pos_labeled_base.merge(volume_features, on='tokenId', how='left')
pos_labeled_base['pre_open_avg_daily_swap_count_3d'] = pos_labeled_base['pre_open_avg_daily_swap_count_3d'].fillna(0)
pos_labeled_base['pre_open_avg_daily_token0_volume_3d'] = pos_labeled_base['pre_open_avg_daily_token0_volume_3d'].fillna(0)

# --- Did the LP's chosen range actually overlap where the pool was recently trading? ---
# Uses swap_agg's min_tick/max_tick over the 3 days before open to get the pool's recent
# trading range, then checks overlap against [tickLower, tickUpper]. A position whose range
# never touches recent trading activity is close to guaranteed near-zero fees.
def attach_range_overlap_features(positions_df, swap_df, lookback_days=3):
    results = []
    positions_df = positions_df.sort_values('open_time')

    for pool in positions_df['pool_address'].unique():
        pool_swaps = swap_df[swap_df['pool_address'] == pool].sort_values('swap_date').set_index('swap_date')
        roll = pool_swaps[['min_tick', 'max_tick']].rolling(f'{lookback_days}D')
        pool_swaps['recent_min_tick'] = roll['min_tick'].min()
        pool_swaps['recent_max_tick'] = roll['max_tick'].max()
        pool_swaps = pool_swaps.reset_index()

        subset = positions_df[positions_df['pool_address'] == pool][
            ['tokenId', 'open_time', 'tickLower', 'tickUpper']
        ].sort_values('open_time')

        merged = pd.merge_asof(
            subset, pool_swaps[['swap_date', 'recent_min_tick', 'recent_max_tick']],
            left_on='open_time', right_on='swap_date',
            direction='backward', tolerance=pd.Timedelta(days=lookback_days + 1)
        )

        # overlap = the position's range intersects the pool's recently-traded range
        merged['range_overlaps_recent_trading'] = (
            (merged['tickLower'] <= merged['recent_max_tick']) &
            (merged['tickUpper'] >= merged['recent_min_tick'])
        ).astype(int)

        results.append(merged[['tokenId', 'range_overlaps_recent_trading']])

    return pd.concat(results, ignore_index=True)

range_overlap_features = attach_range_overlap_features(pos_labeled_base, swap_agg)
pos_labeled_base = pos_labeled_base.merge(range_overlap_features, on='tokenId', how='left')
# no recent swap history in the window at all -> treat as no overlap (conservative default)
pos_labeled_base['range_overlaps_recent_trading'] = pos_labeled_base['range_overlaps_recent_trading'].fillna(0).astype(int)

# Drop positions where we couldn't compute a pre-open volatility window for either token
# (too early in the dataset's history to have 24h of prior price data)
pos_labeled_base = pos_labeled_base[
    pos_labeled_base['token0_price_std_24h'].notna() & pos_labeled_base['token1_price_std_24h'].notna()
].reset_index(drop=True)

# token1_price_pct_change_24h (and in principle token0's) can be NaN for a small number of
# rows near the very start of a token's price history, where a 24h-ago lookup itself has no
# match even though the immediate rolling stats do. Fill with 0 (no measurable momentum)
# rather than dropping — these are a tiny fraction of rows and momentum=0 is a reasonable
# neutral default, not a fabricated value.
n_missing_momentum = (
    pos_labeled_base['token0_price_pct_change_24h'].isna() | pos_labeled_base['token1_price_pct_change_24h'].isna()
).sum()
print(f"[DEBUG] Rows with missing momentum (filled with 0): {n_missing_momentum}")
pos_labeled_base['token0_price_pct_change_24h'] = pos_labeled_base['token0_price_pct_change_24h'].fillna(0)
pos_labeled_base['token1_price_pct_change_24h'] = pos_labeled_base['token1_price_pct_change_24h'].fillna(0)

# lp_prior_position_count can have a handful of NaNs if fee_recipient itself was missing
# for a row (e.g. no collect event at all) — treat as 0 prior experience, not unknown.
pos_labeled_base['lp_prior_position_count'] = pos_labeled_base['lp_prior_position_count'].fillna(0)

# ============================================================
# STEP 10: Final feature selection
# ============================================================
final_cols = [
    'tokenId',
    'open_time', 'close_time', 'position_duration_hours',
    'range_width_normalized',
    'pool_address', 'token0_address', 'token1_address', 'fee_tier_pct',
    'mint_amount0_adj', 'mint_amount1_adj', 'log_deposit_value_usd',
    'token0_price_open', 'token1_price_open',
    'token0_price_close', 'token1_price_close',
    'deposit_value_usd', 'held_value_usd', 'returned_value_usd',
    'true_fees_value_usd', 'impermanent_loss_usd',
    'token0_is_stablecoin', 'token1_is_stablecoin',
    'token0_price_std_24h', 'token0_price_pct_change_24h', 'token0_price_range_pct_24h',
    'token1_price_std_24h', 'token1_price_pct_change_24h', 'token1_price_range_pct_24h',
    'pre_open_avg_daily_swap_count_3d', 'pre_open_avg_daily_token0_volume_3d',
    'range_overlaps_recent_trading',
    'decrease_event_count', 'lp_prior_position_count',
    'profitable',
]
pos_labeled_base = pos_labeled_base[final_cols]

# Final sanity check: confirm no inf/-inf slipped through anywhere in the numeric columns
numeric_cols = pos_labeled_base.select_dtypes(include=[np.number]).columns
n_inf = np.isinf(pos_labeled_base[numeric_cols]).sum().sum()
print(f"[DEBUG] Remaining inf values across all numeric columns: {n_inf}")

print(f"Final labeled dataset with market-condition + structural features: {len(pos_labeled_base)} positions")
print(pos_labeled_base['profitable'].value_counts(normalize=True))

print("Saving final dataset to csv")
pos_labeled_base.to_csv("final_merged_dataset_v5.csv", index=False)