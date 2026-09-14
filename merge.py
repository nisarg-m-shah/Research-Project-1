"""
Project C — merge all 8 raw extraction tables into one position-level dataset.
Column names below match the ACTUAL exported CSVs (confirmed via .columns.tolist()).
"""

import pandas as pd

pd.set_option("display.width", 120)

# ---------------------------------------------------------------------------
# 1. Load raw tables
# ---------------------------------------------------------------------------
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

print("Raw row counts:")
for name, d in [
    ("mint", mint), ("open_events", open_events), ("open_delta", open_delta),
    ("close_events", close_events), ("collect", collect), ("pools", pools),
    ("swap_agg", swap_agg), ("transactions", transactions), ("prices", prices),
]:
    print(f"  {name}: {len(d):,} rows")

# ---------------------------------------------------------------------------
# 2. Sum open_events to one row per tokenId (it's currently the FIRST
#    IncreaseLiquidity only — rn=1 from the original query), then fold in
#    delta (top-up) rows on top, same pattern as tables 3/4.
# ---------------------------------------------------------------------------
open_base = open_events.groupby("tokenId").agg(
    open_evt_tx_hash=("evt_tx_hash", "first"),
    open_time=("evt_block_time", "min"),
    total_liquidity_added=("liquidity_added", "sum"),
    total_token0_deposited=("token0_deposited", "sum"),
    total_token1_deposited=("token1_deposited", "sum"),
).reset_index()

if not open_delta.empty:
    delta_summary = open_delta.groupby("tokenId").agg(
        delta_liquidity=("liquidity_added", "sum"),
        delta_token0=("token0_deposited", "sum"),
        delta_token1=("token1_deposited", "sum"),
    ).reset_index()

    open_full = open_base.merge(delta_summary, on="tokenId", how="left")
    for col in ["delta_liquidity", "delta_token0", "delta_token1"]:
        open_full[col] = open_full[col].fillna(0)

    open_full["total_liquidity_added"] += open_full["delta_liquidity"]
    open_full["total_token0_deposited"] += open_full["delta_token0"]
    open_full["total_token1_deposited"] += open_full["delta_token1"]
    open_full = open_full.drop(columns=["delta_liquidity", "delta_token0", "delta_token1"])
else:
    open_full = open_base.copy()

# ---------------------------------------------------------------------------
# 3. Core join: open (+ top-ups) <-> close, one row per closed position
# ---------------------------------------------------------------------------
df = open_full.merge(close_events, on="tokenId", how="inner")
print(f"\nAfter open<->close join: {len(df):,} rows")

# ---------------------------------------------------------------------------
# 4. Attach tick range from Mint (join on the open transaction hash)
# ---------------------------------------------------------------------------
mint_slim = mint[["evt_tx_hash", "pool_address", "tickLower", "tickUpper"]].rename(
    columns={"evt_tx_hash": "open_evt_tx_hash"}
)
df = df.merge(mint_slim, on="open_evt_tx_hash", how="left")
df["range_width_ticks"] = df["tickUpper"] - df["tickLower"]

# ---------------------------------------------------------------------------
# 5. Attach fees collected (collect.csv can have multiple rows per tokenId
#    if there were multiple recipients — sum to one row per tokenId)
# ---------------------------------------------------------------------------
collect_summary = collect.groupby("tokenId").agg(
    fees_collected_token0=("fees_collected_token0", "sum"),
    fees_collected_token1=("fees_collected_token1", "sum"),
    collect_event_count=("collect_event_count", "sum"),
).reset_index()
df = df.merge(collect_summary, on="tokenId", how="left")
for col in ["fees_collected_token0", "fees_collected_token1", "collect_event_count"]:
    df[col] = df[col].fillna(0)

# ---------------------------------------------------------------------------
# 6. Attach pool metadata (token identities, fee tier)
# ---------------------------------------------------------------------------
pools_slim = pools.rename(columns={"pool_address": "pool_address_meta"})
df = df.merge(
    pools_slim, left_on="pool_address", right_on="pool_address_meta", how="left"
).drop(columns=["pool_address_meta"])

# ---------------------------------------------------------------------------
# 7. Attach swap-agg-derived features: volume + price range DURING the
#    position's lifetime (between open_time and close_time)
# ---------------------------------------------------------------------------
swap_agg["swap_date"] = pd.to_datetime(swap_agg["swap_date"])
df["open_time"] = pd.to_datetime(df["open_time"])
df["close_time"] = pd.to_datetime(df["close_time"])

swap_features = []
for pool_addr, group in swap_agg.groupby("pool_address"):
    positions = df[df["pool_address"] == pool_addr]
    for _, pos in positions.iterrows():
        window = group[
            (group["swap_date"] >= pos["open_time"].floor("D"))
            & (group["swap_date"] <= pos["close_time"].ceil("D"))
        ]
        swap_features.append({
            "tokenId": pos["tokenId"],
            "lifetime_swap_count": window["swap_count"].sum(),
            "lifetime_token0_volume": window["token0_volume"].sum(),
            "lifetime_token1_volume": window["token1_volume"].sum(),
            "lifetime_min_tick": window["min_tick"].min() if not window.empty else None,
            "lifetime_max_tick": window["max_tick"].max() if not window.empty else None,
        })

swap_feat_df = pd.DataFrame(swap_features)
df = df.merge(swap_feat_df, on="tokenId", how="left")

df["went_out_of_range"] = (
    (df["lifetime_min_tick"] < df["tickLower"]) | (df["lifetime_max_tick"] > df["tickUpper"])
)

# ---------------------------------------------------------------------------
# 8. Attach transaction metadata (gas cost) for open and close tx
# ---------------------------------------------------------------------------
tx_slim = transactions.rename(columns={"hash": "tx_hash"})

open_tx = tx_slim.rename(columns={
    "tx_hash": "open_evt_tx_hash", "gas": "open_gas", "gas_price": "open_gas_price",
    "nonce": "open_nonce", "from_address": "open_wallet",
})[["open_evt_tx_hash", "open_gas", "open_gas_price", "open_nonce", "open_wallet"]]

close_tx = tx_slim.rename(columns={
    "tx_hash": "final_evt_tx_hash", "gas": "close_gas", "gas_price": "close_gas_price",
})[["final_evt_tx_hash", "close_gas", "close_gas_price"]]

df = df.merge(open_tx, on="open_evt_tx_hash", how="left")
df = df.merge(close_tx, on="final_evt_tx_hash", how="left")

# ---------------------------------------------------------------------------
# 9. Attach USD prices at open and close
# ---------------------------------------------------------------------------
prices["timestamp"] = pd.to_datetime(prices["timestamp"]).dt.floor("min")

df["open_time_floor"] = df["open_time"].dt.floor("min")
df["close_time_floor"] = df["close_time"].dt.floor("min")

df = df.merge(
    prices.rename(columns={"timestamp": "open_time_floor", "contract_address": "token0_address", "price": "token0_price_usd_at_open"}),
    on=["open_time_floor", "token0_address"], how="left",
)
df = df.merge(
    prices.rename(columns={"timestamp": "open_time_floor", "contract_address": "token1_address", "price": "token1_price_usd_at_open"}),
    on=["open_time_floor", "token1_address"], how="left",
)
df = df.merge(
    prices.rename(columns={"timestamp": "close_time_floor", "contract_address": "token0_address", "price": "token0_price_usd_at_close"}),
    on=["close_time_floor", "token0_address"], how="left",
)
df = df.merge(
    prices.rename(columns={"timestamp": "close_time_floor", "contract_address": "token1_address", "price": "token1_price_usd_at_close"}),
    on=["close_time_floor", "token1_address"], how="left",
)

# ---------------------------------------------------------------------------
# 10. Compute the label: net USD profitability
#     NOTE: decimals NOT yet applied — raw token amounts are in base units.
#     Divide by 10**decimals per token before trusting these USD figures.
#     USDC/USDT = 6, DAI/WETH = 18, WBTC = 8.
# ---------------------------------------------------------------------------
df["deposited_value_usd"] = (
    df["total_token0_deposited"] * df["token0_price_usd_at_open"]
    + df["total_token1_deposited"] * df["token1_price_usd_at_open"]
)
df["withdrawn_value_usd"] = (
    (df["total_token0_returned"] + df["fees_collected_token0"]) * df["token0_price_usd_at_close"]
    + (df["total_token1_returned"] + df["fees_collected_token1"]) * df["token1_price_usd_at_close"]
)
df["net_profit_usd"] = df["withdrawn_value_usd"] - df["deposited_value_usd"]
df["is_profitable"] = (df["net_profit_usd"] > 0).astype(int)

# ---------------------------------------------------------------------------
# 11. Flag leakage-risk columns (kept in file, excluded from features later)
# ---------------------------------------------------------------------------
leakage_cols = [
    "final_evt_tx_hash", "close_time", "close_block", "close_time_floor",
    "total_liquidity_removed", "total_token0_returned", "total_token1_returned",
    "decrease_event_count", "fees_collected_token0", "fees_collected_token1",
    "collect_event_count", "close_gas", "close_gas_price",
    "token0_price_usd_at_close", "token1_price_usd_at_close",
    "withdrawn_value_usd", "lifetime_swap_count", "lifetime_token0_volume",
    "lifetime_token1_volume", "lifetime_min_tick", "lifetime_max_tick",
    "went_out_of_range",
]
print(f"\nNOTE: {len(leakage_cols)} columns are close-time/outcome-dependent.")
print("They stay in merged_positions.csv but must be EXCLUDED from X when training:")
print(leakage_cols)

# ---------------------------------------------------------------------------
# 12. Save
# ---------------------------------------------------------------------------
print(f"\nFinal merged dataset: {len(df):,} rows, {df.shape[1]} columns")
df.to_csv("merged_positions.csv", index=False)
print("Saved to merged_positions.csv")