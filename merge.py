import pandas as pd

# Load your existing full open-events export (already has one row per tokenId, first increase only)
open_df = pd.read_csv("open_events_raw.csv")

# Load the delta — only the extra top-up increases for tokenIds that had more than one
delta_df = pd.read_csv("open_events_topup.csv")

# Sanity check: every tokenId in delta should already exist in open_df
missing = set(delta_df["tokenId"]) - set(open_df["tokenId"])
if missing:
    print(f"Warning: {len(missing)} tokenIds in delta not found in open_df — investigate before merging")

# Combine both into one long table (first-increase rows + delta top-up rows)
combined = pd.concat([open_df, delta_df], ignore_index=True)

# Now sum liquidity/amounts per tokenId, keep the earliest tx as "open" reference
open_summary = combined.groupby("tokenId").agg(
    open_evt_tx_hash=("evt_tx_hash", "first"),   # first() after sorting below gives earliest
    open_time=("evt_block_time", "min"),
    total_liquidity_added=("liquidity_added", "sum"),
    total_token0_deposited=("token0_deposited", "sum"),
    total_token1_deposited=("token1_deposited", "sum"),
    increase_event_count=("evt_tx_hash", "count"),
).reset_index()

# Note: agg("first") depends on row order, so sort by time before grouping to guarantee correctness
combined_sorted = combined.sort_values(["tokenId", "evt_block_time"])
open_summary = combined_sorted.groupby("tokenId").agg(
    open_evt_tx_hash=("evt_tx_hash", "first"),
    open_time=("evt_block_time", "first"),
    total_liquidity_added=("liquidity_added", "sum"),
    total_token0_deposited=("token0_deposited", "sum"),
    total_token1_deposited=("token1_deposited", "sum"),
    increase_event_count=("evt_tx_hash", "count"),
).reset_index()

open_summary.to_csv("open_events.csv", index=False)
print(f"Final open events: {len(open_summary)} unique positions")
print(f"Positions with top-ups: {(open_summary['increase_event_count'] > 1).sum()}")