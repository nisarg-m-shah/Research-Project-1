# Project C — Uniswap V3 LP Position Profitability Prediction
## Full Context Document (as of this point in development)

---

## 1. What This Project Is

**One-line pitch:** Predict, at the moment a Uniswap V3 liquidity position is opened, whether it will end up profitable or not — a binary classification problem built on real on-chain data.

**Context/purpose:**
- This is Nash's **MSc Big Data Analytics final project** at St. Xavier's College, Mumbai.
- It is also intended as a **portfolio/resume piece**, specifically targeting on-chain data analyst roles.
- It was chosen over other candidate Web3 project ideas because it had: a genuine ML need (not a trivial rule-based problem), no rare-event/class-imbalance issue, and a clean, well-defined binary label — unlike alternatives considered which had rarer events or messier labels.

**The core problem it solves:**
Liquidity providers (LPs) on Uniswap V3 choose a price range to concentrate their capital in. This is fundamentally different from V2 (which spreads liquidity across the whole price curve) — V3 lets you concentrate liquidity in a chosen band, earning more fees per dollar *if* the price stays in that band, but exposing you to impermanent loss and "going out of range" (earning zero fees) if price moves outside it. Choosing that range well is the single biggest driver of whether a position ends up profitable. Most LPs currently choose ranges somewhat blindly or by rule of thumb. This project asks: **can we predict profitability at open time, using only information available at open time** (chosen range, pool characteristics, market conditions at that moment) — which would let an LP evaluate a range choice *before* committing capital.

**Why this is a good project (not just a data-plumbing exercise):**
- Real, meaningful ML task — not artificially constructed.
- Binary label, well-balanced (not a rare-event problem).
- Ties into Nash's stated interest in on-chain data analyst roles — directly relevant portfolio piece.
- Non-trivial data engineering (multi-table joins across 8 raw on-chain event tables, careful handling of partial withdrawals/top-ups, avoiding data leakage) — good story to tell in interviews/resume, not just the final model.

---

## 2. The Original Vision (IMPORTANT — do not lose this)

This is the **best, fullest version** of the project as originally scoped, before real-world data constraints (Dune credit limits) forced practical compromises. Anyone continuing this work should treat this as the "north star" and only deviate from it where explicitly noted in Section 3.

**Original scope:**
- **All Uniswap V3 pools** (not just a top-N subset) — full universe of pools since V3 launched.
- **Full historical time range**: 2021-05-01 (V3 launch) through the present (dynamically, "today").
- **Target: 100,000+ closed LP positions** as the training set, aiming for **50+ raw columns** in the final merged dataset (a constraint from the course/project requirements).
- **8 source tables**, each capturing a different aspect of an LP position's lifecycle:
  1. `IncreaseLiquidity` (NFT Position Manager) — position open / top-up events
  2. `DecreaseLiquidity` (NFT Position Manager) — position close / partial withdrawal events
  3. `Collect` (NFT Position Manager) — fees collected
  4. `Mint` (pool-level event) — the tick range (`tickLower`/`tickUpper`) chosen for the position — **this was flagged early on as the single most important feature**, since "price range chosen" is the headline driver of LP profitability, and it does NOT live on the NFT Position Manager events at all, only on the pool-level `Mint` event, joined via transaction hash.
  5. `Factory_evt_PoolCreated` — pool metadata (token0/token1 addresses, fee tier)
  6. `Pair_evt_Swap` — swap activity (aggregated, not raw, due to volume) — used to derive trading volume and whether price moved out of the chosen range during the position's life
  7. `ethereum.transactions` — gas costs for open/close transactions, wallet nonce (as a proxy for LP experience/sophistication)
  8. `prices.minute` — USD-denominated token prices at open and close, needed to compute actual dollar profitability (not just token-unit changes)

**Original Y (label) definition:**
Binary: was the position profitable or not, defined as:
```
net_profit_usd = (value withdrawn at close, in USD, including fees collected)
                − (value deposited at open, in USD)
is_profitable = net_profit_usd > 0
```
This is a genuine economic profitability measure — not just impermanent loss, but full LP economics including fees earned (the reward side) minus IL (the cost side) minus gas.

**Key modeling principle established early and still valid:** the model must only use features **known at the time the position opens** (chosen range, pool state, initial deposit amounts, wallet features, market conditions at open). Any field only knowable at close time (close price, fees eventually collected, whether it went out of range over its life, close-time gas) is fine to use for **constructing the label**, but must be **excluded from the feature set X** used to train the model — otherwise the model would be leaking information from the future into a "predict at open time" framing. This was flagged as the single most important structural risk in the project's initial critique, alongside the missing tickLower/tickUpper problem.

**Known modeling nuances flagged for the modeling phase (not yet addressed, future work):**
- V3 impermanent loss is **range-dependent**, not the same simple constant-product formula as V2 — a position that goes out-of-range behaves very differently (single-asset exposure, no further fee accrual) from one that stays in-range. The IL/profitability calc should eventually account for this, not just apply the naive V2 IL formula.
- **Survivorship bias**: only closed positions are used (open positions at data-pull time are excluded by design, since we don't know their eventual outcome). This could skew the training set toward shorter-lived or abandoned positions, since long-duration well-managed positions are more likely to still be open at any given snapshot. Worth mentioning as a limitation in the final writeup.
- The 50+ raw column requirement is partly a course-compliance artifact (some columns, like tx gas fields, aren't strong economic predictors on their own) — worth being upfront about this if asked, rather than implying every column is a hand-picked strong predictor.

---

## 3. What Changed, and Why (Real-World Constraints)

The original full-universe, full-history vision hit a hard data-access constraint: **Dune Analytics' free-tier API datapoint quota (~2,500 "credits" per account per month)**, where a datapoint ≈ rows × columns of any API-read result. The very first extraction attempt (Table 1 alone, full pool universe, full date range) returned **1,330,394 rows × 10 columns ≈ 13.3M datapoints** — over 5x a single month's entire quota for one table alone, before touching the other 7.

**Key realization during the credit-crisis debugging:** actual Dune datapoint consumption ran at roughly **35-50% of the naive rows×cols estimate** in practice (confirmed repeatedly across multiple tables) — the formula isn't purely linear in cells; smaller data types (ints, addresses) apparently cost less than the worst-case assumption. This meant scoping decisions were made somewhat conservatively (erring toward smaller pulls) relative to what individual accounts could actually have handled, but it also means: **if this project needs to be scaled up later (more pools, more history) — it's very likely more headroom exists than the naive math suggests, worth testing incrementally rather than assuming the original full-universe scope is permanently out of reach.**

**Compromises made, in order of when they were decided:**

1. **Time range narrowed**: from "2021-05-01 → today" down to **2021-05-01 → 2024-01-01** (roughly 2.7 years instead of the full ~5.5 years to present). Rationale: positions opened very recently mostly haven't closed yet and are unusable as labeled training rows anyway, so the practical upper bound was never really "today" — but 2024-01-01 as a fixed cutoff was chosen mainly for credit reasons, and could be pushed later if quota allows.

2. **Pool universe narrowed**: from "all Uniswap V3 pools" down to the **top 10 pools by swap volume/count**. This was decided after testing top-30 (692K Mint rows) and top-20 (292K rows) still didn't fit budget; top-10 (211K Mint rows) was the version actually extracted. Rationale given at the time: pool activity is heavily Pareto-distributed, so top-10 pools likely still captures the large majority of economically meaningful activity, and cuts the long tail of near-dead/wash-traded pools that would mostly add noise. **This is the biggest deviation from the original vision** — 10 pools is a much narrower universe than "all pools," and if there's ever room to expand this (e.g. top 20-30, or eventually all pools), it would make the final dataset more representative and the project's claims more defensible.

3. **Row count actually achieved** (top-10 pools, 2021-05-01 to 2024-01-01): **~154,761 closed positions** (after summing partial withdrawals correctly — see below) — this actually **exceeds** the original 100,000+ position target, despite the narrower pool/time scope. Good news: the reduced scope didn't compromise the core "enough training data" goal.

4. **Column count**: still landed at the target of **50+ raw columns** across the merged dataset (8 tables), satisfying the course requirement — this was never actually compromised.

5. **A real data-quality bug was caught and fixed mid-extraction**: the first version of the Close events (Table 3) and Collect (Table 4) queries used `row_number() ... rn = 1` to take only the **last** decrease/collect event per position — discarding the amounts from any earlier **partial withdrawals** or **multiple fee collections**. About **8% of positions** (13,592 out of 171,707) had more than one withdrawal event, meaning this bug would have silently understated `total_token0/1_returned` for ~8% of the dataset, corrupting the profitability label for those rows. **Fixed by switching to `GROUP BY tokenId` + `SUM()`** instead of taking the last row — this correctly captures total value returned/collected across all partial events, plus adds `decrease_event_count`/`collect_event_count` as legitimate bonus features (number of partial withdrawals or collections is itself a signal of LP behavior/sophistication). **This fix is already applied in the final extracted data** — not a pending task.

6. **Transactions table (Table 7) moved off Dune entirely, onto Google BigQuery**: Dune-side, the full transactions pull (377,031 unique open+close tx hashes × 9 columns) alone would have been ~3.4M datapoints — bigger than any single account's budget. BigQuery's public dataset `bigquery-public-data.crypto_ethereum.transactions` gave free access (1TB/month processing quota, not row×column metered like Dune), so this table was extracted there instead via an uploaded hash-list join. This is a **permanent architectural difference** from the original single-source (Dune-only) vision, not a temporary workaround — worth documenting clearly as such.

7. **Multiple Dune accounts used** to spread the remaining 7 tables' extraction across separate 2,500-credit budgets, since even after all the scope-narrowing, total real demand still exceeded one account's monthly allowance. Final successful extraction was spread across at least 4 different Dune accounts.

8. **Prices table (Table 8) further optimized**: rather than pulling `prices.minute` across the full 2021-2024 date range for all 5 tokens (which alone was 7,020,000 rows — a separate, huge cost), the approach was refined to pull prices **only at the exact minute-timestamps needed** (i.e., only at each position's actual open_time and close_time, uploaded as a 244,387-row timestamp list and joined against `prices.minute`), cutting this to **1,183,170 rows** — an ~83% reduction. This was further split by token (USDC+WETH+WBTC vs. DAI+USDT) across two accounts to fit budget, giving 709,902 + 473,268 rows respectively.

**Net honest assessment:** the core econometric idea, the label definition, the 8-table structure, and the critical tickLower/tickUpper fix are all intact and match the original vision. The main compromises are **pool universe (10 instead of "all") and time range (2021-2024 instead of "to present")** — both purely due to Dune's free-tier constraints, not because they were judged unnecessary. If resource constraints ease (paid Dune tier, more free accounts, more time), **expanding pool count and/or extending the time range toward the present are the most valuable next upgrades to the dataset**, in that order.

---

## 4. Data Extraction — What's Actually Been Done

**Final scope used for extraction:**
- Pools: top 10 Uniswap V3 pools (addresses listed below)
- Date range: 2021-05-01 to 2024-01-01
- NFT Position Manager contract: `0xC36442b4A4522E871399CD717aBDD847Ab11FE88`

**Top 10 pool addresses used:**
```
0x88e6a0c2ddd26feeb64f039a2c41296fcb3f5640
0xc7bbec68d12a0d1830360f8ec58fa599ba1b0e9b
0xe0554a476a092703abdb3ef35c80e0d76d32939f
0x11b815efb8f581194ae79006d24e0d814b7697f6
0x4585fe77225b41b697c938b018e2ac67ac5a20c0
0x60594a405d53811d3bc4766596efd80fd545a270
0x3416cf6c708da44db2624d63ea0aaef7113527c6
0x4e68ccd3e89f51c3074ca5072bbac773960dfa36
0x56534741cd8b152df6d48adf7ac51f75169a83b2
0x8ad599c3a0ff1de082011efddc58f1908eb6e6d8
```

**Token addresses involved (via pool metadata):**
```
USDC: 0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48
WETH: 0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2
WBTC: 0x2260fac5e5542a773aa44fbcfedf7c193bc2c599
DAI:  0x6b175474e89094c44da98b954eedeac495271d0f
USDT: 0xdac17f958d2ee523a2206206994597c13d831ec7
```

**All 8 tables extracted, final row counts, and the CONFIRMED CORRECT filenames to use going forward:**

| # | Table | Source | Final row count | Confirmed correct file |
|---|---|---|---|---|
| 1 | Mint (tick range) | Dune | 211,196 | `mint_data.csv` |
| 2 | Open events (increase liquidity, incl. top-ups summed in) | Dune | 171,707 | `open_events.csv` |
| 3 | Close events (decrease liquidity, all partial withdrawals summed) | Dune | 154,761 | `close_events.csv` |
| 4 | Collect (fees, summed across all collect events) | Dune | 158,193 | `collect.csv` |
| 5 | Pool metadata | Dune | 10 | `pool_metadata.csv` |
| 6 | Swap daily aggregates | Dune | 7,812 | `swap_daily_aggregates.csv` |
| 7 | Transactions (gas costs, wallet/nonce) | **BigQuery** | 363,826 | `transactions.csv` |
| 8a | Prices — USDC/WETH/WBTC | Dune | 709,902 | `raw_prices_usdc_weth_wbtc.csv` |
| 8b | Prices — DAI/USDT | Dune | 473,268 | `raw_prices_dai_usdt.csv` |

**Files confirmed present in the working directory but STALE/SUPERSEDED — do NOT use these, safe to delete:**
- `open_events_raw.csv` + `open_events_topup.csv` — superseded by `open_events.csv`, which already has the top-ups correctly folded in via `GROUP BY tokenId SUM()`.
- `transaction_metadata.csv` — an earlier, different Dune-based attempt at the Transactions table (9 columns, different schema) before the BigQuery approach was adopted. Superseded by `transactions.csv`.
- `raw_prices_daily.csv` — empty/leftover from the very first (abandoned) full-date-range Prices attempt (7,020,000 rows, before the timestamp-filtered approach was adopted).
- `tx_hashes_for_bigquery.csv`, `price_timestamps.csv` — intermediate helper files (hash list for BigQuery join, timestamp list for Dune Prices join) — not model inputs themselves, just process artifacts. Keep for reference/reproducibility but don't feed into the merge.

**Column schemas of the 8 confirmed-correct files** (as actually returned, verified via `.columns.tolist()`):
- `mint_data.csv`: `evt_tx_hash, evt_block_time, pool_address, tickLower, tickUpper, liquidity_minted, mint_amount0, mint_amount1`
- `open_events.csv`: `tokenId, open_evt_tx_hash, open_time, total_liquidity_added, total_token0_deposited, total_token1_deposited, increase_event_count`
- `close_events.csv`: `tokenId, final_evt_tx_hash, close_time, close_block, total_liquidity_removed, total_token0_returned, total_token1_returned, decrease_event_count`
- `collect.csv`: `tokenId, fee_recipient, fees_collected_token0, fees_collected_token1, collect_event_count`
- `pool_metadata.csv`: `factory_address, token0_address, token1_address, pool_fee_tier, tick_spacing, pool_address`
- `swap_daily_aggregates.csv`: `pool_address, swap_date, swap_count, token0_volume, token1_volume, min_tick, max_tick, avg_sqrt_price`
- `transactions.csv`: `block_timestamp, hash, from_address, gas, gas_price, nonce, receipt_status` (7 columns — trimmed from a fuller candidate set; `type`/`block_date` deliberately dropped as redundant/low-value, `nonce` deliberately kept as an LP-experience proxy)
- `raw_prices_usdc_weth_wbtc.csv` / `raw_prices_dai_usdt.csv`: `timestamp, contract_address, price, decimals`

---

## 5. Extraction Infrastructure Notes (useful if more data is ever pulled)

- **Dune free-tier quota**: ~2,500 datapoints/month per account (datapoint ≈ rows × columns of an API-read result, though real consumption ran ~35-50% of this naive estimate in practice). Query *execution* on the website (dune.com UI) is free; only reading results via the API consumes quota. **Always execute on the website first, then use the API purely to fetch the already-cached result** — this was the pattern used throughout and avoided ever touching a separate "execution credits" pool.
- **Multiple free accounts were used** to spread extraction across separate quota pools — at least 4 accounts total by the end. One account was lost/inaccessible partway through and had to be recreated for the affected query.
- **BigQuery**: free tier gives 1TB/month query *processing* (not row×column metered). Public dataset `bigquery-public-data.crypto_ethereum` has raw transactions/logs/blocks but **no ready-made USD price table** — that's why Prices stayed on Dune. Watch bytes-processed carefully: the Transactions pull alone scanned ~508GB (about half a month's free allowance) because the underlying public table is enormous (all of Ethereum history) even though the actual matched result was small. If more BigQuery pulls are needed later, budget-check remaining processing quota first.
- **Python extraction pattern used** (via `dune-client` library):
```python
import time
import pandas as pd
from dune_client.client import DuneClient

def export_dune_to_csv(api_key, query_id, output_filename, max_retries=4):
    dune = DuneClient(api_key)
    for attempt in range(max_retries):
        try:
            df = dune.get_latest_result_dataframe(query_id)
            df.to_csv(output_filename, index=False)
            return df
        except Exception as e:
            if "429" in str(e) and attempt < max_retries - 1:
                time.sleep(60 * (attempt + 1))
            elif "402" in str(e):
                return None  # quota exhausted, switch accounts
            else:
                return None
```
- A real API key was accidentally pasted in plaintext into chat during this process and should be rotated on dune.com (Settings → API Keys) if not already done.

---

## 6. Where Things Stand Right Now (Merge Stage)

All 8 raw tables are extracted and saved locally. A pandas merge script has been written and iterated on to combine them into a single position-level dataset (`merged_positions.csv`), one row per closed LP position (tokenId).

**The merge script performs, in order:**
1. Load all 8 confirmed-correct CSVs (see filenames above).
2. Inner join Open ↔ Close on `tokenId` (this is where the "closed positions only" scoping happens).
3. Attach tick range (`tickLower`/`tickUpper`, `range_width_ticks`) from Mint, joined on the open transaction hash.
4. Attach summed fees from Collect.
5. Attach pool metadata (token addresses, fee tier).
6. Attach swap-aggregate-derived lifetime features (swap volume, min/max tick during the position's life, and a `went_out_of_range` flag) — computed via a per-position loop against each pool's daily swap aggregates, matched to each position's actual open→close window.
7. Attach gas cost / wallet / nonce features from Transactions, separately for the open tx and the close tx.
8. Attach USD prices at open time and close time (joined on rounded-to-minute timestamp + token address), separately for token0 and token1.
9. Compute the label: `deposited_value_usd`, `withdrawn_value_usd` (includes fees), `net_profit_usd`, `is_profitable` (binary).
10. Explicitly flag (but not drop) all close-time/outcome-dependent columns as leakage risks, printed as a list to exclude from the feature set X at model-training time.
11. Save to `merged_positions.csv`.

**This script has NOT yet been run to a fully clean, final completion** — it was rewritten once already after column-name mismatches were discovered (initial guesses about column names didn't match the real exported schemas; corrected version now matches confirmed real schemas listed in Section 4). The most recent version of the script (referred to as `Merge_final.py` in the working files) uses `open_events.csv` directly (already has top-ups summed in) rather than re-deriving it from `open_events_raw.csv` + `open_events_topup.csv`.

**Status: the corrected final merge script has been generated but its output has not yet been confirmed as an error-free, fully successful run in this conversation.** This is the very next thing to do.

---

## 7. Next Steps (in order)

1. **Run the final merge script** (`Merge_final.py` logic — see Section 6) against the confirmed-correct 8 files. Debug any remaining column-mismatch or type errors (a few have already come up and been fixed: mixed-type warning on `mint_amount0`, `KeyError` on renamed columns — watch for similar issues, e.g. timestamp dtype mismatches during the price joins).
2. **Apply token decimals** — this is flagged in the script but NOT yet implemented. Raw on-chain token amounts are in base units (e.g., wei-equivalent), not human-readable units. Must divide every raw amount (`total_token0_deposited`, `total_token1_returned`, `fees_collected_token0`, mint amounts, etc.) by `10 ** decimals` for that token before the USD profitability math is trustworthy. Decimals: USDC/USDT = 6, DAI/WETH = 18, WBTC = 8. **This directly affects the correctness of `net_profit_usd` and `is_profitable` — do not trust the label until this is fixed.**
3. **Sanity-check the merged dataset**: row count (expect close to 154,761, possibly slightly less after all the left-joins, depending on match rates on price/tx joins), null-rate check per column (especially the price joins — some open/close minutes may not have an exact price match and will need a fallback, e.g. nearest available minute instead of exact match), and a manual spot-check of a handful of individual positions' economics to eyeball whether `net_profit_usd` looks sane in USD terms.
4. **Re-derive `is_profitable` correctly** for V3's range-dependent IL behavior (see Section 2's noted nuance) if time allows — the current calc treats it as a simple deposited-vs-withdrawn USD difference, which is directionally correct but doesn't explicitly separate "IL" from "fees earned" from "went-out-of-range effects." Not strictly required for a working binary label, but would strengthen the analysis/writeup.
5. **Build the leakage-safe feature set X**: drop all columns flagged in the script's `leakage_cols` list from the training features; keep them only for label construction and post-hoc analysis.
6. **EDA**: class balance of `is_profitable`, distribution of `range_width_ticks`, correlation of features with the label, checking for the survivorship-bias caveat (Section 2) qualitatively.
7. **Modeling**: baseline model (logistic regression) then a gradient-boosted tree model (XGBoost/LightGBM likely, given tabular data with mixed feature types) — not yet started.
8. **Writeup**: document the original full-scope vision vs. the final constrained scope explicitly (as in Section 3 of this document) as a legitimate "real-world data engineering constraint" narrative — this is a *strength* to highlight in the MSc report and resume framing, not something to hide: it shows the ability to scope a large real-world data problem down to something tractable under genuine resource constraints, which is a realistic on-chain analyst skill.
9. **Optional future upgrade** (if time/resources allow before final deadline): revisit pool count (10 → 20/30) and/or extend the time range closer to present, now that the real (lower-than-naive-estimate) Dune credit costs are better understood — there may be more headroom than originally assumed.

10. **IMPORTANT UPDATE — credits have refreshed / a new free account is available.** Nash has confirmed that free-tier Dune credits are available again (e.g. a new billing cycle and/or a fresh free account), meaning the credit exhaustion that forced the top-10-pools / 2021-2024 scope cut is no longer an active blocker right now. This reopens the option to pursue closer to the **original full vision** (Section 2) — e.g., re-running the top-N-pools count queries at a higher N (20, 30, or beyond) and/or extending the date range toward the present — before committing to the final constrained dataset. Worth deciding early in the next session whether to: (a) proceed with the already-extracted top-10/2021-2024 dataset as final (it already exceeds the 100k-row and 50-column targets, so this is a perfectly legitimate choice), or (b) spend some of this renewed credit budget re-pulling a larger version first. This is a call Nash should make deliberately given deadline pressure — don't assume expansion is free just because credits reset, since the same 2,500/account ceiling per cycle still applies and multiple tables would still need to be re-pulled and re-merged if scope changes.
