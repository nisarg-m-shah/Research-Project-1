"""
Chapter 3 — EDA chart generation
Run against final_merged_dataset_v5.csv. Saves each chart as a PNG you can
paste straight into the report. Captions match the chapter's structure.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")
df = pd.read_csv("final_merged_dataset_v5.csv")

# ============================================================
# 3.1 UNIVARIATE
# ============================================================

# Figure 1: log_deposit_value_usd distribution (+ raw, for skew comparison)
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sns.histplot(df['deposit_value_usd'], bins=60, ax=axes[0])
axes[0].set_title("Raw deposit_value_usd (right-skewed)")
sns.histplot(df['log_deposit_value_usd'], bins=60, ax=axes[1], color="darkorange")
axes[1].set_title("log_deposit_value_usd (skew-corrected)")
plt.tight_layout()
plt.savefig("fig1_deposit_value_distribution.png", dpi=150)
plt.close()

# Figure 2: range_width_normalized distribution
plt.figure(figsize=(8, 4))
sns.histplot(df['range_width_normalized'].clip(upper=df['range_width_normalized'].quantile(0.99)),
             bins=60)
plt.title("range_width_normalized (clipped at 99th percentile for readability)")
plt.tight_layout()
plt.savefig("fig2_range_width_distribution.png", dpi=150)
plt.close()

# Figure 3: target class balance
plt.figure(figsize=(5, 4))
df['profitable'].value_counts(normalize=True).sort_index().plot(kind='bar', color=["#d62728", "#2ca02c"])
plt.xticks([0, 1], ["Unprofitable (0)", "Profitable (1)"], rotation=0)
plt.ylabel("Proportion")
plt.title("Target class balance")
plt.tight_layout()
plt.savefig("fig3_target_balance.png", dpi=150)
plt.close()

# ============================================================
# 3.2 BIVARIATE
# ============================================================

# Figure 4: range_width_normalized by profitable
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='profitable', y='range_width_normalized',
            showfliers=False)
plt.xticks([0, 1], ["Unprofitable", "Profitable"])
plt.title("Chosen range width vs profitability")
plt.tight_layout()
plt.savefig("fig4_range_width_by_target.png", dpi=150)
plt.close()

# Figure 5: log_deposit_value_usd by profitable
plt.figure(figsize=(6, 5))
sns.boxplot(data=df, x='profitable', y='log_deposit_value_usd')
plt.xticks([0, 1], ["Unprofitable", "Profitable"])
plt.title("Deposit size vs profitability")
plt.tight_layout()
plt.savefig("fig5_deposit_by_target.png", dpi=150)
plt.close()

# Figure 6: profitable rate by pool
plt.figure(figsize=(9, 5))
pool_rate = df.groupby('pool_address')['profitable'].agg(['mean', 'count']).sort_values('mean', ascending=False)
sns.barplot(x=pool_rate['mean'].values, y=[a[:10] + '...' for a in pool_rate.index])
plt.xlabel("Profitable rate")
plt.title("Profitable rate by pool (labels truncated)")
plt.tight_layout()
plt.savefig("fig6_profitable_rate_by_pool.png", dpi=150)
plt.close()
print(pool_rate)  # paste this table into the write-up alongside the chart

# Figure 7: range_overlaps_recent_trading vs profitable (the skewed-but-strong feature)
plt.figure(figsize=(6, 5))
overlap_rate = df.groupby('range_overlaps_recent_trading')['profitable'].mean()
overlap_count = df['range_overlaps_recent_trading'].value_counts()
ax = overlap_rate.plot(kind='bar', color=["#d62728", "#2ca02c"])
plt.xticks([0, 1], [f"No overlap\n(n={overlap_count.get(0,0)})", f"Overlaps\n(n={overlap_count.get(1,0)})"], rotation=0)
plt.ylabel("Profitable rate")
plt.title("Profitability by range-overlap-with-recent-trading")
plt.tight_layout()
plt.savefig("fig7_overlap_by_target.png", dpi=150)
plt.close()

# ============================================================
# 3.3 MULTIVARIATE
# ============================================================

# Figure 8: correlation heatmap of numeric features actually used in the model
numeric_features = [
    'range_width_normalized', 'fee_tier_pct', 'mint_amount0_adj', 'mint_amount1_adj',
    'log_deposit_value_usd', 'token0_price_open', 'token1_price_open',
    'token0_price_std_24h', 'token0_price_pct_change_24h', 'token0_price_range_pct_24h',
    'token1_price_std_24h', 'token1_price_pct_change_24h', 'token1_price_range_pct_24h',
    'pre_open_avg_daily_swap_count_3d', 'pre_open_avg_daily_token0_volume_3d',
    'lp_prior_position_count',
]
plt.figure(figsize=(13, 10))
corr = df[numeric_features].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0,
            annot_kws={"size": 7}, cbar_kws={"shrink": 0.8})
plt.title("Correlation heatmap — model numeric features")
plt.xticks(rotation=45, ha='right', fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig("fig8_correlation_heatmap.png", dpi=150)
plt.close()

print("\nSaved 8 figures: fig1 through fig8, all PNG at 150dpi.")
print("Send them back and I'll drop them into Chapter 3 with proper captions.")