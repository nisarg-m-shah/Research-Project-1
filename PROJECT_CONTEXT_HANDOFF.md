# Project Context Handoff — Uniswap V3 LP Profitability Prediction

You are picking up an MSc Big Data Analytics research project (referred to
internally as "Project C") in progress. This document is a complete history
and state dump so you can continue exactly where the previous session left
off, with no re-discovery needed. Read this fully before touching any code.

## 1. Who this is for and what the project is

Nisarg Shah is an MSc Big Data Analytics student (St. Xavier's College, Mumbai).
This project is his final research submission AND a portfolio piece
targeting on-chain data analyst roles. Both purposes matter: the work must
satisfy a formal academic rubric (the professor's lecture decks on Data
Cleaning, EDA, Feature Engineering, Variable Selection, and Logistic
Regression vs. Random Forest define the expected methodology) and also be
presentable as a real, defensible data science project to an employer.

**The problem**: Uniswap V3 liquidity providers (LPs) must choose a
concentrated price range when opening a position. This range choice is the
single biggest driver of whether the position ends up profitable, yet most
LPs pick it by rule of thumb. The project predicts, at the exact moment a
position opens, whether it will close profitably, using only information
knowable at that moment (chosen tick range, pool characteristics, deposit
composition, pre-open market conditions, wallet experience). This is framed
as a binary classification problem.

**Original ambition vs. actual scope**: the original plan was all Uniswap V3
pools, full history since V3's 2021 launch through the present, 100,000+
positions. This was cut back hard by Dune Analytics' free-tier API datapoint
quota (~2,500 datapoints/account/month, where a datapoint is roughly
rows x columns of any API-read result — the very first full-universe pull
alone would have been 5x a single month's quota). The actual scope used:
**top 10 pools by swap volume, 2021-05-01 to 2024-01-01**. This scope
narrowing is a deliberate, documented, real-world data-engineering
constraint — it is written up as a legitimate strength in the report (shows
ability to scope a large real problem under genuine resource limits), not
hidden. If asked to expand scope (more pools, longer time range), that is a
valid stretch goal but should not happen before the current pipeline is
locked and working end-to-end — do not expand scope opportunistically
mid-fix.

## 2. Data extraction history (do not redo unless asked)

Nine raw tables were pulled and are the foundation of everything downstream:

| # | Table | Source | Captures |
|---|---|---|---|
| 1 | Mint | Dune | tick range (tickLower/tickUpper) chosen at open — the headline feature |
| 2 | Open events (IncreaseLiquidity) | Dune | position open/top-up, deposit amounts |
| 3 | Close events (DecreaseLiquidity) | Dune | position close, ALL partial withdrawals summed |
| 4 | Collect | Dune | fees collected, summed across all collect events |
| 5 | Pool metadata (PoolCreated) | Dune | token addresses, fee tier, tick spacing |
| 6 | Swap daily aggregates | Dune | pool trading volume, traded tick range |
| 7 | Transactions | Google BigQuery (public dataset) | gas costs, wallet nonce |
| 8a/8b | Prices (minute) | Dune | USD token prices at exact open/close minutes |

Transactions moved to BigQuery permanently (not a temporary workaround)
because the full Dune pull would have blown the quota; BigQuery's public
`bigquery-public-data.crypto_ethereum` dataset has free 1TB/month processing
(not row x column metered). Prices stayed on Dune because BigQuery's public
dataset has no USD price table. Multiple Dune free accounts (4+) were used
to spread extraction across separate quota pools.

**A real data-quality bug was caught and fixed during extraction**: early
Close/Collect queries used `row_number() ... rn = 1` to take only the LAST
event per position, silently discarding amounts from earlier partial
withdrawals/collections. This affected ~8% of positions. Fixed by switching
to `GROUP BY tokenId` + `SUM()`. This fix is already baked into the current
data — do not reintroduce the row-number bug if touching extraction SQL.

**Files present but STALE/superseded, safe to ignore or delete**:
`open_events_raw.csv` + `open_events_topup.csv` (superseded by
`open_events.csv`), `transaction_metadata.csv` (superseded by
`transactions.csv`), `raw_prices_daily.csv` (abandoned early attempt),
`tx_hashes_for_bigquery.csv` / `price_timestamps.csv` (intermediate helper
files, keep for reproducibility but never feed into the merge).

## 3. Pipeline history — the bugs that mattered

The pipeline is at **v5** (`uniswap_pipeline_v5.py`). Do not regenerate from
scratch; this file's logic is correct and hard-won. Key fixes baked into v5
that must not be silently reverted:

1. **Token decimals conversion**: on-chain amounts are in base units, not
   human-readable. Must divide by `10 ** decimals` per token (USDC/USDT: 6,
   WETH/DAI: 18, WBTC: 8) before ANY profitability arithmetic. Getting this
   wrong doesn't just add rounding error — it shifts values by orders of
   magnitude and can flip the sign of the profitability label itself.
2. **Corrupted deposit-amount outlier fix**: `mint_amount0_adj` /
   `mint_amount1_adj` occasionally carried corrupted/overflowed values up to
   ~1e31, propagating to `inf` downstream. The outlier filter was originally
   scoped only to the withdrawal-side column
   (`total_token0_returned_adj`) — the deposit-side columns needed the
   identical filter (rows where any of the three amount columns exceed 1e10
   in decimal-adjusted units are dropped). **If you ever see a row count
   near 62,265 pre-filter and 62,250 post-filter with `profitable` mean
   0.60008, you are looking at the CORRECT v5 output.** An earlier corrupted
   run produced 55,279 rows — that number is WRONG and should never be
   trusted or cited; it came from a broken intermediate run, not the fixed
   pipeline. If you see 55,279 anywhere, flag it as stale.
3. **Leakage columns**: everything only knowable at or after close must be
   excluded from the feature matrix X. The confirmed, complete list is:
   `close_time`, `token0_price_close`, `token1_price_close`,
   `held_value_usd`, `returned_value_usd`, `true_fees_value_usd`,
   `impermanent_loss_usd`, `position_duration_hours` (= close_time -
   open_time, not knowable at open), `decrease_event_count` (counts
   withdrawal events across the position's WHOLE life, not knowable at
   open). The last two were NOT in the original leakage list Nash's own
   pipeline comments flagged — they were caught during the modeling-prep
   review. Do not let either back into X under any reframing.
4. **`lp_prior_position_count`** is a causal, chronologically-computed
   feature (counts only positions that opened earlier for that wallet). It
   must never be recomputed after downstream row drops — dropped rows still
   represent real historical positions and should still count toward this
   measure for later positions.

**Current confirmed dataset**: `final_merged_dataset_v5.csv`,
**62,250 rows, 35 raw columns**, target `profitable` is 60.0% / 40.0%
balanced. (Note: some older project notes cite "~62,195 rows, 24 columns" —
that figure is stale/imprecise; trust the pipeline's own debug print output
and the 62,250/35 figure verified directly against actual script runs.)

## 4. Modeling pipeline — three blocks, in order

### Block 1 — `block1prep.py` (prep + variable selection)
- Loads `final_merged_dataset_v5.csv`, sorts by `open_time`, does an
  **80/20 chronological split with NO shuffling** (split point falls at
  2023-01-21). This is non-negotiable for this dataset — a random split
  would leak future market regime information into training.
- Builds leakage-safe X/y using the exclusion list in Section 3.3 above,
  plus drops `tokenId`, `open_time`, `token0_address`, `token1_address` as
  IDs/redundant. `pool_address` (10 pools) is one-hot encoded with
  drop_first — kept as a feature because pool identity carries real signal
  (liquidity depth, pair volatility) not captured elsewhere.
- Scales continuous features with `StandardScaler` fit on TRAIN ONLY.
- Variable selection: filter (near-zero-variance check — none found;
  pairwise correlation >0.9 check — none found) + embedded (L1 Lasso +
  Random Forest importance).
- **Known finding, already resolved**: Lasso and RF disagreed sharply on
  `range_overlaps_recent_trading` and one low-frequency pool dummy
  (`0x3416cf6c...`, ~1.8% of training rows) — Lasso gave both huge
  coefficients, RF ranked both near zero. Root cause confirmed by checking
  raw base rates: `range_overlaps_recent_trading` is 99.0% one value (only
  514/49,800 train rows are the minority class), and that pool has too few
  rows for a stable linear coefficient. Resolution: kept both features
  (XGBoost handles rare/skewed categoricals better than linear coefficients
  do), but Lasso's specific coefficient on the low-frequency pool is flagged
  in the report as unreliable.
- Outputs: `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`,
  `rf_feature_importance.csv`, `lasso_selected_features.csv`.
- **One cleanup applied downstream (in Block 2, not Block 1)**:
  `deposit_value_usd` (raw) is dropped as redundant with
  `log_deposit_value_usd` — both scripts that consume Block 1's output
  should drop it if present.

### Block 2 — `block2modelling_evaluation.py` (baseline + challenger + eval battery)
- Baseline Logistic Regression (C=1.0) + challenger XGBoost
  (n_estimators=500 with early_stopping_rounds=30 on a chronological 15%
  validation carve, max_depth=4, learning_rate=0.05, subsample=0.8,
  colsample_bytree=0.8).
- **A real bug was found and fixed here**: the first threshold-selection
  pass used `f1_score(y_true, preds)` with its default (positive-class-only
  F1). Since class 1 (profitable) is the majority, this drove the sweep
  toward predicting "profitable" for nearly everyone (class-0 recall
  collapsed to 0.022 — a near-degenerate classifier). Fixed by switching to
  `average='macro'`. **Always use macro F1 for threshold selection on this
  problem, never positive-class-only.**
- Final threshold: **0.56** (train macro-F1-optimal).
- Full evaluation battery per the course's Model Evaluation deck:
  classification report + confusion matrix at the justified threshold,
  ROC-AUC, PR-AUC, KS statistic, Gini (=2*AUC-1), Brier score, decile/lift
  table, train-vs-test gap, chronological learning curve (train-size
  slices), SHAP (TreeExplainer on a 3000-row test sample, values computed
  but summary plot not yet rendered/saved as an image).
- **Untuned XGBoost final numbers (test set)**: ROC-AUC 0.5955, Gini 0.1910,
  KS 14.4, PR-AUC 0.7452, Brier 0.2191.
- **Diagnosis, load-bearing for the whole report**: the train/test
  performance gap (train AUC 0.68 vs test AUC 0.596) is NOT classical
  overfitting. The learning curve shows train AUC actually FALLING as
  training data grows (0.739 to 0.680) while test AUC barely rises and
  plateaus — the signature of genuine concept drift, not high variance. This
  is corroborated by a real class-balance shift across the time split:
  train is 58.3% profitable, test is 67.0% profitable (2021-2022 vs 2023
  market conditions differ). **This diagnosis was later re-tested and
  confirmed under real hyperparameter tuning in Block 3, not just assumed.**
- PSI/CSI: deliberately NOT computed. There is no live production scoring
  population, only a static historical split — the report states this
  explicitly as "not applicable as scoped" rather than fabricating a drift
  simulation. Do not add a fake PSI number.

### Block 3 — `block3hyperparametertuning.py` (real tuning + real CV)
- Built specifically because Block 2 originally skipped formal tuning under
  a time-pressure framing. **That framing is now void — the project has 10
  days, not 1.5. All steps must be done properly, nothing skipped or
  excused by a compressed timeline. The academic submission timeline is
  officially one month; any internal working-time pressure must NEVER be
  mentioned anywhere in report text.**
- `TimeSeriesSplit(n_splits=5)` on the already-chronologically-sorted
  X_train — 5 genuinely chronological folds, no shuffling, matching the
  no-shuffle discipline used everywhere else in this project.
- `RandomizedSearchCV` (60 iterations, scoring='roc_auc') as primary search
  + Optuna TPE sampler (60 trials) as an independent Bayesian cross-check,
  both over the identical CV folds.
- **Results, already run and confirmed**: RandomizedSearchCV best CV AUC
  0.5936, Optuna best CV AUC 0.5934 — converged to within 0.0002 of each
  other via two independent methods, strong evidence the search space is
  genuinely well-explored. Fold scores for the winning params: [0.5883,
  0.5859, 0.6049, 0.5785, 0.6103], std 0.012 (well under the 0.05
  instability threshold used throughout this project — the model's weakness
  is stable across time windows, not a fluke of one split).
- Best params (RandomizedSearchCV, marginally better than Optuna, carried
  forward): `max_depth=4, learning_rate=0.02, n_estimators=100,
  subsample=0.8, colsample_bytree=0.6, min_child_weight=7, reg_alpha=5,
  reg_lambda=2`.
- **Refit-and-test result (seed 0): ROC-AUC 0.5955, Gini 0.1909, KS 14.9, PR-AUC
  0.7513, Brier 0.2175, versus Block 2's untuned seed-0 numbers (ROC-AUC 0.5929 /
  Gini 0.1859 / KS 14.4 / PR-AUC 0.7438 / Brier 0.2190, from
  `block2_base_metrics.csv`)**. 120 total search trials across two independent
  methods moved seed-0 test AUC by +0.0026 (Gini +0.0051). Across an audited
  seed sweep, untuned test AUC runs 0.5825-0.5966 (mean 0.5929, sd 0.0038) and
  tuned 0.5929-0.5955 (mean 0.5946, sd 0.0008); the per-seed tuning gain spans
  -0.0020..+0.0125 (mean +0.0016), so tuning moves test AUC by roughly a
  thousandth to a few thousandths, comparable to seed-to-seed spread rather
  than a step-change. This is the load-bearing finding of the whole project: it
  upgrades the "feature-set ceiling, not a tuning gap" diagnosis from a
  hypothesis to a tested conclusion.
- **One nuance to keep reporting honestly**: at the same macro-F1 threshold
  (0.56), the tuned model's test class-0 recall (0.239) is notably worse
  than the untuned model's (0.409), despite near-identical AUC. The tuned
  model is more regularized (reg_alpha=5 vs ~0, only 100 trees vs
  early-stopping's 60-tree pick from a 500 budget) and its probability
  distribution has a different shape, so the same threshold-selection
  process lands somewhere less balanced. Report both families of metric,
  never just AUC/Gini alone, for exactly this reason.
- Outputs: `best_hyperparameters.csv`.

### `eda_charts.py` — Chapter 3 chart generation
Generates 8 PNGs (`fig1` through `fig8`) against `final_merged_dataset_v5.csv`
directly (not the split files). Already run once; images already embedded
in the report (see Section 5). If re-run, note real findings already
confirmed from the charts:
- Profitable positions choose visibly NARROWER ranges (median
  `range_width_normalized` ~17) than unprofitable ones (~33) — consistent
  with V3's concentrated-liquidity design rewarding correctly-placed tight
  ranges. This is the project's cleanest, most defensible domain finding.
- Profitable positions carry larger deposits (median log-deposit ~12.4 vs
  ~11.2 for unprofitable).
- `range_overlaps_recent_trading`: non-overlapping positions are profitable
  only 23.0% of the time (n=601) vs 60.3% for overlapping ones (n=61,649) —
  the single strongest bivariate relationship found in the whole EDA.
- Pool-level profitable rate varies 46-82%, but the two smallest pools
  (n=25, n=21) are noise, not real effects — only the two dominant pools
  (n=31,414 at 62.3%, n=18,313 at 56.6%) should be treated as a real pool
  effect.
- No feature pair exceeds the 0.9 correlation threshold, but moderate
  clusters exist (0.57-0.86) among pre-open volatility/deposit features,
  expected given shared derivation; noted in the report but not acted on
  since XGBoost tolerates this better than a linear model would.

## 5. The report — current state

A full 12-chapter report has been generated **programmatically** as a
`.docx` (via Node's `docx` library, not python-docx despite what old notes
say — check which toolchain is actually present in the repo before
assuming) at `Project_Report_Draft.docx`, following
`Project_Report_Template.docx`'s exact chapter structure. Current state:
25 pages, real numbers throughout (no fabricated content anywhere),
real EDA charts embedded with real captions, real Block 3 tuning results
populating Chapters 7.4 and 9.

**One template deviation, intentional**: the title page's "Domain: Banking
/ Healthcare / Retail (delete two)" line doesn't fit a DeFi project — it was
replaced with "Domain: Decentralized Finance (DeFi) / Blockchain Analytics"
rather than forcing a wrong checkbox.

**Standing rule, already enforced, must not regress**: no mention anywhere
in the report of any compressed/short internal working timeline (the
official academic timeline is one month; an earlier draft mistakenly
implied a 1.5-day crunch in several places — Chapters 1.5, 7.4, 9, 11.2, and
12.2 all had this language and it was explicitly stripped out on request).
If you ever add new report text, never reintroduce timeline-pressure
framing as a justification for a methodological choice — if a step was
skipped, either do the step properly or state the limitation without a
timeline excuse.

**Genuinely still open (`[ PENDING ]` placeholders in the docx, marked in
grey italic)**:
1. **Front matter**: Nash's full name, roll number, guide/faculty name,
   institution name, month/year, and a personal 2-3 sentence acknowledgement
   paragraph. Cannot be filled without Nash's input.
2. **Chapter 11.2 (cost-benefit analysis)**: needs the actual dollar cost
   ratio between a false positive (LP opens a position the model called
   profitable that turns out unprofitable) and a false negative (LP skips a
   position that would have been profitable). This should be computed from
   the dataset's own real P&L, not assumed:
   `net_profit_usd = true_fees_value_usd - impermanent_loss_usd`, then
   compare `mean(net_profit_usd | profitable==1)` against
   `mean(net_profit_usd | profitable==0)`. This was explained to Nash but
   not yet run — if he provides the output numbers, write the real Chapter
   11.2 and revisit whether the 0.56 threshold in 11.3 should shift based on
   the resulting cost ratio.
3. **Appendix A**: full data dictionary (all 35 raw columns + 27 final
   model-input columns, one row each with dtype/source/description) —
   mechanical expansion from Chapters 2.2 and 6.2's summaries.
4. **Appendix B**: paste the actual leakage-exclusion code, the time-split
   logic, and the final XGBoost training call as monospace code blocks
   (Consolas/Courier New 9-10pt per the template's own instruction).
5. **Appendix C**: the SHAP summary plot. `shap_values` are already computed
   in Block 2 on a 3000-row test sample but never rendered/saved as an
   image — needs `shap.summary_plot(shap_values, X_test_sample)` run and
   saved as a PNG, then embedded the same way the Chapter 3 EDA figures
   were (see `fig()` helper in the report-generation script — reuse it).

## 6. Working conventions — apply without being asked

- **No em dashes anywhere in written drafts.** Standing project-wide rule.
- **Never invent numbers.** Every figure in the report must trace to an
  actual script run's output. If something is missing, mark it
  `[ PENDING ]` in grey italic (see the `pi()` helper in the report script)
  rather than filling a plausible-looking placeholder value.
- **Honest framing of weak results, always.** Test ROC-AUC 0.596 is modest
  by the course's own reading scale (KS/Gini under 20 = "poor, barely better
  than random" per the Model Evaluation deck) — this has been stated
  plainly throughout and must continue to be. Do not let report language
  drift toward overselling a weak result. Equally, do not undersell real
  findings (the range-width and range-overlap bivariate results are
  genuinely strong and interpretable — say so directly).
- **A suspiciously high metric is a leakage red flag, not a win.** Per the
  course's own reading scale, Gini >75 or near-perfect AUC should trigger
  immediate leakage suspicion, not celebration, if it ever appears after a
  pipeline change.
- **No shuffling, ever, on this dataset.** Every split, every CV scheme,
  every learning-curve slice must respect chronological order. This is
  on-chain time-series-adjacent data; a random split is a leakage bug here.
- **Don't skip methodology steps now that time allows.** If a course-deck
  step (e.g. formal CV, formal hyperparameter search, SHAP, decile/lift
  tables) was previously skipped or stubbed for any reason, redo it properly
  rather than leaving a shortcut in place. Nash has explicitly asked for
  this standard going forward.
- **Nash corrects mistakes directly and expects immediate, consistent
  propagation of the fix** — not just a one-off acknowledgement. If a number
  or claim is corrected once, check the rest of the codebase/report for the
  same stale value before considering the correction done (this has
  happened twice already: the 55,279 vs 62,250 row-count confusion, and the
  timeline-mention strip-out across five separate chapters).
- Nash is highly technically literate — do not over-explain basic ML/stats
  concepts, but DO explain DeFi/crypto-specific mechanics plainly when
  relevant (he has said this explicitly).
- Prefers dense, well-structured, direct communication over long
  prose explanations.

## 7. Immediate next steps, in priority order

1. Run the cost-benefit calculation described in Section 5, item 2, and
   write the real Chapter 11.2 (and re-check Chapter 11.3's threshold
   recommendation against the resulting cost ratio).
2. Generate and embed the SHAP summary plot (Appendix C) using the
   already-tuned Block 3 model and the already-computed `shap_values`.
3. Fill Appendix A (data dictionary) and Appendix B (code snippets) —
   mechanical but necessary for a complete submission.
4. Get front-matter personal details from Nash directly (name, roll number,
   guide name, institution, acknowledgement) — cannot be inferred.
5. Once all of the above are in, do a final full read-through of all 25+
   pages for consistency (this project has had real stale-number bugs
   before — re-verify every number against its actual source script output
   one more time before calling it final).
6. Separately, and lower priority than the report itself: a Streamlit
   dashboard was discussed as a portfolio-piece stretch goal (not required
   for the academic grade) — only pick this up once the report is fully
   submission-ready. A portfolio prompt was drafted in an earlier session
   for a separate context; if asked to continue that work, ask Nash for the
   GitHub repo URL and a live Streamlit demo URL, both still outstanding.

## 8. File inventory (what should exist in the working directory)

- `final_merged_dataset_v5.csv` — the authoritative dataset (62,250 rows).
- `uniswap_pipeline_v5.py` — the pipeline that produces it (do not rerun
  unless the raw source CSVs change; it is correct as-is).
- `block1prep.py`, `block2modelling_evaluation.py`,
  `block3hyperparametertuning.py`, `eda_charts.py` — the four modeling/eval
  scripts, all already run successfully at least once.
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` — Block 1 output,
  consumed by Blocks 2 and 3.
- `rf_feature_importance.csv`, `lasso_selected_features.csv`,
  `model_comparison_test_metrics.csv`, `best_hyperparameters.csv` — result
  artifacts from Blocks 1-3.
- `fig1_deposit_value_distribution.png` through
  `fig8_correlation_heatmap.png` — EDA charts, already embedded in the
  report.
- `Project_Report_Draft.docx` — the live report, 25 pages, generated
  programmatically. Treat the report-generation script (not the .docx
  directly) as the source of truth if further edits are needed — editing
  the generated .docx by hand will be overwritten if the script is rerun.
- `Project_Report_Template.docx` — the original template the report
  structure must continue to follow.

Pick up from Section 7's priority list. Everything above this line is
context, not a to-do — do not redo completed work.
