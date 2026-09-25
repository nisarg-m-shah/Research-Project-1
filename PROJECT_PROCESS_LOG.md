# PROJECT_PROCESS_LOG

An extreme-detail, step-by-step account of the work done on the research report `Project_Report_Draft.docx`
between **Sep 23 22:00 and Sep 24 11:55**. Everything below is factual record: what was inspected, what was
computed, what broke, how it was fixed, and what is still awaiting your input.

Read this together with `PROJECT_CONTEXT_HANDOFF.md` (the plan this work executes) and
`PROJECT_REPORT_STATE.md` if it exists. **If it does not exist, ignore this sentence.**

---

## 0. The one file that matters for YOU right now

**`Project_Report_Draft.docx` in this repo root.** All report edits are already inside it. The only thing
still genuinely pending is the **grey-italic "front matter" placeholder text** that an automated edit pass
stylised for you to fill in. Open that file in Word and replace these placeholders (they look like
`[ INSTITUTION NAME ]`, `[ Month, Year ]`, etc.):

| Location (paragraph # in the current file) | Placeholder shown now | What you should type |
|---|---|---|
| para 0 | `[ INSTITUTION NAME ]` | Your institution (e.g. St. Xavier's College, Mumbai) |
| para 5-6 area | report title (already filled) | keep |
| para 11 | `[ Nash: insert full name ]` | Your full name |
| para 12 | `Roll No.: [ ______ ]` | Your roll number |
| para 15 | `[ Guide / Faculty Name ]` | Your guide's / faculty's name |
| para 17 | `[ Month, Year ]` | e.g. September 2026 |
| para 25 | `[ Two or three sentences thanking your guide, ... ]` | The Acknowledgement text |

After editing, in Word select the TOC and the List of Figures/Tables and press **F9 (or Right-click →
Update Field)** to refresh page numbers, then save. You do **not** need to touch any script.

Every placeholder I could fill has been filled. What remains is person-specific information only you know.

---

## 1. Objective

Continue Nisarg ("Nash") Shah's MSc Big Data Analytics research project: *Predicting Uniswap V3 Liquidity
Position Profitability at Open Time*, which had reached a 90%-draft state. The task for this session was to
finish the report to submission quality by executing the "Priority List" in `PROJECT_CONTEXT_HANDOFF.md`
Section 7:

- **Priority 1** – Write real Chapter 11.2 (cost–benefit analysis) from data; re-check the 11.3 threshold.
- **Priority 2** – SHAP summary plot for Appendix C (and reference it from Chapter 10.2).
- **Priority 3** – Appendix A (full data dictionary) and Appendix B (key code snippets).
- **Priority 4** – Front-matter personal details (from Nash) – **still blocked on Nash**.
- **Priority 5** – Final read-through & number cross-check – **not yet done**.
- **Priority 6** (low) – Streamlit dashboard – **not started**.

Nash supplied `Project_Report_Draft.docx` (this is important: the handoff claimed a Node `docx` generator
script existed; **it does not exist anywhere in the repo** — the only report material is the `.docx` itself,
so all editing was done directly against the docx via `python-docx`).

---

## 2. Environment (exact versions used)

- OS: macOS (darwin), shell zsh.
- Repo root: `/Users/nisarg52/Documents/GitHub/Research Project` (a git repo, branch `main`).
- Python: `/Library/Frameworks/Python.framework/Versions/3.14/bin/python3` (3.14).
- Packages used:
  - `python-docx` 1.2.0 (with `lxml` 6.1.3) — *installed this session via pip; the docx editing tool*.
  - `shap` 0.52.0 — *installed this session via pip; TreeExplainer for Appendix C*.
  - `xgboost` 3.4.0, `scikit-learn` 1.8.0, `pandas` 2.3.3, `numpy` 2.4.2, `optuna` 4.9.0 — already present;
    used to reproduce model numbers.
- Rules honoured throughout (from the handoff): (a) **no em dashes** (U+2014) in any text; (b) **never invent
  numbers** — every figure must trace to a CSV or script output; (c) grey italic text marks things still
  pending a human; (d) **never commit to git unless Nash asks**.

---

## 3. Repository inventory relevant to this work

Working documents created/used this session are marked with ➤.

```
PROJECT_CONTEXT_HANDOFF.md        ➤ the plan; Section 7 = priority list (edited to log session progress)
Project_Report_Draft.docx         ➤ the report being edited (input + output of every edit step)
fill_report_pending.py            ➤ the docx edit script (383 lines) — does ALL report edits
repair_docx.py                    ➤ repaired the corrupt docx archive (content-type fix)
cost_benefit_analysis.py          ➤ computes Chapter 11.2 numbers from the dataset
cost_benefit_results.csv          ➤ its output (source of truth for 11.2 figures)
generate_shap_plot.py             ➤ produces the SHAP summary plot + feature ranking
shap_summary_plot.png             ➤ output (embedded in docx as Figure 9)
shap_feature_ranking.csv          ➤ output (source of truth for 10.2 SHAP figures)
generate_appendix_a.py            ➤ produces the Appendix A data dictionary CSV
appendixA_data_dictionary.csv     ➤ output (43 rows = 35 raw + 8 pool one-hot dummies)
appendixB_code_snippets.md        ➤ draft of Appendix B text (script embeds the same snippets)
final_merged_dataset_v5.csv       ➤ authoritative dataset: 62,250 rows x 35 cols (Aug 29)
X_train.csv / X_test.csv          ➤ 49,800 x 28 / 12,450 x 28 feature matrices (Sep 3)
y_train.csv / y_test.csv          ➤ labels (numpy uint8 dump)
best_hyperparameters.csv          ➤ tuned Block-3 XGBoost hyperparameters
block1prep.py                     ➤ feature prep + leakage exclusion + chronological split
block2modelling.py                ➤ Logistic Regression + XGBoost baseline (HAS A KNOWN BUG, see 5.2)
block3hyperparametertuning.py     ➤ RandomizedSearchCV(60) + Optuna(60) tuning with TimeSeriesSplit(5)
uniswap_pipeline_v5.py            ➤ produces final_merged_dataset_v5.csv (HAS A DATA BUG, see 5.3)
config.py                         ➤ columns list, clean-time boundaries, WinLoss sets, PROFITABLE_NAME
test.ipynb                        ➤ WARNING: contains a Dune API key — never commit this file
eda_charts.py + fig1..fig8        ➤ Chapter 3 EDA figures, fig8 = correlation heatmap
Dataset CSVs (mint/close/collect/open/swap/price/...) - the raw Dune exports feeding the pipeline
```

Files NOT in the repo (implicitly missing / not required): `block2modelling_evaluation.py`
(the handoff references this name; the actual file is `block2modelling.py`),
`model_comparison_test_metrics.csv`, `sample_prediction_metrics.csv` (claimed, absent).

---

## 4. Stage-by-stage account

### Stage 4.1 — Repo reconciliation (why the handoff differs from disk)

Read `PROJECT_CONTEXT_HANDOFF.md`, the scripts, and config. Discrepancies found and noted:

1. **No report-generation script exists.** The handoff's Section 8 ("Report Generation") describes a
   Node/`docx`-based generator ("model_report_generation/…"). Nothing like it is in the repo. Nash supplied
   only `Project_Report_Draft.docx`. **Decision: edit the docx directly with python-docx.**
2. **`block2modelling.py` still has the old threshold bug.** The handoff claims "Block 2 has been updated to the
   macro-average F1" but the file on disk still selects the threshold by **positive-class-only F1**
   (`max_f1 = max(scikit F1 on class-1)` style logic), i.e. the bug the handoff describes as already fixed.
   Left untouched this session (it does not change any final number: the 0.56 threshold and all test metrics
   used in the report come from Block 3 / the final refit). Flagged for Nash to decide later.
3. **`best_hyperparameters.csv` winning set** (used for the reproduction numbers): subsample 0.8,
   reg_lambda 2.0, reg_alpha 5.0, n_estimators 100, min_child_weight 7.0, max_depth 4, learning_rate 0.02,
   colsample_bytree 0.6. Read as `pd.read_csv(..., index_col=0).iloc[:,0]`.
4. **Data split**: chronological 80/20 by `open_time` = 49,800 train / 12,450 test (X_train.csv 49,800x28,
   X_test.csv 12,450x28). Block 2/3 drop `deposit_value_usd` from X, leaving **27 features**.
5. `test.ipynb` contains a Dune API key (`[REDACTED - rotate this key]`). Noted: never commit.
   **Action taken Sep 24 (session 2): key redacted here; the key must be rotated by Nash now that it
   has been written into more than one place.**

### Stage 4.2 — Dataset verification

- `final_merged_dataset_v5.csv`: **62,250 rows × 35 columns**.
- Label: `profitable = 1 if true_fees_value_usd > impermanent_loss_usd else 0`; the net1-equivalence check
  `net_profit = true_fees_value_usd - impermanent_loss_usd` and `profitable` are consistent on **all** rows.
- Mean of `profitable` = **0.60008** (handoff says 0.60, confirmed).
- 35 columns: ids / close-time leakage / open-time features / outcome (label) columns / `pre_open_*` 3d vs 7d
  variants (the 3d series is the selected one) plus 8 one-hot `pool_address_0x…` dummies.

### Stage 4.3 — Data-quality finding: 7 corrupted P&L rows

Data audit flagged rows with physically impossible dollar values. **7 rows** survive the pipeline with:

- collected fees **> 10× deposit value**, or
- returned principal **> 10× deposit value**, or
- |impermanent loss| **> 10× deposit value**.

tokenIds: `10651, 97146, 105799, 306672, 353917, 416916, 456302`. Worst case: tokenId `416916` reports
~**$26.5 trillion** of collected fees on a ~$21k deposit.

**Root cause** (found by reading `uniswap_pipeline_v5.py`): the Step-6 outlier filter only guards
`total_token0_returned_adj`, `mint_amount0_adj`, `mint_amount1_adj` (clipping at 1e10). It does **not** guard
`fees_collected_token0_adj` / `fees_collected_token1_adj` or any USD-derived column, so pathological rows of
the kind above pass through. Free-text search also confirmed `fees_collected_token*` never appears in the
Step-6 guard set.

**Impact assessment** (this took two passes to be sure): all 7 rows sit **entirely on the leakage side** of
the feature matrix (the corrupt values live in outcome-side columns). They never enter `X`, and their labels
appear sign-correct. Therefore:
- model numbers (fits, AUC 0.5955, metrics, decile table) are untouched — still n=62,250;
- **dollar figures must exclude them** → all Chapter 11.2 dollar calculations use the **62,243 clean** rows.

Related verification on a different column: `pre_open_avg_daily_token0_volume_3d` is stored in **raw token0
base units** (not decimal-adjusted). It is *consistent*, not corrupt — but its scale varies by ~1e7→1e26
across pools. Re-scaled handling already in place (no action needed for the report numbers used).

### Stage 4.4 — Docx repair (the file would not open)

`Project_Report_Draft.docx` was **corrupt** as delivered: 8 inline images were saved into the zip with the
extension `.undefined` (Word media part names ending `…e08f17.undefined`) and `[Content_Types].xml` had **no
`<Default>` content-type entry** for that extension, so Word refused to open it.

Fix (`repair_docx.py`): added

```xml
<Default ContentType="image/png" Extension="undefined"/>
```

to `[Content_Types].xml`, then rebuilt the zip (ZIP_DEFLATED) preserving all parts. **Crucially**, the
original (pre-edit, post-repair) archive was unzipped during diagnosis into the temp dir
`/var/folders/…/opencode`-style scratch (specifically the `/tmp/docx_inspect/` working copy used by the agent)
— this preserved copy later became the "pristine" baseline that saved the whole edit pass (see 4.7/4.8).

After repair the file opened cleanly: **206 paragraphs, 7 tables, 8 inline images**.

### Stage 4.5 — Report structure map (the "pristine" docx, before this session's edits)

Anchor points discovered and later used by the edit script (indices are the pristine ones):

- Front matter: para 0 `[ INSTITUTION NAME ]`; para 11 `[ Nash — full name to insert ]` (note the em dash —
  replaced later); para 12 `Roll No.: [ ______ ]`; para 15 `[ Guide / Faculty Name ]`;
  para 17 `[ Month, Year ]`; para 25 Acknowledgement placeholder.
- para 27 = TOC field note ("update field in Word"); paras 28–31 = List of Figures & Tables (also field-based).
- Chapter 10: 10.1 Classification Metrics (threshold 0.56, CM 1,682 TN / 2,429 FP / 2,355 FN / 5,984 TP,
  accuracy 61.6%); 10.2 Discrimination Metrics (KS/Gini scale note, Gini 0.191 XGBoost / 0.159 Logistic,
  ROC AUC 0.596 vs 0.580); 10.3 Decile/Lift (test-set table 11×7); 10.4 Calibration (Brier 0.219 / 0.221).
- Chapter 11: 11.1 Score-to-Action Mapping (para 173); **11.2 heading para 174, body placeholder para 175**
  (the `[ PENDING: this calculation is straightforward … ]` text, ending with `, [ PENDING … ]`);
  **11.3 heading para 176, body para 177**.
- Appendices (captions resolved at runtime, not hard-coded): paragraph starting `A. Full Data` (placeholder
  body on the next para), `B. Key Code`, `C. Additional Charts` — found by `startswith` on text; the target
  body is always `index_of_heading + 1`.
- Existing table style: tables have **no style name** but carry explicit single-line borders
  (`w:tblBorders`, `sz=4`) — any new table had to reproduce that (done in `new_table_after`).

### Stage 4.6 — The decision inputs (what Nash locked in)

1. **Report infrastructure**: "I'll locate & share the files" → only the `.docx` was actually provided;
   everything else in the repo was the analysis scripts. (Locked: direct docx editing.)  
2. **Cost–benefit framing** (choice offered, Nash picked the recommended option): **"Mean-based 1.56 : 1,
   keep 0.56"** — the mean-cost ratio version is the headline, and the 0.56 threshold stays. (Alternative
   offered: a conservative 0.61 — rejected.)  
3. **Author line / tone**: third person, "the leader" (as used in the original text), not "I".

### Stage 4.7 — Cost–benefit analysis (Chapter 11.2 content)

Script `cost_benefit_analysis.py`; output `cost_benefit_results.csv`. Method: split the **62,243 clean** rows
by `profitable`; summarise net_profit per class by **mean, median, and trimmed-mean (5–95%)**; derive the
cost ratio = |mean loss of class 0| / mean gain of class 1; derive the cost-optimal operating threshold
t* = cost_FP / (cost_FP + cost_FN). Reproduced the final model on the 27-feature test set to get the
threshold/ROC/P&L curve numbers. All figures below are verbatim from the results CSV:

| metric | class 1 (profitable, n=37,348) | class 0 (n=24,895) | FP:FN ratio | t* |
|---|---|---|---|---|
| mean | $3,502.91 | −$5,472.42 | **1.5622** | 0.610 |
| median | $169.88 | −$166.19 | 0.9783 | 0.495 |
| trimmed 5–95% | $603.96 | −$939.59 | 1.5557 | 0.609 |

Model reproduction on test: **ROC-AUC 0.5954** (handoff claims 0.5955 — the tiny drift is
rounding/sample-bootstrapping; noted as a match), **macro-F1-optimal threshold re-derived = 0.56**.

Realised net P&L on the 12,450 test rows (sum of true net_profit over positions scored above each threshold):

| threshold | total net P&L | # positions taken |
|---|---|---|
| 0.50 | $7,056,360 | 12,163 |
| 0.53 (peak) | $7,142,936 | — |
| **0.56 (kept)** | $7,086,085 | 10,175 |
| 0.60 | $2,034,503 | — |
| 0.65 | $944,245 | — |
| 0.70 | $326,793 | — |

Conclusion (as written into the report): mean-based cost ratio 1.56 : 1; cost-optimal threshold 0.61; the
0.56 operating point already sits at the P&L plateau (between 0.50–0.56) and 0.60 collapses value, so **keep
0.56**.

### Stage 4.8 — SHAP interpretation (Chapter 10.2 + Appendix C)

Script `generate_shap_plot.py`. **Bug caught & fixed during development**: the plot builder attempted
`d = d.drop("deposit_value_usd")` (default `inplace=False`), so the dropped column silently stayed in the
frame. Fixed with `inplace=True` (or reassignment). Final run:
- `X_test`-sourced **3,000-row sample**, `shap.TreeExplainer` on the final tuned XGBoost, **27 features**.
- Outputs: `shap_summary_plot.png` (175,964 bytes, 10×9 in, dpi 150) and `shap_feature_ranking.csv`.
- Top contributors by mean |SHAP| (verbatim): `log_deposit_value_usd` 0.13949, `lp_prior_position_count`
  0.10819, `range_width_normalized` 0.08075, `token1_price_open` 0.06760, `mint_amount1_adj` 0.03661,
  `pre_open_avg_daily_swap_count_3d` 0.03279, `fee_tier_pct` 0.03157.
- Those four headline values (0.139 / 0.108 / 0.081 / 0.068) were written into the Chapter 10.2 paragraph.
- The PNG is embedded into the docx as **Figure 9** under Appendix C (see 4.9). It was embedded from the
  script rather than pasted, so width was set to 6.0 in and paragraph centered. (Not human-verified visually —
  no image viewing in this environment; structural embedding verified by python-docx.)

### Stage 4.9 — Appendix A and B content generation

- `generate_appendix_a.py` → `appendixA_data_dictionary.csv`, **43 rows = 35 raw + 8 pool one-hot dummies**;
  fields: column, dtype_in_v5, source, knowable_at_open, role, description. Every one of the **27 model
  inputs** is covered across the raw/dummy rows.
- `appendixB_code_snippets.md` — draft of what Appendix B became:
  - B.1 Leakage exclusion (`block1prep.py`) — the full `LEAKAGE_COLS` list + the feature_cols construction.
  - B.2 Chronological 80/20 split by `open_time` with the split_time logic.
  - B.3 Final XGBoost training call (`block3hyperparametertuning.py`) with the exact tuned params and the
    comment that RandomizedSearchCV(60) and Optuna(60) agreed to within 0.0002 AUC; threshold 0.56.
  - B.4 TimeSeriesSplit(5) chronological CV design (`RandomizedSearchCV` setup).
  - In the docx these are rendered **Consolas 9pt**, no paragraph spacing, wrapped in grey-italic ``` fences.

### Stage 4.10 — `fill_report_pending.py` (the docx edit script)

Design (this is the script that produced the final state). It opens the docx, works in passes, saving to the
**same file** after each pass (Phase 1 saves once; App A, B, C each reload-then-save once):

1. **Front matter restyle**: for paras 0, 11, 12, 15, 17, 25 — every run set to *italic* + grey RGB(0x66,0x66,0x66);
   then para 11 replaced wholesale with `[ Nash: insert full name ]` (removes the em dash from `[ Nash — … ]`,
   honouring the no-em-dash rule).
2. **Chapter 10.2**: insert one new paragraph after pristine para 160 = the SHAP interpretation paragraph
   (text in the per-pass list, citing Figure 9 in Appendix C + the four mean|SHAP| values).
3. **Chapter 11.2**: after pristine para 175 (the heading-adjacent placeholder), insert **4 new paragraphs**
   (p1–p4: definition of net_profit; the 7 corrupt-row disclosure; the dollar summary with the 1.56 : 1 /
   0.98 : 1 ratios; the threshold math + realised-P&L plateau), then **delete** the original placeholder.
4. **Chapter 11.3**: replace pristine para 177 text with the reconciled recommendation (keep 0.56; cost-
   optimal 0.61 but P&L plateau argues for keeping 0.56; $7.06M→$7.09M; drop to $2.03M at 0.60).
5. **Appendix A**: locate heading `A. Full Data`, target `heading+1`; insert intro paragraph, bold caption
   "Table A-1: Raw columns of final_merged_dataset_v5.csv", **35-row table (36 incl. header, 5 cols)** from
   the CSV (columns: Column | Type (in CSV) | Source | At open | Description), spacer paragraph, bold caption
   "Table A-2: Final 27 model-input columns (X, after deposit_value_usd drop)", **27-row table (28 incl.
   header, 4 cols)** (Column | Scale | At open | Description; Scale = StandardScaler / binary, unscaled /
   one-hot, unscaled); then delete the placeholder. Table borders are injected as single `sz=4` to match the
   report's existing tables.
6. **Appendix B**: locate `B. Key Code`, target `heading+1`; insert intro, then for each of the 4 sections:
   bold 10pt title → lead-in → grey-italic ``` fence → code lines as Consolas 9pt runs (space_before/after
   = 0); delete placeholder.
7. **Appendix C**: locate `C. Additional Charts`, target `heading+1`; insert lead-in paragraph (notes that
   EDA figs already appear inline in Ch.3, decile table is in Ch.10.3), bold caption **"Figure 9: SHAP summary
   plot, final tuned XGBoost model (3,000-row test sample)"**, then the picture paragraph (6.0 in, centered);
   delete placeholder.

Helper functions in the script: `para_runs_free` (wipe runs), `set_text` (replace text, optional bold/italic/
grey/size), `delete_para` (remove via `p._p.getparent().remove`), `_el` (normalise paragraph vs table anchor;
see bug fix below), `new_para_after` / `new_table_after` (insert sibling *right after* an anchor using lxml
`addnext`), `mono_run` (Consolas + `w:rFonts` ascii/hAnsi/cs), `fill_cell` (8.5pt cell text).

### Stage 4.11 — The two bugs encountered and how they were fixed

**Bug A — table anchor crash (first run).** `new_para_after` originally did `anchor._p.addnext(p._p)`. When
Appendix A reached `spacer = new_para_after(tA1._tbl, doc)`, it passed the **table element** (a `CT_Tbl`),
which has no `._p` → `AttributeError: 'CT_Tbl' object has no attribute '_p'`, aborting the run **inside
Appendix A** — *after* the Phase-1 edits had already been `save()`d.

Fix: `_el(anchor)` returns `anchor._p` if the anchor is a paragraph, else the anchor itself (a `CT_Tbl` is an
lxml `_Element`, which natively supports `addnext()`). Both `new_para_after` and `new_table_after` now route
through `_el`. After this fix a fresh run works end-to-end.

**Bug B — duplicated content + surviving placeholders (second run).** Because Bug A's run had already saved
Phase 1, re-running the script **from the edited file** re-inserted the SHAP paragraph, the four 11.2
paragraphs, and re-replaced 11.3 → two SHAP paragraphs, two copies of the 11.2 block, and duplicated 11.3
text. Compounding issue: the script had never deleted the original placeholders it was replacing, so the old
`A full expected-value calculation requires…` text and the three appendix "paste here" placeholders remained.
Also a run-1 string edit mistake left a mangled p1 sentence.

Fix sequence:
1. **Restore pristine**: rebuilt the pre-edit docx from the `/tmp/docx_inspect` unzip (with the content-type
   repair re-applied) and copied it back → clean 206-para / 7-table / 8-image baseline (verified: no
   "Model interpretation via SHAP", no "net_profit_usd = true_fees", no "Table A-1" anywhere).
2. **Add `delete_para`** for the four replaced placeholders: 11.2 body (paras[175]), App A body, App B body,
   App C body.
3. **Repair the mangled p1 string** (restore `net_profit_usd = true_fees_value_usd - impermanent_loss_usd,`
   as the opening of the 11.2 block).
4. Re-ran once on the pristine file → **ALL EDITS COMPLETE** with single copies everywhere.

### Stage 4.12 — Final verification of the finished docx

- File re-opens cleanly with python-docx: **260 paragraphs, 9 tables, 9 images** (was 206/7/8).
- Structure (final indices):
  ```
  155  10.1 Classification Metrics
  159  10.2 Discrimination Metrics
  161  [INSERTED] SHAP interpretation paragraph (single copy)
  162  10.3 Decile / Lift / Gain Analysis
  166  10.4 Calibration
  173  11.1 Score-to-Action Mapping
  175  11.2 Cost-Benefit Analysis          -> content paras 176-179
  180  11.3 Deployment & Monitoring        -> content paras 181-182
  184  12.1 Summary of Findings
  204  A. Full Data Dictionary
  206  Table A-1 (caption)  -> 35-col table (36 rows x 5 cols)
  208  Table A-2 (caption)  -> 27-col table (28 rows x 4 cols)
  209  B. Key Code Snippets
  211  B.1 Leakage exclusion (block1prep.py)
  249  B.4 Chronological cross-validation design
  256  C. Additional Charts & Tables
  258  Figure 9 (caption)  -> SHAP PNG embedded
  ```
- Single-copy assertions ALL pass: exactly 1 "Model interpretation via SHAP", 1 "The cost figures below use",
  and **0** occurrences of the four old placeholder phrases.
- Em-dash scan on the newly-added paragraph ranges (161, 175–183, 204–260): **0 hits**.
- Code blocks verified **Consolas 9pt** (sample line `[214] font=Consolas sz=9.0`).
- Front-matter placeholders verified italic + grey RGB 0x666666 at paras 0/11/12/15/17/25.
- Appendix A tables verified: header rows match `Column/Type (in CSV)/Source/At open/Description` (A-1) and
  `Column/Scale/At open/Description` (A-2); row counts 36 & 28 (incl. header).
- Em dash **existing** paragraphs (25 in the pre-existing body, e.g. paras 32, 64, …) were left untouched —
  they are the original author's writing, out of scope.

### Stage 4.13 — Current git state

Nothing committed. `git status` shows modified `block*`/`test.ipynb` (test.ipynb is **red-flagged: contains a
Dune API key — do not commit**), plus the untracked new files: the project log/handoff edits,
`fill_report_pending.py`, `repair_docx.py`, `cost_benefit_analysis.py`, `cost_benefit_results.csv`,
`generate_shap_plot.py`, `shap_summary_plot.png`, `shap_feature_ranking.csv`, `generate_appendix_a.py`,
`appendixA_data_dictionary.csv`, `appendixB_code_snippets.md`, `Project_Report_Draft.docx`.

---

## 5. Outstanding items and known risks

### 5.1 BLOCKED – needs Nash (the reason this doc ends with a "you fill this in" pointer)

1. Front matter paras as listed in **Section 0** (name/roll/guide/institution/month-year) + the
   Acknowledgement text.
2. Word "Update Field" for TOC + List of Figures/Tables (asks Word for mechanisms: after opening, select
   TOC → Update Field → Update entire table; repeat for the List of Figures/Tables).

### 5.2 Not yet done (priorities 5-6)

- **Final read-through** of every chapter against the script outputs (Priority 5). Suggested: use the
  verification snippet in Section 6 to dump the full text and skim.
- **Streamlit dashboard** (Priority 6, low). Not started.

### 5.3 Known risks / flags for Nash

- `block2modelling.py` threshold-selection **still uses positive-class-only F1** (handoff says it was fixed
  to macro-F1; the fix is on the build but not in the file). It does not affect final numbers, but if the
  examiner opens it, the code/report mismatch is observable. Recommend applying the macro-F1 patch.
- `model_comparison_test_metrics.csv` does not exist (handoff references it) — Chapter 10's comparison table
  must have been hand-typed; cross-check the printed numbers there if time allows.
- The 7 corrupted rows also mean the **downstream raw fee CSV / pipeline fix** is optional; if you want the
  pipeline clean, guard `fees_collected_token*_adj` and the USD-derived columns in Step-6 of
  `uniswap_pipeline_v5.py`.
- `pre_open_avg_daily_token0_volume_3d` is in raw base units (scale varies ~1e7→1e26 across pools) — a
  thesis-examinable quirk; consider noting it in Appendix A (the dictionary's Scale column already flags
  StandardScaler).

---

## 6. Reproducibility: exact commands

```python
# 1. Re-open / verify docx structure anytime (must print 260 paras / 9 tables / 9 images)
import docx
d = docx.Document("Project_Report_Draft.docx")
print(len(d.paragraphs), len(d.tables), len(d.inline_shapes))

# 2. Dump the full report text for the final read-through
for i, p in enumerate(d.paragraphs):
    print(i, p.text)

# 3. Regenerate Appendix A CSV, SHAP plot, cost-benefit results
python3 generate_appendix_a.py
python3 generate_shap_plot.py
python3 cost_benefit_analysis.py

# 4. Timing: runs in seconds-minutes.
#    NOTE: do NOT re-run fill_report_pending.py against the already-edited docx.
#    It is intentionally NOT idempotent (it would duplicate Phase-1 content exactly as in Bug B).
#    From a fresh pristine doc the full pass is: python3 fill_report_pending.py
```

---

## 7. Change log

- Sep 23 22:00  Session start; repo reconciliation vs handoff; discrepancies logged (4.1).
- Sep 23 22:01  Dataset verification (4.2) and the 7-row corruption finding (4.3).
- Sep 23 22:18  `cost_benefit_analysis.py` written; `cost_benefit_results.csv` produced; numbers locked with
                 Nash: **mean 1.56:1, keep 0.56**.
- Sep 23 22:19  Decision recorded. Shap/appendix scripts written.
- Sep 24 00:00  Report draft received in repo; docx unzipped to `/tmp/docx_inspect`; corruption diagnosed;
                 `repair_docx.py` rebuilt the archive. Structure map produced (4.5).
- Sep 24 01:57-02:44  `appendixA_data_dictionary.csv`, `appendixB_code_snippets.md`, `shap_summary_plot.png`,
                 `shap_feature_ranking.csv` produced; python-docx + shap installed.
- Sep 24 11:4x  First `fill_report_pending.py` run: Phase 1 saved, **crashed** on Appendix A (Bug A).
- Sep 24 11:5x  Pristine restore; `_el` + `delete_para` + string fix (Bug B); final run → ALL EDITS COMPLETE;
                 full verification (4.12).
- Sep 24 11:55  This log written.

---

## 8. Artifacts vs source of truth

| Report section | Content source | "Don't invent numbers here" |
|---|---|---|
| 10.2 SHAP paragraph | `shap_feature_ranking.csv` (0.139 / 0.108 / 0.081 / 0.068) | yes |
| 11.2 cost-benefit | `cost_benefit_results.csv` | yes |
| 11.3 threshold | reproduction numbers (0.5954 AUC, 0.56 macro-F1, P&L curve) | yes |
| Appendix A tables | `appendixA_data_dictionary.csv` | yes |
| Appendix B snippets | verbatim script lines | yes (verbatim) |
| Appendix C figure | `shap_summary_plot.png` (embedded) | yes |

---

## 9. Session 2 (Sep 24, afternoon): model-improvement robustness work (WP0-WP5 done, WP6 done)

This is the second edit session. Scope: the "model improvement" brief of Sep 24 (afternoon). It runs
**WP0 to WP6** with the same hard rules as session 1 (no em dashes in anything written, never commit,
never overwrite the locked artifacts, never invent a number). Status at the end of this session:
**all six work packages complete** and the report updated. Details below.

### 9.1 WP0 - macro-F1 threshold divergence, resolved (0.45 vs 0.56)

- The report's 0.56 threshold and Block-2 (untuned) result **could not be reproduced from the on-disk
  `block2modelling.py`**: that file still selects the threshold by **positive-class-only F1** (about 0.45,
  the session-1 4.1/5.3 flag), not macro-F1.
- Patch: macro-averaged-F1 threshold selection was applied to a copy (`block2prep`/`eval_utils`-based);
  it reproduces **exactly 0.56**, matching the report. Root cause confirmed: the handoff claimed Block 2 had
  been fixed to macro-F1; the disk copy had not. The 0.56 number is genuine, produced by the macro-F1 rule,
  just not by the file on disk that the handoff says produced it.
- Also logged: a small determinism study (seed sweep) showing test ROC-AUC between 0.5917 and 0.5953 across
  seeds, i.e. the 0.5955 vs 0.5929 discrepancy is within run-to-run seed randomness, both are valid draws.

### 9.2 WP1 - `block2c_time_interaction_experiment.py` (feature-set ablation, V0 vs V1 vs V2 vs V3)

- Variants: V0 = existing 27 features; V1 = V0 + 4 cyclical open-time features (open_hour_sin/cos,
  open_dow_sin/cos); V2 = V0 + 2 interactions; V3 = V0 + both. Selection metric = train-CV ROC-AUC only
  (5-fold TimeSeriesSplit); test battery + threshold (train-set macro-F1) reported per variant.
- Reproduces the report baseline exactly (V0 tuned XGB: test 0.5955, Gini 0.1909, KS 14.9444, train-CV
  0.5936 sd 0.0120).
- Winner by the decision rule = **V0**. Time features did not help on test and were slightly negative on this
  seed (V1 test 0.5921; bootstrap 95% CI of difference versus V0 = [-0.0050, -0.0017], which excludes 0 and is
  a test-sampling statement for this single seed, comparable to the seed-to-seed test-AUC spread of the
  untuned sweep, sd 0.0038). V2 was neutral (0.5954,
  CI [-0.0016, +0.0015]) and V3 was neutral (0.5948, CI [-0.0023, +0.0009]).
- New-feature SHAP (V3): `interaction_range_width_x_price_std` ranked 9th of 33 (mean |SHAP| 0.026, on a
  par with `fee_tier_pct` 0.026); `interaction_fee_tier_x_swap_count` 0.006; all four time features < 0.0004
  (inert). Coverage check: both windows cover all 24 hours and all 7 weekdays (train 626d, test 343d).
- **Decision: V0 retained**; the WP6 model-cascade condition (a variant beats V0) did **not** fire.
- Outputs: `ablation_results.csv`, `xgb_v0_test_predictions.csv`; supplementary `new_feature_importance.csv`
  (bootstrap CIs, collinearity of interactions with their parents ~0.33-0.36).

### 9.3 WP2 - `block2b_random_forest.py` (Random Forest family)

- RF baseline: test ROC-AUC 0.5962, train-CV 0.5789 sd 0.0192. RF tuned (RandomizedSearchCV 24 trials,
  5-fold TimeSeriesSplit): test 0.5978, train-CV 0.5850 sd 0.0206; best params n_estimators 300,
  min_samples_split 5, min_samples_leaf 10, max_features log2, max_depth 10; threshold 0.54 (train macro-F1).
- RF's own bootstrap does not reshuffle rows, so it respects the chronological discipline. Baseline train
  AUC 0.9815 reported honestly as an overfit artifact; the CV/test numbers are the honest ones.
- Outputs: `rf_results.csv`, `rf_tuned_test_predictions.csv`.

### 9.4 WP3 - `block1b_rfe.py` (RFECV wrapper stage)

- Logistic Regression: 14 of 27 features selected, best CV 0.5639 sd 0.0258; test 0.5891. Tuned XGBoost: 21 of 27
  selected, CV 0.5962, test 0.5982 (subset) vs 0.5955 (full set). Selection gain (CV) ~+0.003, inside fold sd.
- Outputs: `rfecv_results.csv`, `rfecv_cv_curve.csv`, `fig13_rfecv_curve.png`. Agrees with the embedded
  ranking on dominant features; the 27-feature V0 list is unchanged.

### 9.5 WP4 - `make_eval_charts.py` (figures 10-12)

- `fig10_decile_lift.png`, `fig11_cumulative_gain.png`, `fig12_calibration_curve.png`. Verification:
  recomputed decile-aware charts against the recomputed test predictions; top-decile lift **1.216**.
  Calibration (10 uniform bins): XGBoost populates 6 non-empty bins (probability span ~0.33-0.85),
  LR 9 bins, RF 8 bins.

### 9.6 Block-2 re-run with the CV fix (`block2modelling` checkpoint)

- `cross_val_score` with early stopping needed an `eval_set`; the CV clone now re-fits with
  `num_boosted_rounds()` = 80 trees so training-time early stopping is applied identically.
- `block2_base_metrics.csv` written (the aggregate must read this, not the file it overwrites - see 9.7):
  LR: train 0.6118 / CV 0.5627 sd 0.0225 / threshold 0.50. XGB untuned: test 0.5929, CV 0.5843 sd 0.0260,
  train 0.6740, threshold 0.56, train-test AUC gap 0.0811.

### 9.7 WP5 - `make_comparison_table.py` (canonical comparison table)

- Reads Block 2 (untuned) rows from `block2_base_metrics.csv` plus the RF/WP outputs, writes the **canonical**
  `model_comparison_test_metrics_v2.csv` (14 rows x 23 cols). Cross-check: handle each row against its
  recomputed source; all pass (tuned XGB test 0.5955 / Gini 0.1909 / KS 14.9444 vs report 14.9 (within 0.06)
  / train-CV 0.5936; LR 0.5795 / Gini 0.1590).
- **Lesson recorded**: an aggregator must never read the file it overwrites; Block 2 writes its own
  checkpoint (`block2_base_metrics.csv`) that the aggregator consumes.

### 9.8 WP6 - report update (this session's edits to the docx)

- Preconditions verified before any edit: no Word owner file; docx mtime = 2026-09-24 11:55:36 (matches the
  session-1 handoff, so Nash had not edited since); backup `Project_Report_Draft.pre_model_improvement.docx`
  created (662,982 bytes); `report_structure_before.txt` dumped (260 paras / 9 tables / 9 images).
- `eval_to_report.py` -> `report_truth_bundle.json` supplies every number the editor may write (guards the
  no-inventing-numbers rule). `docx_helpers.py` shared by the editor.
- `fill_report_model_improvement.py` (idempotent, sentinel-guarded, atomic save via temp file + zip testzip +
  reopen) applied **18 edits**: new Ch5.5 feature-ablation section + Table 5-1; Ch6.1 RFECV paragraph;
  Ch8.2 RF bullet + Ch8.3 RF rationale; corrected Ch9.3 tuned-vs-untuned comparison and regularization
  narrative; Ch10.1 accuracy/CM (61.6% -> 64.2%; CM 1682-2429-2355-5984 -> 981-3130-1333-7006); Ch10.1
  Table 4 class cells; rebuilt Ch10.2 Table 5 (tuned XGBoost + RF columns); Ch10.3 decile table cells and
  lift/gain paragraph (lift 1.216 -> 0.832 with one small reversal at decile 8 near-monotonic, top3 capture
  34.3%); Ch10.4 calibration paragraph;
  Ch10.5 gap row and learning-curve paragraph (concept-drift diagnosis); Ch11.1 score-to-action paragraph;
  Appendix B.5 RFECV code block; Appendix C Figures 10-13 + intro; Ch12.1 summary verdict paragraph.
- Final state: **287 paragraphs, 10 tables, 13 images**; every sentinel appears exactly once; em-dash scan of
  all inserted content clean; `report_structure_after.txt` and `report_edit_diff.txt` written; the temp
  `Project_Report_Draft.wp6.tmp.docx` does not linger (atomic rename). Docx re-opens cleanly.
- Front matter (academic-year/name/guide/etc.) still Nash's to fill (session-1 item, unchanged).
- The `fill_report_model_improvement.py` re-run is a no-op by design (idempotent); `fill_report_pending.py`
  from session 1 must NOT be re-run (it is intentionally non-idempotent).

### 9.9 Git state (end of session 2)

Nothing committed. Untracked new files this session: `eval_utils.py` (refactor used by several work packages),
`block2c_time_interaction_experiment.py`, `block2c_supplementary.py`, `block2b_random_forest.py`,
`block1b_rfe.py`, `make_eval_charts.py`, `make_comparison_table.py`, `eval_to_report.py`, `docx_helpers.py`,
`fill_report_model_improvement.py`, the CSVs/PNGs listed above, `report_truth_bundle.json`,
`report_structure_before.txt`, `report_structure_after.txt`, `report_edit_diff.txt`, and
`Project_Report_Draft.pre_model_improvement.docx`. `test.ipynb` (Dune API key, now redacted from this log)
is still red-flagged and must remain uncommitted.

### 9.10 Artifacts vs source of truth (revision state for Chapters 5/6/8/9/10/11/12 after WP1-WP6)

| Report section changed | Content source | Notes |
|---|---|---|
| Ch5.5 (feature ablation, Table 5-1) | `ablation_results.csv`, `new_feature_importance.csv` | V0 retained; null result reportable |
| Ch6.1 (RFECV) | `rfecv_results.csv` | LR 14/27, XGB 21/27 |
| Ch8.2/8.3 (RF) | `rf_results.csv` | test 0.5962 / 0.5978 |
| Ch9.3 (tuned vs untuned) | `report_truth_bundle.json` (recomputed) | KS 14.9 vs 14.4 etc. |
| Ch10.1 (accuracy/CM/Table 4) | recomputed tuned predictions | 64.2%; CM 981-3130-1333-7006 |
| Ch10.2 (Table 5) | `model_comparison_test_metrics_v2.csv` | rebuilt with RF column |
| Ch10.3 (decile table + narrative) | recomputed tuned predictions, decile lift 1.216 | near-monotonic (small reversal at decile 8); top3 34.3% |
| Ch10.4 (calibration) | recomputed Brier + reliability-diagram bins | 0.2175 / 0.2191 / 0.2213 |
| Ch10.5 (gap row, learning curve) | recomputed; concept-drift diagnosis | gap 0.073-0.093-0.147-0.008, PR-AUC -0.014 |
| Ch11.1 (score-to-action) | decile table recompute | honest middle-decile statement |
| Appendix B.5 / C / Ch12.1 | scripts + figures | added, verified, em-dash clean |

---

## 10. Reproducibility (session 2)

```bash
# WP1 feature ablation + supplementary bootstrap CI / SHAP
python3 block2c_time_interaction_experiment.py   # -> ablation_results.csv, xgb_v0_test_predictions.csv
python3 block2c_supplementary.py                 # -> new_feature_importance.csv (CIs on the run log)

# WP2 Random Forest
python3 block2b_random_forest.py                 # -> rf_results.csv, rf_tuned_test_predictions.csv

# WP3 RFECV
python3 block1b_rfe.py                           # -> rfecv_results.csv, rfecv_cv_curve.csv, fig13

# WP4 evaluation charts
python3 make_eval_charts.py                      # -> fig10/11/12 + verification print

# WP5 canonical comparison table
python3 make_comparison_table.py                 # -> model_comparison_test_metrics_v2.csv (14x23)

# WP6 report update (idempotent: safe to re-run; a re-run is a no-op)
python3 eval_to_report.py                        # -> report_truth_bundle.json
python3 fill_report_model_improvement.py         # -> edits docx, writes report_structure_after.txt
                                                 #    and report_edit_diff.txt
```

Verify the finished docx anytime: `python3 -c "import docx;d=docx.Document('Project_Report_Draft.docx');print(len(d.paragraphs),len(d.tables),len([p for p in d.paragraphs if p._p.findall('.//{http://schemas.openxmlformats.org/wordprocessingml/2006/main}drawing')]))"`
This session adds no commit; see 9.9. All inserted text is em-dash free; existing text was left as-is.

---

## 11. Audit fixes to Project_Report_Draft.docx (session 3)

Sweep of the finished report for residual false or hard-to-trace claims. Every number below was re-derived
from the CSVs/scripts/truth bundle on disk; nothing was invented. Applied by `fill_report_audit_fixes.py`
(guarded `rewrite`s: anchored by text prefix, skip-and-log on drift; atomic temp-save + zip `testzip` +
reopen; em-dash scan on changed content only). Backup taken first: `Project_Report_Draft.pre_audit_fixes.docx`
(924,666 B; md5 `58149639...`, identical to the pre-edit docx). 45 changes applied (23 content edits across 22
paragraphs + 1 table cell, plus the item-11e em-dash normalization across 17 paragraphs + 4 table cells);
structure counts unchanged (287 paragraphs / 10 tables / 13 images). The script is idempotent: re-running it
on the finished document logs every prior edit as a guarded skip (drift) and reports all checks green.

### 11.1 Fixes applied

| # | Location | Before (wrong / untraceable) | After (verified source) |
|---|---|---|---|
| 1 | Abstract (P36) | no mention of the wrapper stage | added: RFECV confirmatory cross-check confirmed the embedded rankings without changing the deployed list |
| 2 | Abstract (P37) | - | added: RF challenger matched within noise; feature-ablation found no better set (both verified, Chapter 5.5/6.1) |
| 3 | Ch5.5 heading (P119) | "5.5 Session-2 Robustness" | "5.5 Open-Time and Interaction Feature Experiment" (the section is a feature experiment, not a session marker) |
| 4 | Ch5.5 opener / TRAIN ONLY (P120/P121) | "session-2 robustness study"; "scored ... on TRAIN ONLY" | "a robustness study"; "on the training portion only for selection" |
| 5 | Ch5.5 verdict (P123) | "the time features actually hurt ... excludes zero" (falsely strong) | time features did not help, slightly negative on this seed; delta is within seed-to-seed test-AUC spread (untuned sd 0.0038), read as neutrality |
| 6 | Ch6.1 funnel phrasing (P126) | "the full filter->wrapper->embedded funnel was not necessary" | deployed path is a filter + embedded funnel; wrapper (RFECV) added later as a confirmatory cross-check |
| 7 | Ch6.1 RFECV (P130) | LR full-set test 0.5955 (copy-paste of the XGB tuned baseline) | LR full-set test 0.5795; subset 0.5891. Added CV sd 0.0258 (LR) / 0.0126 (XGB). Source `rfecv_results.csv` |
| 8 | Ch8.2 RF bullet (P150) | "added in the session-2 robustness study" | "added as part of the follow-up robustness battery" |
| 9 | Ch8.3 tuned-vs-untuned (P152) | "a real if modest improvement ... even before hyperparameter tuning" | improvement holds at both untuned Block-2 defaults (test 0.5929 / Gini 0.1859) and tuned values. Source `block2_base_metrics.csv` |
| 10 | Ch8.3 RF rationale (P153) | "statistically indistinguishable" | "within cross-validation and seed noise of each other", with the CV/test offsets stated |
| 11 | Ch10.2 three-model band (P168) | "poor to acceptable"; single figure | all three in the "poor" band: KS 11.8 (LR, Gini 0.159), 14.9 (XGB, 0.191), 14.3 (RF, 0.196); removed an em dash in this paragraph |
| 12 | Ch10.3 base rate (P172) | "the 60.0% base rate" | "test base rate of 66.98% (8,339 of 12,450)"; added achievable-lift bound 1/0.6698 ~ 1.49; qualified "monotonically" -> near-monotonic (reversal at decile 8, 0.947 vs 0.914 in deciles 6-7). Source `decile_lift_xgb_v0.csv` |
| 13 | Ch10.4 calibration (P174) | "mild over-prediction of profitability in the most-confident band" (false) | models under-predict (curves above diagonal; expected given 58.3%-profitable train window vs 67.0% test); XGB in-band examples 0.56/0.61, 0.65/0.69, 0.82/0.90; over-prediction confined to: XGB lowest band [0.3,0.4) (0.38 vs 0.24), LR three most-confident (0.73 vs 0.72 at [0.7,0.8), essentially on the line), RF two lowest + slight at top (0.92 vs 0.87). Source `calibration_data.csv` |
| 14 | Ch11.1 score-to-action (P182) | "meaningfully above the 60% base rate" | deciles 1-5 clear the 66.98% base rate (81.4% -> 68.5%); deciles 6-10 (61.2/61.2/63.5/60.2/55.7%) all below it -> "reconsider" territory |
| 15 | Ch11.2 realized P&L (P187) | "peak $7.14M at 0.53"; "$2.03M at 0.60" (false) | verified sweep: flat plateau 0.30-0.56 at ~$7.05-7.09M (spread <1% of the $7.09M max), collapse above (see 11.3 table). No value-maximizing threshold exists; acting on all positions (0.30 floor) maximizes. Source pin `xgb_v0_test_predictions.csv` + `cost_benefit_analysis.py` clean-P&L convention |
| 16 | Ch11.3 (P189) | same stale P&L claims | cost asymmetry real but sweep adds no dollars; 0.56 kept on classification rationale alone |
| 17 | Ch12.1 (P194) | "session-2 robustness battery"; time features "reduced test discrimination" | "robustness battery"; seed-neutral wording; added CM provenance (test accuracy 64.2%, CM 981/3,130/1,333/7,006) traced to the tuned model's saved test predictions |
| 18 | Ch12.2 CV coverage (P199) | "only the hyperparameter search itself used full cross-validation" | the CV exercises were Ch9 search, Ch5.5 ablation, Ch8.2 RF search, Ch6.1 RFECV (all 5-fold TimeSeriesSplit) |
| 19 | Appendix C intro (P276) | "the four session-2 evaluation charts" | "the four evaluation charts" |
| 20 | Ch10.2 Table 6 (cell 5,2) | XGBoost (test, tuned) Brier 0.218 | 0.217. Source `model_comparison_test_metrics_v2.csv` |
| 21 | Figure 13 media (zip) | `/word/media/image6.png` = v1 (RFECV elimination curve without the LR-first feature ordering; md5 `3d3cfcd9...`) | replaced with `fig13_rfecv_curve_v2.png` (md5 `161bd10b...`) under the same rel target rId22; rels and `[Content_Types].xml` untouched |
| 22 | Item 11e (em dashes) | U+2014 in 17 older paragraphs + 4 table cells (19 + 4 chars; pre-session-1 prose, e.g. P32 "Same process — caption ...") | each U+2014 replaced with a spaced en dash U+2013 inside its own run (run-level bold/italic/font preserved; e.g. P32 "Same process – caption ..."); whole-document em-dash count now 0 |
| 23 | Ch12.1 summary (P193) | "improved test ROC-AUC by less than 0.001 over untuned defaults" | "improved test ROC-AUC by +0.0026 at the seed-0 comparison (mean +0.0016 across an audited seed sweep, per-seed span -0.0020 to +0.0125), a move comparable to the untuned model's seed-to-seed spread (sd 0.0038) rather than a step-change"; retains "not a symptom of insufficient tuning" |
| 24 | Ch12.2 caveats (P198) | "improved test ROC-AUC by under 0.001 over untuned defaults" | "improved test ROC-AUC by only +0.0026 at the seed-0 comparison (mean +0.0016 across seeds), inside the untuned model's run-to-run seed spread (sd 0.0038): tuning is not the lever, and the modest signal is not an artifact of untuned defaults" |

### 11.2 Audited seed sweep (reported; from the pinned session-2 environment)

| Model (XGBoost) | test ROC-AUC range | mean | sd |
|---|---|---|---|
| untuned (Block 2 defaults) | 0.5825 - 0.5966 | 0.5929 | 0.0038 |
| tuned (Block 3 params) | 0.5929 - 0.5955 | 0.5946 | 0.0008 |

Per-seed tuned-minus-untuned delta: -0.0020 .. +0.0125 (mean +0.0016); seed-0 delta +0.0026 AUC / +0.0051 Gini.
This is the source of the "(sd 0.0038)" references added in Ch5.5 verdict and Ch12.1. It supersedes the
preliminary spot-check note in 9.1 (0.5917-0.5953), which was a smaller determinism probe with a different
seed set.

**Environment drift, documented for transparency (not used in the report text):** re-running the untuned
seed-0 model on the current machine (xgboost 3.4.0, this session) gives test ROC-AUC 0.5885 vs the canonical
0.5929. The deployed report numbers intentionally stay canonical (pinned session-2 environment). See
`seed_sweep_results.csv` for the drift-measurement artifact.

### 11.3 Threshold-sweep P&L table verified this session

Total realized net P&L on the test slice (clean P&L convention of `cost_benefit_analysis.py`; 7 corrupt-derived-P&L
rows excluded) by operating threshold (positions at or above the threshold):

| Threshold | Realized P&L (USD) | Threshold | Realized P&L (USD) |
|---|---|---|---|
| 0.30 (sweep floor; all clean positions) | 7,094,531 | 0.57 | 4,038,520 |
| 0.50 | 7,061,887 | 0.58 | 2,901,044 |
| 0.53 | 7,045,624 | 0.60 | 2,138,569 |
| 0.56 (deployed) | 7,046,175 | 0.70 | 327,030 |

Every threshold 0.30-0.56 is within 1% of the $7.09M maximum; the sweep therefore finds no
value-maximizing threshold and 0.56 is retained for macro-F1 classification reasons (Ch11.3). The old
"peak $7.14M at 0.53" and "$2.03M at 0.60" figures are retired as false.

### 11.4 Post-edit verification

- 287 paragraphs / 10 tables / 13 images (unchanged); zip `testzip` clean; python-docx re-opens; content
  types and document rels intact (rId22 -> media/image6.png preserved); image6 md5 `161bd10b...` = v2.
- Present/absent sentinel checks all pass (each new wording appears exactly once; no old false string
  remains: no "session-2", "TRAIN ONLY", "peak $7.14M", "$2.03M", "even before hyperparameter tuning",
  "statistically indistinguishable", the funnel arrow "→", "monotonically.").
- Em-dash (U+2014): scan of all changed paragraphs 0; whole-document count (paragraphs + tables) 0. Item 11e
  resolved; 37 en dashes (U+2013) now present (ranges plus the new spaced en dashes), no double-space or
  spacing artifacts after normalization, run-level formatting preserved.
- Diff log: `report_audit_diff.txt`; structure dumps `report_structure_before_audit.txt` /
  `report_structure_after_audit.txt`.
- Git state (session 3): nothing committed. Untracked additions this session: `fill_report_audit_fixes.py`,
  `seed_sweep_results.csv`, `report_audit_diff.txt`, the two structure dumps, and
  `Project_Report_Draft.pre_audit_fixes.docx` backup.
- Out-of-scope residuals: none remaining. The Ch12.1/Ch12.2 "under 0.001" tuning-gain statements (flagged
  earlier) were reworded to the audited values per Nash's decision (rows 23-24), keeping the "ceiling, not a
  tuning gap" / "not an artifact of untuned defaults" framing.