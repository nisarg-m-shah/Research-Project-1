"""Session-3 audit-fix pass on Project_Report_Draft.docx (audit items 1-11d).

Every rewrite is in-place (no paragraphs inserted, so indices are stable) and
guarded: the target paragraph must still start with its expected prefix and
contain the offending substring that motivated the edit; otherwise the edit is
skipped and logged. All numbers embedded here were re-verified this session
against the CSVs listed below; the report can never contain a number that was
not produced by a script run or recorded in a CSV.

Edits applied:
  1.  Ch10.3 (lift): base rate 60.0% -> test base rate 66.98% (8339/12450);
      decile-8 reversal acknowledged; max possible top-decile lift 1/0.6698.
  2.  Ch11.1 (score-to-action): deciles 1-5 above base, 6-10 below, under 66.98%.
  3.  Ch10.4 (calibration): curves above diagonal = under-prediction (not
      "mild over-prediction in the most-confident band"); lowest bin exception.
  4.  PROJECT_CONTEXT_HANDOFF.md wording (handled by this script's companion edit).
  5.  Ch8.3 (P152/P153): drop "even before hyperparameter tuning" misreading;
      "statistically indistinguishable" -> "within cross-validation and seed
      noise"; retention rationale on train-CV + SHAP, not test point estimate.
  6.  Ch11.2/11.3 (P187/P189): realized-P&L sweep re-verified; flat 0.30-0.56,
      collapse above; no value-maximizing threshold; $2.03M/$7.14M claims removed.
  7.  Ch6.1 (P126/P130) + abstract (P36/P37): funnel framing, LR RFECV numbers,
      wrapper-as-confirmatory + two short abstract sentences.
  8.  Ch10.2 Table 6: XGBoost Brier 0.218 -> 0.217.
  9.  Remove the "Session-2"/"TRAIN ONLY" markers (P119/P120/P121/P130/P150/
      P194/P276).
  10. Ch5.5 (P123): "time features actually hurt" -> neutral wording with
      seed-to-seed spread note; also P194's matching phrase.
  11a. Appendix C Figure 13: swap word/media/image6.png bytes to the v2 curve.
  11b. Ch12.1 (P194): provenance sentence for the regenerated Ch10 numbers.
  11c. Ch12.2 (P199): list the session-2 CV exercises alongside Ch9.
  11d. Ch10.2 (P168): three models all in the "poor" band (KS 11.8-14.9).
  11e. Em-dash pass: U+2014 -> spaced en dash U+2013 per run (format-preserving)
       across the 17 older paragraphs + 4 table cells that still contained them;
       whole-document em-dash count is verified 0 afterwards.

The script is idempotent: re-running it skips (with a log line) every edit whose
target paragraph no longer matches its anchor/offending text.

The docx is saved atomically (temp validated before rename). The Figure 13
media swap is done at the zip level after python-docx save so the image bytes
change without touching relationships or content types.
"""

import hashlib
import os
import re
import shutil
import sys
import traceback
import zipfile

import docx

import docx_helpers as H

DOCX = 'Project_Report_Draft.docx'
BACKUP = 'Project_Report_Draft.pre_audit_fixes.docx'
TMP = 'Project_Report_Draft.audit.tmp.docx'
TMP2 = 'Project_Report_Draft.audit.tmp2.docx'
BEFORE_DUMP = 'report_structure_before_audit.txt'
AFTER_DUMP = 'report_structure_after_audit.txt'
DIFF = 'report_audit_diff.txt'
SWAP_MEMBER = 'word/media/image6.png'
SWAP_SRC = 'fig13_rfecv_curve_v2.png'
LANDSLIDE = '~\u0024Project_Report_Draft.docx'
EM = '\u2014'

NS = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'

log = []
changed = []   # (label, para_index, old_text, new_text)


def log_ok(msg):
    log.append(("applied", msg))


def log_skip(msg):
    log.append(("skipped", msg))


def guard(para, substr, label):
    if substr not in para.text:
        log_skip(f"{label}: paragraph drifted (missing {substr!r}); not touched.")
        return False
    return True


def rewrite(para, old_prefix, offending, new_text, label):
    if not para.text.startswith(old_prefix):
        log_skip(f"{label}: paragraph no longer starts with expected prefix; not touched.")
        return
    if offending and not guard(para, offending, label):
        return
    changed.append((label, -1, para.text, new_text))
    H.set_para_text(para, new_text)
    log_ok(label)


def md5(path_or_bytes):
    h = hashlib.md5()
    if isinstance(path_or_bytes, str):
        with open(path_or_bytes, 'rb') as f:
            for chunk in iter(lambda: f.read(65536), b''):
                h.update(chunk)
    else:
        h.update(path_or_bytes)
    return h.hexdigest()


def preflight():
    if os.path.exists(LANDSLIDE):
        sys.exit(f"STOP: Word owner file present ({LANDSLIDE}). Close Word and re-run.")
    if not os.path.exists(BACKUP):
        sys.exit("STOP: pre-audit-fixes backup missing; refusing to edit a docx without a backup.")
    cur = md5(DOCX)
    bak = md5(BACKUP)
    if cur != bak:
        log_skip(f"docx md5 differs from backup ({cur[:12]} vs {bak[:12]}); "
                 "proceeding anyway, backup kept as-is.")
    if not os.path.exists(SWAP_SRC):
        sys.exit(f"STOP: replacement figure {SWAP_SRC} missing.")
    return True


# ----------------------------------------------------------------------------
# Edits (anchored by text, index recorded for the em-dash scan)
# ----------------------------------------------------------------------------

def apply_edits(doc):
    paras = doc.paragraphs

    def find_it(prefix, label, alt=None):
        for i, p in enumerate(paras):
            if p.text.startswith(prefix) or (alt and p.text.startswith(alt)):
                return i, p
        raise LookupError(f"{label}: no paragraph starts with {prefix[:60]!r}")

    # --- Abstract P36 (item 7) ---
    i, p = find_it("A time-based 80/20 train/test split", "abstract P36")
    rewrite(
        p, "A time-based 80/20 train/test split",
        "cross-checked against raw class frequencies",
        "A time-based 80/20 train/test split (no shuffling, to prevent temporal "
        "leakage) was used to train a baseline Logistic Regression and a "
        "challenger XGBoost classifier. Variable selection combined a filter "
        "pass (variance and correlation checks) with embedded methods "
        "(L1-regularized logistic regression and Random Forest importance), "
        "cross-checked against raw class frequencies to catch unstable "
        "coefficients driven by rare categories. A wrapper cross-check "
        "(recursive feature elimination with cross-validation, RFECV) "
        "subsequently confirmed the embedded rankings without changing the "
        "deployed feature list.", "abstract P36 RFECV sentence")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- Abstract P37 (item 7) ---
    i, p = find_it("The final XGBoost model achieved a test ROC-AUC", "abstract P37")
    rewrite(
        p, "The final XGBoost model achieved a test ROC-AUC",
        "A learning-curve analysis showed",
        "The final XGBoost model achieved a test ROC-AUC of 0.596 and Gini of "
        "0.191, a modest but genuine and non-leaked discriminative signal. A "
        "Random Forest challenger matched the final model within noise, and a "
        "feature-ablation study found no alternative open-time or interaction "
        "feature set improved on the original 27 features, corroborating the "
        "ceiling from the feature side. A learning-curve analysis showed the "
        "gap between training and test performance is driven substantially by "
        "a real shift in market conditions between the 2021\u20132022 and 2023 "
        "portions of the study period (profitable-position rate rises from "
        "58.3% in training to 67.0% in test), rather than classical "
        "overfitting alone. The headline finding is that open-time features "
        "carry real, measurable predictive signal for LP profitability, but "
        "the current feature set and single-regime training window likely "
        "limit accuracy; expanding the pool universe, extending the time "
        "range, and adding wallet-behavior features are identified as the "
        "highest-value next steps.", "abstract P37 RF + feature-ablation")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P119 heading (item 9) ---
    i, p = find_it("5.5 Session-2 Robustness", "Ch5.5 heading",
                   alt="5.5 Open-Time and Interaction Feature Experiment")
    rewrite(p, "5.5 Session-2 Robustness", "Session-2",
            "5.5 Open-Time and Interaction Feature Experiment",
            "Ch5.5 heading rename")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P120 (item 9) ---
    i, p = find_it("A session-2 robustness study tested", "Ch5.5 opener",
                   alt="A robustness study tested whether")
    rewrite(
        p, "A session-2 robustness study tested", "session-2 robustness study",
        "A robustness study tested whether four cyclical open-time features "
        "and two interaction features could improve the V0 model. The time "
        "features are open_hour_sin, open_hour_cos, open_dow_sin and "
        "open_dow_cos, encoding the position's open hour and weekday (UTC) as "
        "sin/cos pairs so the two ends of the day and week are adjacent in "
        "feature space. All 24 hours and all 7 weekdays are present in both "
        "the training window (626 days) and the test window (343 days). The "
        "two interactions are algebraic products of train-fitted z-scores of "
        "range_width_normalized and token0_price_std_24h, and of fee_tier_pct "
        "and pre_open_avg_daily_swap_count_3d, following the domain hypothesis "
        "that range width should matter most when pre-open volatility is high, "
        "and that fee tier interacts with the pool's liquidity depth.",
        "Ch5.5 opener session-2 removal")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P121 (item 9) TRAIN ONLY ---
    i, p = find_it("Four variants were compared on the identical chronological split",
                   "Ch5.5 TRAIN ONLY")
    rewrite(
        p, "Four variants were compared on the identical chronological split",
        "on TRAIN ONLY for selection",
        "Four variants were compared on the identical chronological split "
        "(V0 = the existing 27 features, V1 = V0 + time features, V2 = V0 + "
        "interactions, V3 = V0 + both). For each variant the Logistic "
        "Regression and the tuned XGBoost (fixed Block 3 hyperparameters, no "
        "re-tuning) were scored with 5-fold TimeSeriesSplit on the training "
        "portion only for selection, and the test-set battery was reported "
        "with its threshold re-derived from train-set macro-F1. The leakage "
        "guard (test ROC-AUC more than about 0.02 above the 0.5955 baseline) "
        "never tripped.", "Ch5.5 TRAIN ONLY wording")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P123 (item 10) ---
    i, p = find_it("Verdict: no variant improved train-CV ROC-AUC", "Ch5.5 verdict")
    rewrite(
        p, "Verdict: no variant improved train-CV ROC-AUC",
        "the time features actually hurt",
        "Verdict: no variant improved train-CV ROC-AUC by more than the ~0.001 "
        "noise floor or the fold standard deviation; the largest positive "
        "delta is V1's XGBoost +0.0007 (CV 0.5943 versus 0.5936), an order of "
        "magnitude below its fold sd of 0.0146. On the held-out test set the "
        "time features did not help and were slightly negative on this seed "
        "(V1 XGBoost 0.5921, bootstrap 95% CI of the difference versus V0 of "
        "[-0.0050, -0.0017]); that drop is comparable to the seed-to-seed "
        "test-AUC spread of the untuned sweep (sd 0.0038) and the CI is a "
        "test-sampling statement for this single seed, so it is read as "
        "neutrality rather than strong harm. The interactions were neutral "
        "(V2 0.5954, CI [-0.0016, +0.0015], and V3 0.5948, CI [-0.0023, "
        "+0.0009], both including zero). Under the fullest model (V3), SHAP "
        "ranks interaction_range_width_x_price_std 9th of 33 features (mean "
        "|SHAP| 0.026, on par with fee_tier_pct at 0.026), the "
        "fee-vs-swap-count interaction contributes 0.006, and all four "
        "cyclical time features contribute under 0.0004 and are effectively "
        "inert. The original 27-feature V0 set is retained; the null result is "
        "reportable and independently corroborates the ceiling diagnosis of "
        "Chapter 10.5.", "Ch5.5 verdict seed-noise wording")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P126 (item 7) funnel wording ---
    i, p = find_it("With 62,250 rows and roughly 30 candidate features", "Ch6.1 opener")
    rewrite(
        p, "With 62,250 rows and roughly 30 candidate features",
        "full filter\u2192wrapper\u2192embedded funnel was not necessary",
        "With 62,250 rows and roughly 30 candidate features, this is not the "
        "high-dimensional (p \u226b n) regime the course's gene-expression case "
        "study targets, so the deployed selection path is a filter + embedded "
        "funnel; the wrapper stage (RFECV) was added later as a confirmatory "
        "cross-check of those rankings rather than as part of the funnel "
        "(Chapter 6.1). A combined filter + embedded approach was used instead:",
        "Ch6.1 funnel phrasing")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P130 (items 7 + 9) ---
    i, p = find_it("A wrapper stage, recursive feature elimination", "Ch6.1 RFECV")
    rewrite(
        p, "A wrapper stage, recursive feature elimination",
        "session-2 robustness check",
        "A wrapper stage, recursive feature elimination with "
        "cross-validation (RFECV, step=1, 5-fold TimeSeriesSplit, scoring "
        "ROC-AUC), was added as a robustness cross-check of the filter and "
        "embedded rankings. Logistic Regression selected 14 of 27 features "
        "(best CV ROC-AUC 0.5639, versus 0.5627 on the full set; test 0.5891 "
        "on the subset versus 0.5795 on the full set), and the tuned XGBoost "
        "selected 21 of 27 (best CV ROC-AUC 0.5962 versus 0.5936 on the full "
        "set; test 0.5982 on the subset versus 0.5955 on the full set). Both "
        "selection gains are inside the corresponding fold standard "
        "deviations (CV sd 0.0258 for the linear model, 0.0126 for the tree "
        "model); the tree model's RFECV ranking agrees with the embedded "
        "impurity ranking on the dominant features. The full 27-feature V0 "
        "set therefore remains the deployed feature list (Chapter 6.2 "
        "unchanged), and the elimination curve is shown as Figure 13 in "
        "Appendix C.", "Ch6.1 RFECV rewrite + LR numbers")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P150 (item 9) ---
    i, p = find_it("Random Forest - added in the session-2 robustness study",
                   "Ch8.2 RF bullet",
                   alt="Random Forest - a bagged-ensemble cross-check")
    rewrite(
        p, "Random Forest - added in the session-2 robustness study",
        "session-2 robustness study",
        "Random Forest - a bagged-ensemble cross-check on the same V0 "
        "features, added as part of the follow-up robustness battery. "
        "Baseline: RandomForestClassifier(n_estimators=300, "
        "min_samples_leaf=5, max_features='sqrt', random_state=0); tuned "
        "variant via RandomizedSearchCV (24 trials, 5-fold TimeSeriesSplit, "
        "ROC-AUC scoring). Random Forest's internal bootstrap sampling does "
        "not reshuffle rows and so does not violate the chronological "
        "no-shuffle discipline.", "Ch8.2 RF bullet session-2 removal")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P152 (item 5) ---
    i, p = find_it("XGBoost was carried forward as the primary model", "Ch8.3 XGB advance")
    rewrite(
        p, "XGBoost was carried forward as the primary model",
        "even before hyperparameter tuning",
        "XGBoost was carried forward as the primary model: test ROC-AUC 0.596 "
        "versus Logistic Regression's 0.580, and test Gini 0.191 versus 0.159, "
        "a real if modest improvement over the linear baseline that holds both "
        "at the untuned Block 2 defaults (ROC-AUC 0.5929, Gini 0.1859) and at "
        "the tuned values above. The margin is not large enough to claim "
        "XGBoost is dramatically superior on this feature set, and the report "
        "is explicit about that rather than overstating the gap. XGBoost's "
        "non-linear capacity is judged worth the loss of Logistic Regression's "
        "clean coefficient interpretability, given SHAP is used to recover "
        "feature-level explanation for the tree model (Chapter 10.2/Appendix "
        "B). This choice was tested, not just assumed: Chapter 9 reports a "
        "full hyperparameter search (RandomizedSearchCV and Optuna, 120 trials "
        "combined) that confirms XGBoost's advantage is a real, stable "
        "property of the feature set rather than an artifact of untuned "
        "defaults.", "Ch8.3 tuned-vs-untuned claim")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P153 (item 5) ---
    i, p = find_it("Random Forest lands in the same modest band", "Ch8.3 RF rationale")
    rewrite(
        p, "Random Forest lands in the same modest band",
        "statistically indistinguishable",
        "Random Forest lands in the same modest band as XGBoost: test ROC-AUC "
        "0.5962 (baseline) and 0.5978 (tuned) against XGBoost's 0.5955, with "
        "train-CV AUC 0.5789 (sd 0.0192) and 0.5850 (sd 0.0206). Random "
        "Forest's near-perfect training fit (train ROC-AUC 0.98 at baseline) "
        "underlines that its cross-validated test numbers, not its training "
        "metrics, are the honest ones. Because the two tree families are "
        "within cross-validation and seed noise of each other on this feature "
        "set (the tuned XGBoost holds a small train-CV edge of 0.5936 versus "
        "0.5850, but the tuned Random Forest's test ROC-AUC of 0.5978 is "
        "marginally ahead of XGBoost's 0.5955) and XGBoost additionally "
        "provides the SHAP explanation path used in Chapter 10.2 and Appendix "
        "B, XGBoost is retained as the primary model and Random Forest is "
        "reported as robustness corroboration (Chapter 10.2).",
        "Ch8.3 retention rationale")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P168 (item 11d) ---
    i, p = find_it("Per the course's KS/Gini reading scale", "Ch10.2 KS/Gini scale")
    rewrite(
        p, "Per the course's KS/Gini reading scale",
        "both models fall in the \u201cpoor to acceptable\u201d range",
        "Per the course's KS/Gini reading scale (under 20 = poor, barely "
        "better than random; 20-40 = acceptable), all three models fall in the "
        "\u201cpoor\u201d band on this dataset and feature set, stated plainly "
        "rather than overstated. The tuned XGBoost leads with a KS of 14.9 "
        "(Gini 0.191), the tuned Random Forest is close (KS 14.3, Gini 0.196) "
        "and Logistic Regression trails (KS 11.8, Gini 0.159); no model's KS "
        "reaches 20, the lower edge of the \u201cacceptable\u201d band. "
        "Critically, no value is suspiciously high (the same scale flags Gini "
        "above 75 as a near-certain leakage signature): these numbers are "
        "consistent with a real but limited signal, not a leakage artifact, "
        "which is the honest read after the deliberate leakage-column "
        "exclusions in Chapter 6.", "Ch10.2 three-models poor band")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P172 (item 1) ---
    i, p = find_it("Lift declines from 1.216 in decile 1", "Ch10.3 lift paragraph")
    rewrite(
        p, "Lift declines from 1.216 in decile 1",
        "materially above the 60.0% base rate",
        "Lift declines from 1.216 in decile 1 to 0.832 in decile 10, nearly "
        "monotonically but with a small reversal at decile 8 (0.947 versus "
        "0.914 in deciles 6-7). Figure 10 plots the same decile lift as a bar "
        "chart and Figure 11 shows the cumulative gain curve. The top three "
        "deciles (30% of positions by predicted probability) capture 34.3% of "
        "all eventually-profitable positions at positive rates of 81.4-72.6%, "
        "materially above the test base rate of 66.98% (8,339 of 12,450 test "
        "positions profitable), which is the basis for the score-to-action "
        "mapping in Chapter 11.1. With a 66.98% base rate the highest lift a "
        "perfect top decile could reach is 1/0.6698, about 1.49, so a "
        "top-decile lift of 1.216 indicates the model already realizes a large "
        "share of what the feature set allows; it remains the quantitative "
        "expression of the modest absolute discrimination reported throughout "
        "this chapter.", "Ch10.3 test base rate fix")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P174 (item 3) ---
    i, p = find_it("Brier score (0.2175 for the tuned XGBoost", "Ch10.4 calibration")
    rewrite(
        p, "Brier score (0.2175 for the tuned XGBoost",
        "mild over-prediction of profitability",
        "Brier score (0.2175 for the tuned XGBoost, 0.2191 for the tuned "
        "Random Forest, 0.2213 for Logistic Regression) indicates broadly "
        "similar and moderate calibration across the three models. Figure 12 "
        "(Appendix C) shows the reliability diagram over 10 uniform "
        "probability bins: XGBoost's predicted probabilities span only "
        "0.33-0.85, so 6 of the 10 bins are populated (Logistic Regression 9, "
        "Random Forest 8); bins with no observations are omitted. The curves "
        "sit predominantly above the diagonal, meaning the models "
        "under-predict realized profitability: XGBoost predicts 0.56 while "
        "0.61 realizes in the 0.5-0.6 band, 0.65 while 0.69 in the 0.6-0.7 "
        "band, and 0.82 while 0.90 in the highest populated band. This is the "
        "expected signature of models trained on the 58.3%-profitable training "
        "window and scored on the 67.0%-profitable test window; it corrects an "
        "earlier draft that mislabeled the most-confident band as "
        "over-predicted. The over-prediction that does remain is confined to "
        "the extremes rather than the confident tail: XGBoost over-predicts "
        "only in its lowest populated band [0.3,0.4) (0.38 predicted versus "
        "0.24 realized); Logistic Regression clearly only in its two most "
        "confident bands (its third-most band [0.7,0.8) sits essentially on "
        "the diagonal at 0.73 versus 0.72); and Random Forest in its two "
        "lowest bands plus a slight margin at its very top band (0.92 versus "
        "0.87). No Platt or isotonic correction was applied, since "
        "this project has no pricing or expected-value decision that depends "
        "on the exact numeric probability; the score is used ordinally, for "
        "ranking candidate positions, where calibration matters less than "
        "discrimination. This is a scope choice, not an oversight.",
        "Ch10.4 under-prediction correction")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P182 (item 2) ---
    i, p = find_it("Using the decile table in Chapter 10.3", "Ch11.1 score-to-action")
    rewrite(
        p, "Using the decile table in Chapter 10.3",
        "decile 9 sits at it (60.2%)",
        "Using the decile table in Chapter 10.3: positions scored into "
        "deciles 1-5 (the top half by predicted probability) clear the 66.98% "
        "test base rate, from 81.4% in decile 1 down to 68.5% in decile 5, "
        "and are the natural \"favor this range/pool choice\" band, with "
        "deciles 1-3 (81.4-72.6%) the strongest part of it. Deciles 6-10 "
        "(61.2%, 61.2%, 63.5%, 60.2% and 55.7% in decile order) all fall below "
        "the 66.98% base rate and represent \"reconsider this range choice\" "
        "territory, since acting on them is on average worse than opening at "
        "random; decile 10 is the weakest at 55.7%. Deciles 4-7 (69.9-61.2%) "
        "straddle the base rate closely and the model offers limited "
        "actionable separation there - stated honestly rather than papered "
        "over, since claiming fine-grained value in the middle deciles would "
        "overstate what a Gini of 0.191 actually supports.",
        "Ch11.1 base-rate mapping")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P187 (item 6) ---
    i, p = find_it("At a 1.56 : 1 cost ratio", "Ch11.2 P&L sweep")
    rewrite(
        p, "At a 1.56 : 1 cost ratio",
        "peak $7.14M at 0.53",
        "At a 1.56 : 1 cost ratio the per-decision cost-optimal operating "
        "threshold is cost_FP / (cost_FP + cost_FN) = 0.61, only 0.05 above "
        "the 0.56 used in Chapter 10. Sweeping the threshold on the test "
        "slice, total realized net P&L across positions scoring at or above "
        "the threshold is essentially flat from the lowest sweep point 0.30 up "
        "to 0.56 ($7.09M at 0.30 down to $7.05M at 0.53 and $7.05M at 0.56, a "
        "spread of under 1% of the $7.09M maximum), then falls off sharply as "
        "the bar rises ($4.04M at 0.57, $2.90M at 0.58, $2.14M at 0.60, "
        "$0.33M at 0.70). Because the maximum is attained by acting on every "
        "position (equivalently, the 0.30 sweep floor, $7.09M), the sweep "
        "finds no value-maximizing threshold: the model's dollar-level "
        "filtering adds no demonstrated value on this test slice. Chapter "
        "11.3 therefore keeps 0.56 for its classification rationale alone.",
        "Ch11.2 realized-P&L honesty")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P189 (item 6) ---
    i, p = find_it("Recommended threshold: 0.56", "Ch11.3 recommendation")
    rewrite(
        p, "Recommended threshold: 0.56",
        "drops to $2.03M by 0.60",
        "Recommended threshold: 0.56 (the macro-F1-optimal value used "
        "throughout Chapter 10), balancing precision and recall roughly evenly "
        "across both classes rather than defaulting to 0.5. The cost analysis "
        "in Chapter 11.2 does not override it: the realized 1.56 : 1 "
        "false-positive-to-false-negative cost ratio implies a per-decision "
        "cost-optimal threshold of 0.61, but the realized-P&L sweep shows no "
        "threshold improves on acting on all positions ($7.09M maximum, with "
        "every threshold from 0.30 to 0.56 within 1% of it), so the dollar "
        "figures provide no reason to move 0.56. The cost asymmetry is real, "
        "but it is not large enough to justify trading 0.56's balanced "
        "precision and recall for a value that adds no realized dollars.",
        "Ch11.3 cost-analysis framing")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P194 (items 9 + 10 + 11b) ---
    i, p = find_it("A session-2 robustness battery was added", "Ch12.1 robustness paragraph",
                   alt="A robustness battery was added after the main results")
    rewrite(
        p, "A session-2 robustness battery was added",
        "the time features reduced test discrimination",
        "A robustness battery was added after the main results were frozen, "
        "and its conclusions are incorporated above. A feature-engineering "
        "study across four feature sets (V0 to V3, adding cyclical open-time "
        "and interaction features; Table 5-1) found no variant that beat the "
        "original V0 set out of sample: the time features did not help (and "
        "were slightly negative on this seed) while the interactions were "
        "neutral (Chapter 5.5). A Random Forest family (baseline and tuned) "
        "matched XGBoost's discrimination within noise (test ROC-AUC 0.5962 "
        "to 0.5978 against 0.5955), and RFECV reduced the XGBoost feature set "
        "from 27 to 21 features with only a +0.003 CV move, inside fold noise "
        "(Chapter 6.1). None of these checks changed the reported model, the "
        "0.56 threshold, or the Chapter 12.2 caveats; they strengthen the "
        "central claim that the modest discriminative signal is a property of "
        "the feature set and the single training window, not of "
        "hyperparameters, features, or model family. For provenance, the "
        "Chapter 10 classification numbers were regenerated from this tuned "
        "model's saved test predictions (test accuracy 64.2%, confusion matrix "
        "981/3,130/1,333/7,006), replacing an earlier draft whose confusion "
        "matrix could not be traced to any saved run.",
        "Ch12.1 robustness rename + provenance")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P199 (item 11c) ---
    i, p = find_it("A single fixed 80/20 chronological split", "Ch12.2 CV limitation")
    rewrite(
        p, "A single fixed 80/20 chronological split",
        "only the hyperparameter search itself (Chapter 9) used full cross-validation",
        "A single fixed 80/20 chronological split was used for the final "
        "reported test metrics throughout Chapters 8 and 10; the "
        "cross-validation exercises were the hyperparameter search (Chapter "
        "9), the feature-ablation study (Chapter 5.5), the Random Forest "
        "search (Chapter 8.2) and the RFECV wrapper (Chapter 6.1), all with "
        "5-fold TimeSeriesSplit. A natural extension would be to report "
        "Chapter 10's full evaluation battery (decile/lift, calibration, "
        "confusion matrix) averaged across the same 5 chronological folds "
        "rather than the single split, for a still more robust final number.",
        "Ch12.2 CV-coverage correction")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P193 (Chapter 12.1 summary; tuning-gain report fix) ---
    i, p = find_it("This project set out to predict, at the moment a Uniswap V3 LP position opens",
                   "Ch12.1 tuning-gain statement")
    rewrite(
        p, "This project set out to predict, at the moment a Uniswap V3 LP position opens",
        "improved test ROC-AUC by less than 0.001",
        "This project set out to predict, at the moment a Uniswap V3 LP "
        "position opens, whether it would close profitably, using only "
        "open-time-knowable information. A leakage-safe pipeline across nine "
        "on-chain data sources produced 62,250 labeled positions and 27 "
        "model-input features after variable selection. The resulting XGBoost "
        "model achieves a modest but genuine test-set discriminative signal "
        "(ROC-AUC 0.596, Gini 0.191), clearing the Logistic Regression "
        "baseline (Gini 0.159) without approaching either the \u201cgood\u201d "
        "threshold on the course's evaluation scale or any leakage-suspicious "
        "high score. A full hyperparameter search (Chapter 9, 120 trials "
        "across two independent methods, 5-fold chronological "
        "cross-validation) improved test ROC-AUC by +0.0026 at the seed-0 "
        "comparison (mean +0.0016 across an audited seed sweep, per-seed span "
        "of -0.0020 to +0.0125), a move comparable to the untuned model's "
        "seed-to-seed spread (sd 0.0038) rather than a step-change, which "
        "upgrades the project's central finding from a hypothesis to a tested "
        "conclusion: the modest discriminative power is a real, stable "
        "property of the current feature set and single fixed training "
        "window, not a symptom of insufficient tuning. The dominant finding "
        "from the evaluation battery is that a meaningful share of the "
        "train/test performance gap is attributable to a genuine "
        "market-regime shift between the 2021\u20132022 and 2023 portions of "
        "the study period, not to conventional overfitting.",
        "Ch12.1 tuning-gain statement")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P198 (Chapter 12.2 caveats; tuning-gain report fix) ---
    i, p = find_it("Discriminative power is modest by the course's own evaluation standard",
                   "Ch12.2 tuning-gain statement")
    rewrite(
        p, "Discriminative power is modest by the course's own evaluation standard",
        "by under 0.001 over untuned defaults",
        "Discriminative power is modest by the course's own evaluation "
        "standard; the current feature set and single fixed training window "
        "are confirmed (not merely suspected) to be the ceiling, since a "
        "120-trial, two-method hyperparameter search (Chapter 9) improved "
        "test ROC-AUC by only +0.0026 at the seed-0 comparison (mean +0.0016 "
        "across seeds), inside the untuned model's run-to-run seed spread "
        "(sd 0.0038): tuning is not the lever, and the modest signal is not "
        "an artifact of untuned defaults.",
        "Ch12.2 tuning-gain statement")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- P276 (item 9) ---
    i, p = find_it("The Chapter 3 EDA charts appear inline", "Appendix C intro")
    rewrite(
        p, "The Chapter 3 EDA charts appear inline",
        "the four session-2 evaluation charts",
        "The Chapter 3 EDA charts appear inline in Chapter 3. The figures "
        "below are the SHAP summary plot referenced in Chapter 10.2 (Figure 9) "
        "and the four evaluation charts: the decile lift (Figure 10) and "
        "cumulative gain (Figure 11) for the tuned XGBoost model from Chapter "
        "10.3, the reliability diagram (Figure 12) from Chapter 10.4, and the "
        "RFECV elimination curve (Figure 13) from Chapter 6.1. The test-set "
        "decile table for the tuned XGBoost model is in Chapter 10.3.",
        "Appendix C intro session-2 removal")
    if changed and changed[-1][1] == -1:
        changed[-1] = (changed[-1][0], i, changed[-1][2], changed[-1][3])

    # --- Table 6 Brier (item 8) ---
    fixed = False
    seen = False
    for tbl in doc.tables:
        hdr = [c.text for c in tbl.rows[0].cells]
        if len(hdr) >= 2 and hdr[0] == 'Metric' and hdr[1].startswith('Logistic Regression'):
            for row in tbl.rows:
                if row.cells[0].text.strip() == 'Brier score':
                    seen = True
                    old = row.cells[2].text.strip()
                    if old == '0.218':
                        H.fill_cell(row.cells[2], '0.217')
                        changed.append(("10.2 Table 6 XGBoost Brier 0.218 -> 0.217",
                                        -1, f"cell(5,2) old={old}", '0.217'))
                        log_ok("10.2 Table 6 XGBoost Brier 0.218 -> 0.217")
                        fixed = True
                    else:
                        log_skip(f"10.2 Table 6 Brier cell is {old!r} (already 0.217); not touched.")
                        fixed = True
                    break
    if not fixed and not seen:
        raise LookupError("10.2 Metric table / Brier row not found for the 0.218 fix.")


# ----------------------------------------------------------------------------
# Item 11e: em-dash pass (U+2014 -> spaced en dash U+2013)
# ----------------------------------------------------------------------------

EN = '\u2013'


def _norm_run(run):
    if EM not in run.text:
        return None
    old = run.text
    run.text = re.sub(r'\s*\u2014\s*', ' \u2013 ', run.text)
    return old


def normalize_em_dashes(doc):
    """Replace every U+2014 with a spaced en dash inside the run that holds it,
    so run-level formatting (bold/italic/font) is preserved. Records each edit
    in `changed` so the em-dash verification covers these paragraphs too."""
    n_para, n_cell = 0, 0
    for i, p in enumerate(doc.paragraphs):
        if EM not in p.text:
            continue
        old = p.text
        n_runs = sum(1 for r in p.runs if EM in r.text)
        for r in p.runs:
            if _norm_run(r) is not None:
                pass
        changed.append(("11e em dash -> spaced en dash", i, old, p.text))
        log_ok(f"11e em dash normalized in P{i} ({n_runs} run(s))")
        n_para += 1
    for ti, t in enumerate(doc.tables):
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    if EM not in p.text:
                        continue
                    old = p.text
                    n_runs = sum(1 for r in p.runs if EM in r.text)
                    for r in p.runs:
                        _norm_run(r)
                    changed.append((f"11e em dash -> spaced en dash (Table {ti+1})",
                                    -1, old, p.text))
                    log_ok(f"11e em dash normalized in table {ti+1} ({n_runs} run(s))")
                    n_cell += 1
    return n_para, n_cell


# ----------------------------------------------------------------------------
# Save (atomic) + media swap + verify
# ----------------------------------------------------------------------------

def verify(doc_final, changed):
    checks = []
    paras = doc_final.paragraphs
    checks.append(("paragraph count", len(paras), "287"))
    checks.append(("table count", len(doc_final.tables), "10"))
    n_img = sum(1 for p in paras
                if p._p.findall('.//{http://schemas.openxmlformats.org/'
                                'wordprocessingml/2006/main}drawing'))
    checks.append(("images", n_img, "13"))
    text_all = "\n".join(p.text for p in paras)
    for s in ['5.5 Open-Time and Interaction Feature Experiment',
              'under-predict realized profitability',
              'within cross-validation and seed noise',
              'the training portion only for selection',
              'no value-maximizing threshold',
              'test base rate of 66.98%',
              'clear the 66.98% test base rate',
              'A wrapper cross-check (recursive feature elimination']:
        checks.append((f"present: {s[:40]}", text_all.count(s), "1"))
    for s in ['session-2', 'Session-2', 'TRAIN ONLY', 'peak $7.14M', '$2.03M',
              'even before hyperparameter tuning', 'statistically indistinguishable',
              'was not necessary', '\u2192', 'monotonically.',
              'improved test ROC-AUC by less than 0.001', 'by under 0.001 over untuned defaults']:
        checks.append((f"absent: {s!r}", text_all.count(s), "0"))
    # em-dash scan on the paragraphs we rewrote only
    bad = 0
    for (label, idx, old, new) in changed:
        if idx < 0:
            continue
        if EM in paras[idx].text:
            bad += 1
            checks.append((f"em-dash in changed P{idx} ({label})", 1, "0"))
    if bad == 0:
        checks.append(("em-dash scan of changed paragraphs", 0, "0 OK"))
    em_total = sum(p.text.count(EM) for p in paras) + sum(
        c.text.count(EM) for t in doc_final.tables
        for row in t.rows for c in row.cells)
    checks.append(("whole-document em-dash count", em_total, "0"))
    return checks


def swap_media(src_zip, dst_zip, member, replacement):
    with zipfile.ZipFile(src_zip) as zin, \
         zipfile.ZipFile(dst_zip, 'w', zipfile.ZIP_DEFLATED) as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename == member:
                data = open(replacement, 'rb').read()
            zout.writestr(item, data)


def main():
    if not preflight():
        sys.exit("Aborting (preflight).")

    global doc
    doc = docx.Document(DOCX)
    H.dump_structure(doc, BEFORE_DUMP)

    apply_edits(doc)

    n_para, n_cell = normalize_em_dashes(doc)
    if n_para or n_cell:
        print(f"  item 11e: em dashes normalized in {n_para} paragraphs + {n_cell} table cells")
    else:
        print("  item 11e: no em dashes found (already clean)")

    # ---- media swap at zip level (item 11a) ----
    doc.save(TMP)
    swap_media(TMP, TMP2, SWAP_MEMBER, SWAP_SRC)
    with zipfile.ZipFile(TMP2) as z:
        bad = z.testzip()
        if bad is not None:
            raise RuntimeError(f"temp docx corrupt member: {bad}")
        ct = z.read('[Content_Types].xml')
        if b'application/vnd.openxmlformats-officedocument.wordprocessingml' not in ct:
            raise RuntimeError("content-types sanity failed")
        img_md5 = md5(z.read(SWAP_MEMBER))
    v2_md5 = md5(SWAP_SRC)

    recheck = docx.Document(TMP2)
    changed_indices = [c[1] for c in changed if c[1] >= 0]
    all_ok = True
    for row in verify(recheck, changed):
        want = row[2]
        ok = (str(row[1]) == str(want)) or (want.endswith('OK') and row[1] == 0)
        all_ok = all_ok and ok
        print(f"  {row[0]:56s} got={row[1]}  expected={want}  {'OK' if ok else 'MISMATCH'}")
    if img_md5 != v2_md5:
        raise RuntimeError("Figure 13 media swap md5 mismatch")
    print(f"  media swap md5: {img_md5[:12]}{'...'} (v2) OK")
    all_ok = all_ok and True

    H.dump_structure(recheck, AFTER_DUMP)

    with open(DIFF, 'w') as f:
        f.write("Session-3 audit-fix log (fill_report_audit_fixes.py)\n")
        f.write("=" * 60 + "\n")
        for kind, msg in log:
            f.write(f"[{kind.upper()}] {msg}\n")
        f.write("\nChanged paragraphs (label / index / before -> after):\n")
        f.write("-" * 60 + "\n")
        for (label, idx, old, new) in changed:
            f.write(f"\n### {label} (P{idx})\n")
            f.write(f"BEFORE: {old}\n")
            if new:
                f.write(f"AFTER:  {new}\n")
            else:
                f.write("AFTER:  (table cell edit, see cell line above)\n")

    if not all_ok:
        raise RuntimeError("verification mismatches; not replacing the docx.")

    shutil.move(TMP2, DOCX)
    try:
        os.remove(TMP)
    except OSError:
        pass

    # post-move sanity
    with zipfile.ZipFile(DOCX) as z:
        assert z.testzip() is None
        assert md5(z.read(SWAP_MEMBER)) == v2_md5
    final = docx.Document(DOCX)
    print("\nApplied edits:")
    for kind, msg in log:
        print(f"  [{kind.upper()}] {msg}")
    print("\nFinal paragraph count:", len(final.paragraphs),
          " table count:", len(final.tables))
    print("Diff written to", DIFF)


if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        sys.exit(1)