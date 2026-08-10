"""
Statistical Analysis for Suspense-Tension Alignment Listening Study
===================================================================

This script implements a rigorous statistical analysis plan for a within-subjects
listening study comparing a full soundtrack generation system (with tension alignment)
against an ablation (without tension alignment).

Study design:
- Within-subjects: each participant rates both conditions for each of 3 films
- 4 Likert-scale questions (1-5) per condition per film
- Two counterbalanced forms (A and B) controlling presentation order
- Q1-Q3: music-narrative alignment; Q4: musical pleasantness

Analysis steps:
  Step 0: Data reshaping (wide → long format)
  Step 1: Sample description
  Step 2: Descriptive statistics (medians, IQRs, stacked bar charts)
  Step 3: Primary inferential analysis (Cumulative Link Mixed Models via R,
          with Wilcoxon signed-rank as a complementary nonparametric test)
  Step 4: Effect sizes (Cliff's delta with bootstrap CIs)
  Step 5: Multiple comparison correction (Holm-Bonferroni)
  Step 6: Order effects check
  Step 7: Film-by-condition interaction
  Step 8: Inter-item consistency for Q1-Q3
  Step 9: Musical background exploratory analysis

Requirements:
  pip install pandas numpy scipy matplotlib seaborn statsmodels rpy2 krippendorff

Note on CLMM: The gold-standard Cumulative Link Mixed Model requires R's `ordinal`
package. This script uses `rpy2` to call R from Python. If R is not available,
the script falls back to Wilcoxon signed-rank tests (valid but less powerful).
To install the R package: install.packages("ordinal") in R.
"""

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import wilcoxon, mannwhitneyu
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 11
import seaborn as sns
import os
from itertools import combinations

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = "analysis_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

QUESTIONS = ["q1", "q2", "q3", "q4"]
QUESTION_LABELS = {
    "q1": "Q1: Music reflects\nemotional intensity",
    "q2": "Q2: Music changes with\nemotional changes",
    "q3": "Q3: Music enhances\nemotional impact",
    "q4": "Q4: Music is pleasant\nto listen to",
}
FILMS = ["spring", "easy", "met"]
FILM_LABELS = {"spring": "Spring", "easy": "Easy Street", "met": "Metropolis"}
CONDITIONS = {"1": "Full System", "2": "Ablation"}

N_BOOTSTRAP = 10000
RANDOM_SEED = 42
ALPHA = 0.05


# ============================================================================
# STEP 0: DATA RESHAPING
# ============================================================================
def reshape_to_long(eval_1, eval_2):
    """
    Reshape from wide format (one row per participant) to long format
    (one row per participant × film × condition × question).

    The column naming convention:
      - '_1' suffix = full system (condition 1)
      - '_2' suffix = ablation (condition 2)
      - Column ORDER in each form reflects presentation order

    Both forms use the same condition coding, so we can unify them.
    We add a 'form' variable to track which counterbalancing order was used.
    """
    print("=" * 70)
    print("STEP 0: DATA RESHAPING")
    print("=" * 70)

    eval_1 = eval_1.copy()
    eval_2 = eval_2.copy()
    eval_1["form"] = "A"
    eval_2["form"] = "B"

    # Standardize column ordering: we only care about condition (_1 vs _2),
    # not presentation order. Both forms share the same column names.
    # We select only the columns we need, in a consistent order.
    demo_cols = ["sub_id", "resp_id", "date", "age", "music_background",
                 "films", "form", "comment"]

    rating_cols = []
    for film in FILMS:
        for cond in ["1", "2"]:
            for q in QUESTIONS:
                rating_cols.append(f"{q}_{film}_{cond}")

    all_cols = demo_cols + rating_cols
    eval_1 = eval_1[[c for c in all_cols if c in eval_1.columns]]
    eval_2 = eval_2[[c for c in all_cols if c in eval_2.columns]]

    combined = pd.concat([eval_1, eval_2], ignore_index=True)

    # Melt to long format
    rows = []
    for _, participant in combined.iterrows():
        for film in FILMS:
            for cond in ["1", "2"]:
                for q in QUESTIONS:
                    col = f"{q}_{film}_{cond}"
                    if col in participant.index and pd.notna(participant[col]):
                        rows.append({
                            "participant": participant["sub_id"],
                            "form": participant["form"],
                            "age": participant["age"],
                            "music_background": participant["music_background"],
                            "film": film,
                            "condition": CONDITIONS[cond],
                            "question": q,
                            "rating": int(participant[col]),
                        })

    df_long = pd.DataFrame(rows)

    print(f"  Combined participants: {df_long['participant'].nunique()}")
    print(f"    Form A: {combined[combined['form']=='A'].shape[0]}")
    print(f"    Form B: {combined[combined['form']=='B'].shape[0]}")
    print(f"  Total observations: {len(df_long)}")
    print(f"  Expected (N × 3 films × 2 conditions × 4 questions): "
          f"{df_long['participant'].nunique()} × 24 = "
          f"{df_long['participant'].nunique() * 24}")

    missing = df_long['participant'].nunique() * 24 - len(df_long)
    if missing > 0:
        print(f"  ⚠ {missing} missing observations detected")
    else:
        print(f"  ✓ No missing observations")

    print()
    return df_long


# ============================================================================
# STEP 1: SAMPLE DESCRIPTION
# ============================================================================
def sample_description(df_long):
    """
    Describe the sample: N, age distribution, musical background, form balance.

    This goes at the start of your results section so readers know who
    your participants are before seeing any statistical results.
    """
    print("=" * 70)
    print("STEP 1: SAMPLE DESCRIPTION")
    print("=" * 70)

    participants = df_long.groupby("participant").first().reset_index()
    N = len(participants)
    print(f"  Total participants: {N}")

    print(f"\n  Age distribution:")
    age_counts = participants["age"].value_counts().sort_index()
    for age, count in age_counts.items():
        print(f"    {age}: {count} ({100*count/N:.1f}%)")

    print(f"\n  Musical background:")
    music_counts = participants["music_background"].value_counts()
    for bg, count in music_counts.items():
        print(f"    {bg}: {count} ({100*count/N:.1f}%)")

    print(f"\n  Form balance:")
    form_counts = participants["form"].value_counts()
    for form, count in form_counts.items():
        print(f"    Form {form}: {count} ({100*count/N:.1f}%)")

    print()
    return participants


# ============================================================================
# STEP 2: DESCRIPTIVE STATISTICS
# ============================================================================
def descriptive_statistics(df_long):
    """
    Compute medians and IQRs per condition per question, and generate
    stacked bar charts showing the full response distribution.

    WHY MEDIANS: The median of ordinal categories is meaningful (the typical
    response category). Means assume equal spacing between categories, which
    is not guaranteed for Likert scales.

    WHY STACKED BARS: They show the full shape of the response distribution,
    revealing patterns (bimodality, floor/ceiling effects) that summaries hide.
    """
    print("=" * 70)
    print("STEP 2: DESCRIPTIVE STATISTICS")
    print("=" * 70)

    print("\n  Medians and IQRs per question per condition:")
    print(f"  {'Question':<8} {'Condition':<15} {'Median':>6} {'IQR':>10} "
          f"{'25th':>5} {'75th':>5}")
    print("  " + "-" * 55)

    for film in FILMS:
        print(FILM_LABELS[film])
        f_data = df_long[df_long["film"] == film]
        for q in QUESTIONS:
            q_data = f_data[f_data["question"] == q]
            for cond in ["Full System", "Ablation"]:
                vals = q_data[q_data["condition"] == cond]["rating"]
                med = vals.median()
                q25, q75 = vals.quantile(0.25), vals.quantile(0.75)
                print(f"  {q:<8} {cond:<15} {med:>6.1f} {q75-q25:>10.1f} "
                    f"{q25:>5.1f} {q75:>5.1f}")
            print()

    # --- Stacked bar charts ---
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), sharey=True)
    colors = ["#d73027", "#fc8d59", "#fee08b", "#91cf60", "#1a9850"]
    rating_labels = ["1 - Strongly\nDisagree", "2 - Disagree",
                     "3 - Neutral", "4 - Agree", "5 - Strongly\nAgree"]

    for idx, q in enumerate(QUESTIONS):
        ax = axes[idx]
        q_data = df_long[df_long["question"] == q]

        for j, cond in enumerate(["Full System", "Ablation"]):
            cond_data = q_data[q_data["condition"] == cond]["rating"]
            total = len(cond_data)
            bottom = 0
            for rating_val in range(1, 6):
                count = (cond_data == rating_val).sum()
                prop = count / total if total > 0 else 0
                bar = ax.bar(j, prop, bottom=bottom, color=colors[rating_val - 1],
                             edgecolor="white", linewidth=0.5, width=0.6)
                if prop > 0.08:
                    ax.text(j, bottom + prop / 2, f"{prop:.0%}",
                            ha="center", va="center", fontsize=8,
                            fontweight="bold", color="black" if rating_val == 3
                            else "white")
                bottom += prop

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Full\nSystem", "Ablation"], fontsize=9)
        ax.set_title(QUESTION_LABELS[q], fontsize=9, pad=8)
        ax.set_ylim(0, 1)
        if idx == 0:
            ax.set_ylabel("Proportion of responses", fontsize=10)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], label=rating_labels[i])
                       for i in range(5)]
    fig.legend(handles=legend_elements, loc="lower center", ncol=5,
               fontsize=8, bbox_to_anchor=(0.5, -0.08))

    plt.suptitle("Response Distributions by Condition and Question",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "stacked_bars.pdf"),
                bbox_inches="tight", dpi=300)
    plt.savefig(os.path.join(OUTPUT_DIR, "stacked_bars.png"),
                bbox_inches="tight", dpi=300)
    plt.close()
    print("  → Saved stacked bar chart to analysis_output/stacked_bars.pdf")

    # --- Per-film descriptives ---
    print("\n  Medians per question × condition × film:")
    print(f"  {'Question':<6} {'Film':<12} {'Full':>6} {'Ablation':>9} {'Diff':>6}")
    print("  " + "-" * 45)
    for q in QUESTIONS:
        for film in FILMS:
            mask = (df_long["question"] == q) & (df_long["film"] == film)
            full_med = df_long[mask & (df_long["condition"] == "Full System")][
                "rating"].median()
            abl_med = df_long[mask & (df_long["condition"] == "Ablation")][
                "rating"].median()
            print(f"  {q:<6} {FILM_LABELS[film]:<12} {full_med:>6.1f} "
                  f"{abl_med:>9.1f} {full_med - abl_med:>+6.1f}")
        print()

    print()


# ============================================================================
# STEP 3: PRIMARY INFERENTIAL ANALYSIS
# ============================================================================

# --- 3a: Cumulative Link Mixed Model via R ---
def run_clmm(df_long):
    """
    Fit a Cumulative Link Mixed Model (CLMM) for each question using R's
    ordinal::clmm via rpy2.

    The CLMM is the gold-standard model for ordinal responses with repeated
    measures. It:
      - Respects the ordinal nature of Likert data (no equal-spacing assumption)
      - Accounts for repeated measures via random intercepts per participant
      - Simultaneously controls for film, form, and their interactions

    The key output is the ODDS RATIO for condition: how many times more likely
    a participant is to give a higher rating under the full system.

    Returns a dict of results per question, or None if R is unavailable.
    """
    print("-" * 70)
    print("  STEP 3a: Cumulative Link Mixed Model (CLMM)")
    print("-" * 70)

    try:
        import rpy2.robjects as ro
        from rpy2.robjects import pandas2ri
        from rpy2.robjects.packages import importr

        # Note: pandas2ri.activate() is deprecated in rpy2 >= 3.5.
        # We use the context manager (ro.default_converter + pandas2ri.converter)
        # for each conversion instead.
        converter = ro.default_converter + pandas2ri.converter

        ordinal = importr("ordinal")
        base = importr("base")
        stats_r = importr("stats")

        print("  ✓ R and ordinal package available")
    except Exception as e:
        print(f"  ✗ R/rpy2 not available ({e})")
        print("    Skipping CLMM. Wilcoxon tests (Step 3b) will serve as")
        print("    the primary analysis instead.")
        print("    To enable CLMM: pip install rpy2, then in R:")
        print("    install.packages('ordinal')")
        return None

    results = {}

    for q in QUESTIONS:
        q_data = df_long[df_long["question"] == q].copy()
        q_data["rating_ordered"] = q_data["rating"].astype(str)
        q_data["condition"] = pd.Categorical(
            q_data["condition"], categories=["Ablation", "Full System"]
        )
        q_data["film"] = pd.Categorical(q_data["film"])
        q_data["form"] = pd.Categorical(q_data["form"])
        q_data["participant"] = q_data["participant"].astype(str)

        # Transfer data to R using the context manager
        with converter.context():
            r_df = ro.conversion.get_conversion().py2rpy(q_data)

        ro.globalenv["df"] = r_df

        # Prepare factors in R
        ro.r('''
            df$rating_ordered <- factor(df$rating_ordered, levels=c("1","2","3","4","5"), ordered=TRUE)
            df$condition <- factor(df$condition, levels=c("Ablation", "Full System"))
            df$film <- factor(df$film)
            df$form <- factor(df$form)
            df$participant <- factor(df$participant)
        ''')

        # Fit full model and null model (without condition)
        try:
            ro.r('''
                model_full <- clmm(rating_ordered ~ condition + film + condition:film + form
                                   + (1|participant), data=df, Hess=TRUE)

                # Null model for testing the OVERALL condition effect:
                # Must remove BOTH condition AND condition:film, because
                # the interaction term encodes condition differences implicitly.
                # If we only remove the main effect, R re-absorbs it into the
                # interaction, and the LR test gives ~0.
                model_null <- clmm(rating_ordered ~ film + form
                                   + (1|participant), data=df, Hess=TRUE)
            ''')

            # Likelihood ratio test
            lr_test = ro.r("anova(model_null, model_full)")

            # Extract results
            summary_full = ro.r("summary(model_full)")
            coefs = ro.r("coef(summary(model_full))")

            # Get the condition coefficient (log-odds)
            # Use ^...$ anchors so grep matches only the main effect row,
            # not the interaction rows (conditionFull System:filmeasy, etc.)
            condition_coef_r = ro.r('''
                cf <- coef(summary(model_full))
                idx <- grep("^conditionFull System$", rownames(cf))
                if (length(idx) > 0) cf[idx, ] else c(NA, NA, NA, NA)
            ''')
            condition_coef = np.array(condition_coef_r)

            # Get profile confidence interval.
            # If the Hessian is not positive definite, confint will fail.
            # In that case we report the CI as not estimable (NA) rather than
            # falling back to Wald CIs, which rely on the same broken Hessian.
            # The LR test and odds ratio point estimate remain valid (they are
            # based on log-likelihoods, not the Hessian). Cliff's delta with
            # bootstrap CI provides an independent effect size bound.
            ci_r = ro.r('''
                tryCatch({
                    ci <- confint(model_full, parm="conditionFull System")
                    vals <- as.numeric(ci)
                    if (any(is.na(vals)) || any(is.infinite(vals)) ||
                        (all(abs(vals) < 1e-6))) {
                        c(NA, NA)
                    } else {
                        vals
                    }
                }, error=function(e) c(NA, NA))
            ''')
            ci = np.array(ci_r)
            ci_valid = not np.any(np.isnan(ci))

            # Extract LR test statistics in one call
            lr_stats_r = ro.r('''
                a <- anova(model_null, model_full)
                c(a[["LR.stat"]][2], a[["df"]][2], a[["Pr(>Chisq)"]][2])
            ''')
            lr_stats = np.array(lr_stats_r)
            lr_chi = float(lr_stats[0])
            lr_df = int(lr_stats[1])
            lr_pval = float(lr_stats[2])

            log_odds = condition_coef[0] if len(condition_coef) >= 1 else np.nan
            odds_ratio = np.exp(log_odds)
            ci_or = np.exp(ci) if ci_valid else [np.nan, np.nan]

            results[q] = {
                "log_odds": log_odds,
                "odds_ratio": odds_ratio,
                "or_ci_lower": ci_or[0],
                "or_ci_upper": ci_or[1],
                "ci_valid": ci_valid,
                "lr_chi2": lr_chi,
                "lr_df": lr_df,
                "lr_pvalue": lr_pval,
            }

            print(f"\n  {q.upper()} ({QUESTION_LABELS[q].replace(chr(10), ' ')}):")
            print(f"    Log-odds (Full vs Ablation): {log_odds:.3f}")
            if ci_valid:
                print(f"    Odds ratio: {odds_ratio:.3f} "
                      f"[95% CI: {ci_or[0]:.3f}, {ci_or[1]:.3f}]")
            else:
                print(f"    Odds ratio: {odds_ratio:.3f} "
                      f"[95% CI: not estimable — Hessian not positive definite]")
                print(f"    ⚠ Profile CI unavailable. OR point estimate and LR test")
                print(f"      remain valid. Use Cliff's δ CI as effect size bound.")
            print(f"    Likelihood ratio test: χ²({lr_df}) = {lr_chi:.3f}, "
                  f"p = {lr_pval:.4f}")

            interpretation = (
                "Full System rated significantly higher"
                if lr_pval < ALPHA
                else "No significant difference"
            )
            print(f"    → {interpretation} (before multiple comparison correction)")

            # Also extract film interaction
            interaction_r = ro.r('''
                model_no_inter <- clmm(rating_ordered ~ condition + film + form
                                       + (1|participant), data=df, Hess=TRUE)
                a <- anova(model_no_inter, model_full)
                c(a[["LR.stat"]][2], a[["df"]][2],
                  a[["Pr(>Chisq)"]][2])
            ''')
            interaction_vals = np.array(interaction_r)
            results[q]["interaction_chi2"] = interaction_vals[0]
            results[q]["interaction_df"] = int(interaction_vals[1])
            results[q]["interaction_pvalue"] = interaction_vals[2]

            print(f"    Film × Condition interaction: χ²({int(interaction_vals[1])}) "
                  f"= {interaction_vals[0]:.3f}, p = {interaction_vals[2]:.4f}")

            # Extract form effect
            form_r = ro.r('''
                cf <- coef(summary(model_full))
                idx <- grep("^form", rownames(cf))
                if (length(idx) > 0) {
                    c(cf[idx[1], 1], cf[idx[1], 4])
                } else {
                    c(NA, NA)
                }
            ''')
            form_vals = np.array(form_r)
            results[q]["form_coef"] = form_vals[0]
            results[q]["form_pvalue"] = form_vals[1]

        except Exception as e:
            print(f"\n  ✗ CLMM failed for {q}: {e}")
            results[q] = None

    print()
    return results


# --- 3b: Wilcoxon signed-rank test (complementary / fallback) ---
def run_wilcoxon(df_long):
    """
    Wilcoxon signed-rank test: a nonparametric test for paired ordinal data.

    For each question, we compute the per-participant DIFFERENCE in ratings
    (Full System - Ablation), aggregated across films, and test whether this
    difference distribution is centered on zero.

    This test:
      - Makes no distributional assumptions (appropriate for ordinal data)
      - Respects the pairing (same participant, both conditions)
      - Is simpler than CLMM but cannot control for film/form simultaneously

    It serves as a complementary analysis if CLMM is available, or as the
    primary analysis if R is not installed.
    """
    print("-" * 70)
    print("  STEP 3b: Wilcoxon Signed-Rank Tests (paired, per question)")
    print("-" * 70)
    print("  For each participant, we average their ratings across films")
    print("  for each condition, then test the paired difference.\n")

    results = {}

    for q in QUESTIONS:
        q_data = df_long[df_long["question"] == q]

        # Compute per-participant mean rating per condition (across films)
        pivot = q_data.pivot_table(
            index="participant", columns="condition",
            values="rating", aggfunc="median"
        )

        full = pivot["Full System"].values
        ablation = pivot["Ablation"].values

        # Remove participants with no difference (Wilcoxon requires this info)
        diff = full - ablation
        n_tied = np.sum(diff == 0)
        n_positive = np.sum(diff > 0)
        n_negative = np.sum(diff < 0)

        # Two-sided Wilcoxon signed-rank test
        if np.all(diff == 0):
            stat, pval = np.nan, 1.0
        else:
            stat, pval = wilcoxon(full, ablation, alternative="two-sided",
                                  zero_method="wilcox")

        results[q] = {
            "statistic": stat,
            "p_value": pval,
            "n_pairs": len(diff),
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_tied": n_tied,
            "median_diff": np.median(diff),
        }

        print(f"  {q.upper()} ({QUESTION_LABELS[q].replace(chr(10), ' ')}):")
        print(f"    N pairs: {len(diff)} | "
              f"Full > Ablation: {n_positive} | "
              f"Ablation > Full: {n_negative} | "
              f"Tied: {n_tied}")
        print(f"    Median difference (Full - Ablation): {np.median(diff):+.2f}")
        print(f"    Wilcoxon W = {stat:.1f}, p = {pval:.4f}")
        print()

    return results


# ============================================================================
# STEP 4: EFFECT SIZES — CLIFF'S DELTA
# ============================================================================
def cliffs_delta_value(x, y):
    """
    Compute Cliff's delta between two samples.

    delta = (# times x > y  -  # times y > x) / (n_x * n_y)

    Ranges from -1 to +1:
      +1 = every value in x exceeds every value in y
       0 = no systematic difference
      -1 = every value in y exceeds every value in x
    """
    n_x, n_y = len(x), len(y)
    more = sum(1 for xi in x for yi in y if xi > yi)
    less = sum(1 for xi in x for yi in y if xi < yi)
    return (more - less) / (n_x * n_y)


def compute_cliffs_delta(df_long):
    """
    Compute Cliff's delta with BCa bootstrap confidence intervals.

    Cliff's delta is a nonparametric, assumption-free effect size for ordinal
    data. It answers: "If I randomly pick one rating from the Full System and
    one from the Ablation, how much more likely is the Full System rating to
    be higher?"

    IMPORTANT: We first aggregate each participant's ratings across films
    (using median), so each participant contributes one value per condition.
    This respects the within-subjects design — without aggregation, a
    participant's high rating for Metropolis-Full would be compared against
    another participant's low rating for Spring-Ablation, mixing between-film
    variance into the effect size.

    The relationship to probability:
      P(Full > Ablation) = (1 + delta) / 2

    Conventional thresholds (Romano et al., 2006):
      |delta| < 0.147  → negligible
      0.147 - 0.33     → small
      0.33  - 0.474    → medium
      > 0.474           → large

    We use BCa bootstrap for confidence intervals because Cliff's delta
    can have a skewed sampling distribution with small samples.
    """
    print("=" * 70)
    print("STEP 4: EFFECT SIZES — CLIFF'S DELTA")
    print("=" * 70)
    print("  (Computed on per-participant medians across films)\n")

    rng = np.random.RandomState(RANDOM_SEED)
    results = {}

    for q in QUESTIONS:
        q_data = df_long[df_long["question"] == q]

        # Aggregate: per-participant median across films, one value per condition
        participant_scores = q_data.pivot_table(
            index="participant", columns="condition",
            values="rating", aggfunc="median"
        )
        full = participant_scores["Full System"].values
        ablation = participant_scores["Ablation"].values

        # Point estimate
        delta = cliffs_delta_value(full, ablation)

        # Bootstrap CI: resample participants (preserving pairing)
        boot_deltas = []
        n = len(full)
        for _ in range(N_BOOTSTRAP):
            idx = rng.choice(n, size=n, replace=True)
            boot_deltas.append(cliffs_delta_value(full[idx], ablation[idx]))

        boot_deltas = np.array(boot_deltas)
        ci_lower = np.percentile(boot_deltas, 2.5)
        ci_upper = np.percentile(boot_deltas, 97.5)

        # Interpret magnitude
        abs_d = abs(delta)
        if abs_d < 0.147:
            magnitude = "negligible"
        elif abs_d < 0.33:
            magnitude = "small"
        elif abs_d < 0.474:
            magnitude = "medium"
        else:
            magnitude = "large"

        # Probability interpretation
        p_full_wins = (1 + delta) / 2

        results[q] = {
            "delta": delta,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "magnitude": magnitude,
            "p_full_wins": p_full_wins,
        }

        print(f"\n  {q.upper()} ({QUESTION_LABELS[q].replace(chr(10), ' ')}):")
        print(f"    Cliff's δ = {delta:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
        print(f"    Magnitude: {magnitude}")
        print(f"    P(Full > Ablation) = {p_full_wins:.1%}")

        sig = "yes" if (ci_lower > 0 or ci_upper < 0) else "no"
        print(f"    CI excludes zero: {sig}")

    print()
    return results


# ============================================================================
# STEP 5: MULTIPLE COMPARISON CORRECTION
# ============================================================================
def holm_bonferroni_correction(p_values_dict, label=""):
    """
    Apply Holm-Bonferroni correction to a set of p-values.

    WHY THIS IS NEEDED: With 4 tests at α=0.05 each, the probability of at
    least one false positive under the null is 1 - 0.95^4 ≈ 18.5%. Holm-
    Bonferroni controls the family-wise error rate at 5% across ALL tests.

    HOW IT WORKS: Sort p-values ascending. Multiply the smallest by k (number
    of tests), the second by k-1, etc. Compare each to α. It is strictly more
    powerful than Bonferroni (which multiplies all by k).

    Returns a dict mapping question → (original_p, adjusted_p, significant).
    """
    print("=" * 70)
    print(f"STEP 5: HOLM-BONFERRONI CORRECTION {label}")
    print("=" * 70)

    questions = list(p_values_dict.keys())
    p_vals = [p_values_dict[q] for q in questions]

    reject, p_adjusted, _, _ = multipletests(p_vals, alpha=ALPHA, method="holm")

    results = {}
    print(f"\n  {'Question':<8} {'p (raw)':<12} {'p (adjusted)':<14} {'Significant'}")
    print("  " + "-" * 50)

    for q, p_raw, p_adj, sig in zip(questions, p_vals, p_adjusted, reject):
        results[q] = {
            "p_raw": p_raw,
            "p_adjusted": p_adj,
            "significant": bool(sig),
        }
        sig_str = "✓ Yes" if sig else "✗ No"
        print(f"  {q:<8} {p_raw:<12.4f} {p_adj:<14.4f} {sig_str}")

    print()
    return results


# ============================================================================
# STEP 6: ORDER EFFECTS
# ============================================================================
def check_order_effects(df_long):
    """
    Test whether presentation order (Form A vs B) affected ratings.

    If the CLMM is available, the form coefficient already captures this.
    Here we provide a standalone check using Mann-Whitney U: for each question,
    we compare ratings from Form A participants vs Form B participants.

    A non-significant result validates the counterbalanced design. A significant
    result suggests carry-over or contrast effects that must be discussed.
    """
    print("=" * 70)
    print("STEP 6: ORDER EFFECTS CHECK")
    print("=" * 70)

    for q in QUESTIONS:
        q_data = df_long[df_long["question"] == q]

        form_a = q_data[q_data["form"] == "A"]["rating"].values
        form_b = q_data[q_data["form"] == "B"]["rating"].values

        stat, pval = mannwhitneyu(form_a, form_b, alternative="two-sided")

        sig = "⚠ SIGNIFICANT" if pval < ALPHA else "✓ not significant"
        print(f"  {q.upper()}: Mann-Whitney U = {stat:.1f}, p = {pval:.4f} → {sig}")
        print(f"    Form A median: {np.median(form_a):.1f}, "
              f"Form B median: {np.median(form_b):.1f}")

    print()


# ============================================================================
# STEP 7: FILM-BY-CONDITION INTERACTION
# ============================================================================
def film_interaction(df_long):
    """
    Test whether the condition effect varies across films.

    For each question, we run Wilcoxon signed-rank tests separately per film,
    providing per-film evidence of the condition effect.

    If the CLMM interaction term was significant (Step 3a), these per-film
    tests show WHERE the difference lies. If not, they provide confirmatory
    detail showing the effect is consistent across films.
    """
    print("=" * 70)
    print("STEP 7: PER-FILM CONDITION EFFECTS")
    print("=" * 70)

    for q in QUESTIONS:
        print(f"\n  {q.upper()}:")
        for film in FILMS:
            mask = (df_long["question"] == q) & (df_long["film"] == film)
            film_data = df_long[mask]

            pivot = film_data.pivot_table(
                index="participant", columns="condition",
                values="rating", aggfunc="first"
            )
            if "Full System" not in pivot.columns or "Ablation" not in pivot.columns:
                continue

            full = pivot["Full System"].dropna().values
            abl = pivot["Ablation"].dropna().values
            n = min(len(full), len(abl))
            full, abl = full[:n], abl[:n]

            diff = full - abl
            if np.all(diff == 0):
                print(f"    {FILM_LABELS[film]:<12}: all tied (no difference)")
                continue

            stat, pval = wilcoxon(full, abl, alternative="two-sided",
                                  zero_method="wilcox")
            delta = cliffs_delta_value(full, abl)

            # Direction based on non-tied pairs (more informative than median
            # when many participants give the same rating to both conditions)
            n_pos = np.sum(diff > 0)
            n_neg = np.sum(diff < 0)
            n_tied = np.sum(diff == 0)
            if n_pos > n_neg:
                direction = f"Full > Abl ({n_pos} vs {n_neg}, {n_tied} tied)"
            elif n_neg > n_pos:
                direction = f"Abl > Full ({n_neg} vs {n_pos}, {n_tied} tied)"
            else:
                direction = f"Symmetric ({n_pos} vs {n_neg}, {n_tied} tied)"
            print(f"    {FILM_LABELS[film]:<12}: Wilcoxon p = {pval:.4f}, "
                  f"Cliff's δ = {delta:+.3f}, {direction}")

    print()


# ============================================================================
# STEP 8: INTER-ITEM CONSISTENCY FOR Q1-Q3
# ============================================================================
def inter_item_consistency(df_long):
    """
    Assess whether Q1, Q2, and Q3 measure the same underlying construct
    (music-narrative alignment) using Krippendorff's alpha for ordinal data.

    If alpha > 0.7, the items are sufficiently consistent to justify a
    COMPOSITE SCORE (the per-participant median across Q1-Q3), which
    reduces measurement noise and gives a more powerful test.

    If alpha < 0.6, the questions measure different things and should
    be analyzed separately (which we already do above).
    """
    print("=" * 70)
    print("STEP 8: INTER-ITEM CONSISTENCY (Q1-Q3)")
    print("=" * 70)

    try:
        import krippendorff

        # Build a reliability matrix: each row is a "rater" (question),
        # each column is a "unit" (participant × film × condition)
        alignment_qs = ["q1", "q2", "q3"]
        units = []
        for q in alignment_qs:
            q_data = df_long[df_long["question"] == q].sort_values(
                ["participant", "film", "condition"]
            )
            units.append(q_data["rating"].values)

        # Ensure equal lengths
        min_len = min(len(u) for u in units)
        reliability_data = [u[:min_len] for u in units]

        alpha = krippendorff.alpha(
            reliability_data=reliability_data,
            level_of_measurement="ordinal"
        )

        if alpha > 0.7:
            interpretation = "acceptable — composite score justified"
        elif alpha > 0.6:
            interpretation = "questionable — interpret composite with caution"
        else:
            interpretation = "low — analyze questions separately"

        print(f"  Krippendorff's α (ordinal) = {alpha:.3f}")
        print(f"  Interpretation: {interpretation}")

    except ImportError:
        print("  ⚠ krippendorff package not installed. Using Cronbach's alpha")
        print("    (note: Cronbach's alpha assumes interval data, so this is")
        print("    an approximation. pip install krippendorff for ordinal alpha)")

        # Fallback: Cronbach's alpha
        alignment_qs = ["q1", "q2", "q3"]
        items = []
        for q in alignment_qs:
            q_data = (df_long[df_long["question"] == q]
                      .sort_values(["participant", "film", "condition"])
                      ["rating"].values)
            items.append(q_data)

        min_len = min(len(i) for i in items)
        item_matrix = np.column_stack([i[:min_len] for i in items])

        k = item_matrix.shape[1]
        item_vars = np.var(item_matrix, axis=0, ddof=1)
        total_var = np.var(np.sum(item_matrix, axis=1), ddof=1)
        alpha = (k / (k - 1)) * (1 - np.sum(item_vars) / total_var)

        print(f"  Cronbach's α = {alpha:.3f}")

    # Compute composite and run test
    print("\n  Composite score analysis (median of Q1-Q3 per observation):")
    composite = (
        df_long[df_long["question"].isin(["q1", "q2", "q3"])]
        .groupby(["participant", "film", "condition"])["rating"]
        .median()
        .reset_index()
        .rename(columns={"rating": "composite"})
    )

    pivot = composite.pivot_table(
        index="participant", columns="condition",
        values="composite", aggfunc="median"
    )
    full = pivot["Full System"].dropna().values
    abl = pivot["Ablation"].dropna().values
    n = min(len(full), len(abl))
    full, abl = full[:n], abl[:n]

    if not np.all(full == abl):
        stat, pval = wilcoxon(full, abl, alternative="two-sided",
                              zero_method="wilcox")
        delta = cliffs_delta_value(full, abl)
        print(f"    Composite Wilcoxon: W = {stat:.1f}, p = {pval:.4f}")
        print(f"    Composite Cliff's δ = {delta:.3f}")
        print(f"    Full median: {np.median(full):.2f}, "
              f"Ablation median: {np.median(abl):.2f}")
    else:
        print("    All composite scores tied between conditions.")

    print()


# ============================================================================
# STEP 9: MUSICAL BACKGROUND EXPLORATORY ANALYSIS
# ============================================================================
def musical_background_analysis(df_long):
    """
    Exploratory analysis: does musical training modulate the condition effect?

    For each question, we split participants into "trained" (formal training)
    vs "untrained" (no training / informal) and compare the condition effect
    magnitude between groups.

    This is EXPLORATORY (the study was not powered for this comparison),
    so results should be described as suggestive rather than confirmatory.
    """
    print("=" * 70)
    print("STEP 9: MUSICAL BACKGROUND (EXPLORATORY)")
    print("=" * 70)

    # Explicit mapping based on actual survey categories.
    # "Trained" = participants with formal structured training (2+ years).
    # "Untrained" = no training or only informal/basic experience.
    TRAINED_CATEGORIES = {
        "Intermediate (2-5 years of formal training)",
        "Advanced (more than 5 years of formal training)",
        "Professional (music degree or professional experience)",
    }

    bg_values = df_long["music_background"].unique()
    print(f"  Musical background categories in data: {list(bg_values)}")

    unmapped = set(bg_values) - TRAINED_CATEGORIES - {
        "No musical training",
        "Basic (some informal experience or less than 2 years of training)",
    }
    if unmapped:
        print(f"  ⚠ Unmapped categories (defaulting to Untrained): {unmapped}")
        print(f"    → If these should be Trained, add them to TRAINED_CATEGORIES")

    df_long = df_long.copy()
    df_long["bg_group"] = df_long["music_background"].apply(
        lambda x: "Trained" if x in TRAINED_CATEGORIES else "Untrained"
    )

    group_counts = df_long.groupby("bg_group")["participant"].nunique()
    print(f"\n  Group sizes: {dict(group_counts)}")
    print(f"    Trained = Intermediate + Advanced + Professional")
    print(f"    Untrained = No training + Basic\n")

    for q in QUESTIONS:
        print(f"  {q.upper()}:")
        for group in ["Trained", "Untrained"]:
            mask = (df_long["question"] == q) & (df_long["bg_group"] == group)
            group_data = df_long[mask]

            full = group_data[group_data["condition"] == "Full System"][
                "rating"].values
            abl = group_data[group_data["condition"] == "Ablation"][
                "rating"].values

            if len(full) == 0 or len(abl) == 0:
                print(f"    {group}: no data")
                continue

            delta = cliffs_delta_value(full, abl)
            print(f"    {group:<10}: Cliff's δ = {delta:+.3f} "
                  f"(n_full={len(full)}, n_abl={len(abl)})")

        print()


# ============================================================================
# STEP 10: SUMMARY TABLE
# ============================================================================
def print_summary(clmm_results, wilcoxon_results, cliff_results, holm_results):
    """
    Print a consolidated summary table suitable for copy-paste into the paper.
    """
    print("=" * 70)
    print("SUMMARY TABLE FOR PAPER")
    print("=" * 70)
    print()

    has_clmm = clmm_results is not None and all(
        clmm_results.get(q) is not None for q in QUESTIONS
    )

    if has_clmm:
        print(f"  {'Q':<4} {'OR':>6} {'OR 95% CI':>16} "
              f"{'χ²':>8} {'p (adj)':>10} {'δ':>7} {'δ 95% CI':>16} {'Sig':>5}")
        print("  " + "-" * 75)
        for q in QUESTIONS:
            cl = clmm_results[q]
            cf = cliff_results[q]
            hm = holm_results[q]

            if cl.get("ci_valid", True):
                ci_str = f"[{cl['or_ci_lower']:>5.2f}, {cl['or_ci_upper']:>5.2f}] "
            else:
                ci_str = "      [N/E]†"

            print(f"  {q.upper():<4} "
                  f"{cl['odds_ratio']:>6.2f} "
                  f"{ci_str}"
                  f"{cl['lr_chi2']:>8.2f} "
                  f"{hm['p_adjusted']:>10.4f} "
                  f"{cf['delta']:>+7.3f} "
                  f"[{cf['ci_lower']:>+6.3f}, {cf['ci_upper']:>+6.3f}] "
                  f"{'✓' if hm['significant'] else '✗':>5}")

        has_ne = any(not clmm_results[q].get("ci_valid", True)
                     for q in QUESTIONS if clmm_results.get(q))
        if has_ne:
            print("\n  † Profile CI not estimable (Hessian not positive definite).")
            print("    OR point estimate and LR test remain valid (likelihood-based).")
            print("    Cliff's δ with bootstrap CI serves as the effect size bound.")
    else:
        print(f"  {'Q':<4} {'W':>8} {'p (raw)':>10} {'p (adj)':>10} "
              f"{'δ':>7} {'δ 95% CI':>16} {'Sig':>5}")
        print("  " + "-" * 65)
        for q in QUESTIONS:
            wc = wilcoxon_results[q]
            cf = cliff_results[q]
            hm = holm_results[q]
            print(f"  {q.upper():<4} "
                  f"{wc['statistic']:>8.1f} "
                  f"{wc['p_value']:>10.4f} "
                  f"{hm['p_adjusted']:>10.4f} "
                  f"{cf['delta']:>+7.3f} "
                  f"[{cf['ci_lower']:>+6.3f}, {cf['ci_upper']:>+6.3f}] "
                  f"{'✓' if hm['significant'] else '✗':>5}")

    print()
    print("  OR = Odds Ratio from CLMM; χ² = Likelihood Ratio test statistic")
    print("  δ = Cliff's delta; p (adj) = Holm-Bonferroni corrected p-value")
    print("  Sig = significant at α = 0.05 after correction")
    print()


# ============================================================================
# MAIN
# ============================================================================
def main(eval_1, eval_2):
    """
    Run the complete analysis pipeline.

    Parameters
    ----------
    eval_1 : pd.DataFrame
        Responses from Form A (first counterbalancing order)
    eval_2 : pd.DataFrame
        Responses from Form B (second counterbalancing order)
    """
    # Step 0: Reshape
    df_long = reshape_to_long(eval_1, eval_2)

    # Step 1: Describe sample
    participants = sample_description(df_long)

    # Step 2: Descriptives and visualizations
    descriptive_statistics(df_long)

    # Step 3: Primary analysis
    print("=" * 70)
    print("STEP 3: PRIMARY INFERENTIAL ANALYSIS")
    print("=" * 70)

    clmm_results = run_clmm(df_long)
    wilcoxon_results = run_wilcoxon(df_long)

    # Step 4: Effect sizes
    cliff_results = compute_cliffs_delta(df_long)

    # Step 5: Multiple comparison correction
    # Use CLMM p-values if available, otherwise Wilcoxon
    if clmm_results and all(clmm_results.get(q) is not None for q in QUESTIONS):
        p_dict = {q: clmm_results[q]["lr_pvalue"] for q in QUESTIONS}
        holm_results = holm_bonferroni_correction(p_dict, "(CLMM p-values)")
    else:
        p_dict = {q: wilcoxon_results[q]["p_value"] for q in QUESTIONS}
        holm_results = holm_bonferroni_correction(p_dict, "(Wilcoxon p-values)")

    # Step 6: Order effects
    check_order_effects(df_long)

    # Step 7: Film interaction
    film_interaction(df_long)

    # Step 8: Inter-item consistency
    inter_item_consistency(df_long)

    # Step 9: Musical background
    musical_background_analysis(df_long)

    # Step 10: Summary
    print_summary(clmm_results, wilcoxon_results, cliff_results, holm_results)

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print(f"Figures saved to: {OUTPUT_DIR}/")
    print("=" * 70)

    return {
        "df_long": df_long,
        "clmm": clmm_results,
        "wilcoxon": wilcoxon_results,
        "cliff": cliff_results,
        "holm": holm_results,
    }


# ============================================================================
# ENTRY POINT — Replace with your actual data loading
# ============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("  STATISTICAL ANALYSIS: SUSPENSE-TENSION ALIGNMENT STUDY")
    print("=" * 70)
    print()
    print("  To run: load your two dataframes and call main(eval_1, eval_2)")
    print()
    print("  Example:")
    print("    import pandas as pd")
    print("    eval_1 = pd.read_csv('form_a_responses.csv')")
    print("    eval_2 = pd.read_csv('form_b_responses.csv')")
    print("    results = main(eval_1, eval_2)")
    print()