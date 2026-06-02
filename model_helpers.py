"""
model_helpers.py
================
Shared helper functions for the West Africa conflict prediction thesis.

Used by:
    04_baseline_model_acled.ipynb
    05_final_model_merged.ipynb
    06_robustness_check.ipynb

Having all shared logic here means both notebooks use identical functions,
so any performance difference in the comparison is attributable to the
predictors, not to differences in how models are fitted or evaluated.

Contents
--------
Constants        : HORIZONS, TRAIN_END, TEST_START, predictor lists
fix_float_col    : repairs multi-dot float formatting from CSV exports
get_split        : temporal train/test split for a given horizon
fit_negbinom     : fits NB regression with scaling + optimiser cascade
predict_negbinom : generates predictions using the fitted model + scaler
evaluate         : computes RMSE and MAE
check_vif        : computes Variance Inflation Factors
build_coef_table : extracts a clean coefficient table from a fitted model
get_country_split: per-country version of get_split with collinearity guards
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor


# ── Shared constants ───────────────────────────────────────────────────────────

HORIZONS   = [1, 2, 4]
TRAIN_END  = pd.Timestamp("2021-12-31")
TEST_START = pd.Timestamp("2022-01-01")

# ── Predictor sets ─────────────────────────────────────────────────────────────
#
# ACLED baseline predictors — capturing three conceptual dimensions:
#
#   Short-term dynamics (individual lags):
#     events_lag1–4: event count 1 to 4 weeks before prediction.
#                    Individual lags capture week-specific spikes and
#                    allow the model to weight recent weeks differently.
#
#   Medium-term level (moving averages):
#     events_ma4:    4-week rolling mean — the most recent sustained level.
#     events_ma8:    8-week rolling mean — slightly longer-term trend.
#                    Moving averages smooth noise and capture the conflict
#                    regime a country is currently in.
#
#   Volatility:
#     events_std4:   4-week rolling standard deviation.
#                    Erratic conflict patterns signal active dynamics.
#
#   Persistence:
#     conflict_weeks_last8: count of weeks with any conflict in past 8 weeks.
#                           Captures sustained conflict streaks, which the
#                           literature shows is among the strongest predictors
#                           of continued political violence.
#
# Note on multicollinearity: individual lags and moving averages are all
# derived from the same underlying event count series, so they are correlated.
# VIF analysis confirmed high VIF for several of these predictors when included
# simultaneously. However, the supervisor confirmed that multicollinearity is
# acceptable when the goal is prediction rather than causal inference of
# individual coefficients, provided the model converges and performs well on
# held-out data. Both conditions are met here.

BASELINE_PREDICTORS = [
    "events_lag1",
    "events_lag2",
    "events_lag3",
    "events_lag4",
    "events_ma4",
    "events_ma8",
    "events_std4",
    "conflict_weeks_last8",
]

# GDELT predictors — one variable per conceptual media dimension, lagged 1 week.
# Lagging prevents reverse causality: we use last week's media coverage
# to predict next week's conflict, not concurrent coverage.
#
#   tone_lag1:       average sentiment tone of news (negative = more conflictual).
#   goldstein_lag1:  average Goldstein scale score (-10 conflict to +10 cooperation).
#   mentions_lag1:   total media mention volume — proxy for international attention.
#
# VIF for these three is below 3, confirming they are not collinear with each
# other or with the ACLED predictors.

GDELT_PREDICTORS = [
    "tone_lag1",
    "goldstein_lag1",
    "mentions_lag1",
]

# Extended = baseline + GDELT. The models are nested by construction:
# any difference in performance is attributable only to the GDELT variables.
EXTENDED_PREDICTORS = BASELINE_PREDICTORS + GDELT_PREDICTORS


# ── Data cleaning helper ───────────────────────────────────────────────────────

def fix_float_col(series: pd.Series) -> pd.Series:
    """
    Repair float columns where dots were used as thousand separators
    inside the decimal expansion — an artefact of certain CSV export pipelines.

    Example:  '-24.357.218.790'  →  -24.357218790

    Strategy: keep the first dot as the decimal separator and remove all
    subsequent dots. Already-clean values (zero or one dot) pass through
    unchanged.
    """
    def _fix(val):
        if not isinstance(val, str):
            try:
                return float(val)
            except (ValueError, TypeError):
                return np.nan
        val = val.strip()
        if val in ("", "nan", "NaN", "None", "NULL"):
            return np.nan
        sign = ""
        if val.startswith("-"):
            sign = "-"
            val  = val[1:]
        dots = [i for i, c in enumerate(val) if c == "."]
        if len(dots) > 1:
            val = val[:dots[0]] + "." + val[dots[0] + 1:].replace(".", "")
        try:
            return float(sign + val)
        except ValueError:
            return np.nan

    return series.apply(_fix)


# ── Modelling helpers ──────────────────────────────────────────────────────────

def get_split(panel, horizon, predictors):
    """
    Extract a train/test split for a given forecast horizon.

    Uses a strict temporal split — train on 2015-2021, test on 2022-2023 —
    mirroring real forecasting where the model never sees future data.
    Random shuffling is avoided because lagged predictors would leak future
    information across the split boundary if rows were shuffled.

    Parameters
    ----------
    panel      : pd.DataFrame  full country-week panel
    horizon    : int           forecast horizon in weeks (1, 2, or 4)
    predictors : list of str   predictor column names

    Returns
    -------
    X_train, X_test : pd.DataFrame  (unstandardised — scaling handled in fit)
    y_train, y_test : pd.Series     (integer event counts)
    preds_used      : list of str   (columns actually found in the panel)
    """
    outcome_col = f"events_h{horizon}"

    sub = panel.dropna(subset=[outcome_col]).copy()
    sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors="coerce")

    preds_used = [p for p in predictors if p in sub.columns]
    missing    = [p for p in predictors if p not in sub.columns]
    if missing:
        print(f"  WARNING: predictors not in panel (skipped): {missing}")

    for col in preds_used:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan)
    sub = sub.dropna(subset=preds_used + [outcome_col])

    train = sub[sub["week_start"] <= TRAIN_END]
    test  = sub[sub["week_start"] >= TEST_START]

    X_train = train[preds_used].copy()
    y_train = train[outcome_col].astype(int)
    X_test  = test[preds_used].copy()
    y_test  = test[outcome_col].astype(int)

    print(
        f"  h={horizon}: "
        f"train={len(X_train)} rows "
        f"({train['week_start'].min().date()} → {train['week_start'].max().date()}), "
        f"test={len(X_test)} rows "
        f"({test['week_start'].min().date()} → {test['week_start'].max().date()}), "
        f"predictors={len(preds_used)}"
    )

    return X_train, X_test, y_train, y_test, preds_used


def fit_negbinom(X_train, y_train):
    """
    Fit a negative binomial regression model (NB2 parameterisation).

    Predictors are standardised (zero mean, unit variance) before fitting.
    This helps the optimiser converge because all predictors are on the same
    scale — without scaling, a predictor with values in the thousands (e.g.
    mentions_lag1) would dominate the Hessian relative to one in the range
    0-10 (e.g. goldstein_lag1). The scaler is stored on the result object
    so that predict_negbinom applies the same transformation to new data.

    Optimiser cascade: Newton-Raphson first (fastest, second-order), then
    BFGS, L-BFGS, Nelder-Mead as fallbacks. The first method that reports
    successful convergence is used.

    Parameters
    ----------
    X_train : pd.DataFrame  predictor matrix (unstandardised)
    y_train : pd.Series     integer event counts (>= 0)

    Returns
    -------
    Fitted NegativeBinomialResults with ._scaler and ._pred_cols attributes.
    """
    scaler   = StandardScaler()
    X_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )

    X_c   = sm.add_constant(X_scaled, has_constant="add")
    model = sm.NegativeBinomial(endog=y_train, exog=X_c)

    for method in ["newton", "bfgs", "lbfgs", "nm"]:
        try:
            result = model.fit(method=method, maxiter=500, disp=False)
            if result.mle_retvals.get("converged", False):
                result._scaler    = scaler
                result._pred_cols = list(X_train.columns)
                return result
            print(f"    Optimiser '{method}' did not converge — trying next.")
        except Exception as e:
            print(f"    Optimiser '{method}' failed: {type(e).__name__} — trying next.")

    raise RuntimeError(
        "All optimisers failed. Check VIF — multicollinearity may be preventing convergence."
    )


def predict_negbinom(result, X):
    """
    Generate predictions using a fitted NegativeBinomialResults object.

    Applies the same standardisation used during fitting (stored on result
    by fit_negbinom). Clips predictions to zero because event counts cannot
    be negative.

    Parameters
    ----------
    result : fitted NegativeBinomialResults (from fit_negbinom)
    X      : pd.DataFrame  new predictor data, unstandardised, same columns as training

    Returns
    -------
    np.ndarray of predicted event counts (float, >= 0)
    """
    X_scaled = pd.DataFrame(
        result._scaler.transform(X[result._pred_cols]),
        columns=result._pred_cols,
        index=X.index,
    )
    X_c = sm.add_constant(X_scaled, has_constant="add")
    X_c = X_c.reindex(columns=result.model.exog_names, fill_value=0)
    return np.clip(result.predict(X_c), 0, None)


def evaluate(y_true, y_pred):
    """
    Compute RMSE and MAE between observed and predicted event counts.

    Both metrics are on the original count scale (events per country-week),
    making them directly interpretable. RMSE penalises large errors
    quadratically, making it sensitive to conflict spikes. MAE is more
    robust and gives equal weight to all errors.

    NaN values in either array are ignored.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mask   = ~np.isnan(y_true) & ~np.isnan(y_pred)
    rmse   = np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2))
    mae    = np.mean(np.abs(y_true[mask] - y_pred[mask]))
    return {"rmse": round(float(rmse), 4), "mae": round(float(mae), 4)}


def check_vif(X):
    """
    Compute the Variance Inflation Factor (VIF) for each predictor.

    VIF quantifies how much a predictor's variance is inflated by correlation
    with other predictors. A VIF above 10 is commonly taken as indicating
    problematic multicollinearity. The intercept column is excluded.

    Parameters
    ----------
    X : pd.DataFrame  predictor matrix (without intercept)

    Returns
    -------
    pd.DataFrame with column 'VIF', sorted descending, indexed by predictor name
    """
    X_c = sm.add_constant(X, has_constant="add")
    vif = pd.DataFrame(
        {"VIF": [variance_inflation_factor(X_c.values, i)
                 for i in range(X_c.shape[1])]},
        index=X_c.columns,
    ).drop("const")
    return vif.sort_values("VIF", ascending=False)


def build_coef_table(result):
    """
    Extract a clean coefficient table from a fitted NegativeBinomialResults.

    The intercept (const) and dispersion parameter (alpha) are excluded
    from the main table — alpha is reported separately as a model diagnostic.

    Returns
    -------
    pd.DataFrame with columns: Coefficient, Std. Error, z, p-value, CI 2.5%, CI 97.5%
    """
    df = pd.DataFrame({
        "Coefficient": result.params,
        "Std. Error":  result.bse,
        "z":           result.tvalues,
        "p-value":     result.pvalues,
        "CI 2.5%":     result.conf_int()[0],
        "CI 97.5%":    result.conf_int()[1],
    }).round(4)
    return df.drop(index=["const", "alpha"], errors="ignore")


def get_country_split(panel, country, horizon, predictors):
    """
    Extract train/test data for a single country and horizon.

    Identical temporal split to get_split but filtered to one country.
    Additionally drops predictors with zero or near-zero variance in the
    training data — these cause singular matrix errors when fitting on
    small single-country samples (365 training rows vs 1825 in pooled).
    Also drops predictors with pairwise correlation above 0.98, which is
    a stricter threshold than the pooled model to account for amplified
    collinearity in smaller samples.

    Parameters
    ----------
    panel      : pd.DataFrame  full merged panel
    country    : str           country name (must match COUNTRY column)
    horizon    : int           forecast horizon (1, 2, or 4)
    predictors : list of str   candidate predictor columns

    Returns
    -------
    X_train, X_test : pd.DataFrame
    y_train, y_test : pd.Series
    final_preds     : list of str  (after dropping problematic predictors)
    """
    outcome_col = f"events_h{horizon}"

    sub = panel[panel["COUNTRY"] == country].copy()
    sub = sub.dropna(subset=[outcome_col])
    sub[outcome_col] = pd.to_numeric(sub[outcome_col], errors="coerce")

    preds = [p for p in predictors if p in sub.columns]
    for col in preds:
        sub[col] = pd.to_numeric(sub[col], errors="coerce")
    sub = sub.replace([np.inf, -np.inf], np.nan).dropna(subset=preds + [outcome_col])

    train = sub[sub["week_start"] <= TRAIN_END]
    test  = sub[sub["week_start"] >= TEST_START]

    X_train_full = train[preds].copy()
    X_test_full  = test[preds].copy()

    # Drop zero-variance predictors
    std      = X_train_full.std()
    zero_var = std[std < 1e-8].index.tolist()
    if zero_var:
        print(f"    Dropping zero-variance: {zero_var}")

    # Drop highly correlated predictors (|r| > 0.98)
    remaining = [c for c in preds if c not in zero_var]
    high_corr = []
    if len(remaining) > 1:
        corr  = X_train_full[remaining].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        high_corr = [c for c in upper.columns if any(upper[c] > 0.98)]
        if high_corr:
            print(f"    Dropping high-correlation: {high_corr}")

    final_preds = [c for c in preds if c not in zero_var + high_corr]

    return (
        X_train_full[final_preds],
        X_test_full[final_preds],
        train[outcome_col].astype(int),
        test[outcome_col].astype(int),
        final_preds,
    )
