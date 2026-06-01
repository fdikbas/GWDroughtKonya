# -*- coding: utf-8 -*-
"""
================================================================================
BEGINNER-FRIENDLY GUIDE TO THIS SCRIPT
================================================================================

File name
---------
GWMonthlyKNN_full_network_validation_2026.05.31_v6.py

What this script is for
-----------------------
This script is a validation companion for the main GWMonthlyKNN imputation workflow.
The main manuscript originally showed illustrative validation on a small set of wells.
Reviewers asked for a stronger, network-scale validation using many more wells and
using error metrics beyond correlation. This script was written to answer that request.

In the simplest possible terms, the script does the following:

1. It reads the grouped monthly groundwater-level file:

       gwl-monthly.grouped.csv

2. For each well, it looks for fully observed 12-month and 24-month periods.
   A fully observed period means that the well has a real WSE value in every month
   of that candidate period.

3. It temporarily hides one selected observed period from the well.
   This is called a masked-gap test.

   Example:
   - Suppose a well has real observations from January 2015 to December 2015.
   - The script removes those 12 values from the copy used for imputation.
   - The removed values are kept separately as the known truth.
   - The imputation method then tries to reconstruct those missing months.
   - The reconstructed values are compared with the real hidden values.

4. It repeats this procedure across all eligible wells in the network, not only for
   a small hand-picked subset.

5. It evaluates the default GWMonthlyKNN configuration and a few sensitivity variants.
   In this version, the tested KNN variants are:

       KNN_k5_w3_dist  : k = 5, rolling window = 3 months, distance weighting
       KNN_k3_w3_dist  : k = 3, rolling window = 3 months, distance weighting
       KNN_k8_w3_dist  : k = 8, rolling window = 3 months, distance weighting

6. It compares GWMonthlyKNN with simple practical baseline methods:

       baseline_linear_interp
           Linear interpolation within the target well.

       baseline_seasonal_climatology
           Month-of-year average from the target well. For example, missing March
           values are estimated from the available March values of the same well.

       baseline_neighbor_regression
           A simple regression based on the most correlated neighboring well inside
           the same group, if such a donor well has enough overlapping data.

7. It computes both correlation-based and magnitude-based performance metrics:

       R
           Pearson correlation between true hidden values and imputed values.
           It tells whether the reconstructed series follows the same shape.

       R2
           Squared Pearson correlation.
           This is useful as an association metric, but it does not measure bias.

       RMSE
           Root mean squared error.
           Large errors are penalized strongly. Lower is better.

       MAE
           Mean absolute error.
           Average absolute size of the error. Lower is better.

       Bias
           Mean signed error, computed as prediction minus observation.
           Positive Bias means overestimation on average.
           Negative Bias means underestimation on average.

       NSE
           Nash-Sutcliffe efficiency.
           Values closer to 1 are better. Values below 0 indicate that the method
           can be worse than simply using the observed mean for that test segment.

8. It writes CSV outputs and optional summary figures into:

       outputs_full_network_validation/

Why this validation is useful
-----------------------------
A reviewer concern was that showing only a few example wells is not enough to support
network-scale claims. This script addresses that concern by running the same masked-gap
logic across every eligible station in the dataset.

A second reviewer concern was that correlation alone can be misleading. A method can
have high correlation but still be systematically too high or too low. Therefore, this
script reports RMSE, MAE, Bias, and NSE in addition to R and R2.

A third reviewer concern was that KNN should be compared with simpler methods. This
script includes interpolation, seasonal climatology, and neighboring-well regression
baselines so that GWMonthlyKNN is not evaluated in isolation.

Important conceptual point: this is NOT random masking
-----------------------------------------------------
The script does not randomly remove scattered points. Instead, it creates blocked gaps.
This is intentional.

Groundwater monitoring gaps often happen as contiguous periods, for example because of
sensor failure, telemetry outage, site-access problems, maintenance, or quality-control
exclusions. A 12-month or 24-month blocked gap is therefore more realistic and more
challenging than randomly deleting isolated monthly observations.

How the test window is selected
-------------------------------
The setting WINDOW_SELECTION = "max_std" means:

   Among all valid fully observed candidate windows for a station, choose the window
   with the highest standard deviation in WSE.

This makes the validation more demanding because the hidden period is not a flat or
quiet period. It tends to select a period with stronger variation, which is harder to
reconstruct.

What input file is required
---------------------------
The script expects this file in the same working directory:

       gwl-monthly.grouped.csv

The file must contain at least these columns:

       STATION
           Unique station/well identifier.

       GROUP
           Well-group identifier used to restrict candidate donor wells.
           In the manuscript this is based on the PLSS-prefix grouping.

       MSMT_DATE
           Monthly date field. It must be readable by pandas.to_datetime().

       WSE
           Monthly water-surface-elevation value. Missing values must be blank/NaN.

The script will stop with a clear KeyError if any of these columns are missing.

Where to put the files
----------------------
For the simplest use in Spyder on Windows:

1. Put this Python file and gwl-monthly.grouped.csv in the same folder.
2. Set Spyder's working directory to that folder.
3. Run the script with:

       runfile('GWMonthlyKNN_full_network_validation_2026.05.31_v6.py',
               wdir='YOUR_PROJECT_FOLDER')

4. The output folder will be created automatically:

       outputs_full_network_validation

Main user settings
------------------
The most important settings are near the top of the script.

INPUT_CSV
    Name of the grouped monthly input file.
    Default: "gwl-monthly.grouped.csv"

OUTDIR
    Output directory for validation results.
    Default: "outputs_full_network_validation"

RANDOM_SEED
    Included for reproducibility. The current validation window selection is
    deterministic, but keeping a seed is useful if random options are added later.

GAP_LENGTHS
    Masked gap lengths to test.
    Default: [12, 24]
    This means one-year and two-year consecutive artificial gaps.

WINDOW_SELECTION
    Rule for choosing the hidden observed window.
    Default: "max_std"
    This selects the highest-variability fully observed window for each station.

MIN_TRAIN_OBS
    Minimum number of overlapping observed months required to fit the neighboring-well
    regression baseline.
    Default: 24

MIN_OUTSIDE_OBS
    Minimum number of observed months that must remain outside the hidden window.
    Default: 12
    This prevents selecting a validation window when too little observed context remains.

KNN_VARIANTS
    List of GWMonthlyKNN configurations to test.
    Each entry has:
       name    : label used in output tables
       k       : number of KNN neighbors
       w       : rolling-window length in months
       weights : KNN weighting mode, usually "distance" or "uniform"

RUN_BASELINES
    If True, run simple baseline methods.
    Keep True for manuscript/reviewer validation.

MAKE_SUMMARY_FIGS
    If True, create RMSE boxplots as PDF and PNG.

What each function does
-----------------------
month_id(dt)
    Converts a date into a single integer month index.
    This makes it easy to check whether months are consecutive.

contiguous_observed_windows(st_df, L)
    Searches one station's time series for candidate fully observed windows of length L.
    It only accepts a candidate if:
       - the months are consecutive, and
       - every WSE value in that window is observed.

select_window(st_df, L, rule="max_std")
    Selects one candidate validation window for a station.
    With "max_std", it chooses the fully observed window with the largest WSE variability.

rmse(y, yhat), mae(y, yhat), bias(y, yhat), pearson_r(y, yhat), r2(y, yhat), nse(y, yhat)
    Small metric helper functions.
    compute_metrics() is the main combined function used in the evaluation loop.

compute_metrics(y, yhat)
    Computes all metrics for one masked test:
       R, R2, RMSE, MAE, Bias, NSE, and n_test.
    It automatically ignores pairs where either the true value or prediction is missing.

knn_group_impute(group_data, k=5, w=3, weights="distance")
    Recreates the core GWMonthlyKNN group-wise imputation procedure for validation.
    It constructs lag and rolling features, pivots the data by month/year and station,
    and applies scikit-learn's KNNImputer.

baseline_linear_interpolation(st_df_masked, target_dates)
    Estimates the hidden values using time interpolation inside the target well only.

baseline_seasonal_climatology(st_df_masked, target_dates)
    Estimates the hidden values using same-month averages from the target well.

baseline_neighbor_regression(group_df_masked, target_station, target_dates)
    Finds the best available donor well inside the same group and fits a simple linear
    regression from donor WSE to target WSE.
    If no usable donor exists, it returns None.

compute_win_rates(res_df, target_variant="KNN_k5_w3_dist")
    Calculates how often the default GWMonthlyKNN has lower RMSE/MAE than the simple
    baseline methods, and also computes median RMSE skill relative to baselines.

Important implementation detail: long consecutive gaps
------------------------------------------------------
Lag and rolling features are computed before imputation from the masked/original group
series. The script does not fill the first missing month, then recompute lag features,
then fill the second missing month, and so on. In other words, the validation does not
use an iterative month-by-month self-feeding procedure inside the artificial gaps.

This matters because iterative gap filling can accidentally make later predictions depend
on earlier imputed values. Here, the KNNImputer estimates the masked missing values from
the available feature matrix in one imputation pass.

Main output files
-----------------
The script writes the following main outputs:

station_missingness_summary.csv
    One row per station. Includes number of months, number of missing months, first and
    last dates, group identifier, and missing fraction.

selected_windows_per_station.csv
    Shows which 12-month and/or 24-month validation windows were selected for each
    station. Stations without a valid window are marked as no_valid_window.

full_network_masked_gap_metrics.csv
    Detailed station-level validation results. This is the most complete output.
    It includes one row per method, station, and gap length.

summary_by_method_gap.csv
    Paper-ready summary table by method and gap length.
    It reports medians and IQR-style summaries.

winrate_summary_vs_baselines.csv
    Shows how often the default GWMonthlyKNN beats linear interpolation and seasonal
    climatology in terms of RMSE and MAE, plus median RMSE skill.

rmse_boxplot_gap12.pdf / rmse_boxplot_gap12.png
rmse_boxplot_gap24.pdf / rmse_boxplot_gap24.png
    Optional boxplots of RMSE distributions for the tested methods.

How to interpret the output quickly
-----------------------------------
Start with summary_by_method_gap.csv.
Look for the row where:

       method  = GWMonthlyKNN
       variant = KNN_k5_w3_dist

This is the default configuration used in the manuscript.
Compare its RMSE_median, MAE_median, Bias_median, R2_median, and NSE_median with the
baseline methods.

Then inspect winrate_summary_vs_baselines.csv.
This tells whether GWMonthlyKNN improves RMSE/MAE for most comparable wells, not only
whether its median error is smaller.

Then inspect full_network_masked_gap_metrics.csv.
This file is useful for finding individual wells where the method performs poorly, for
example due to irregular hydrographs, local pumping effects, weak donor coherence, or
concurrent gaps in the same group.

Common mistakes and easy fixes
------------------------------
Problem:
    FileNotFoundError: gwl-monthly.grouped.csv not found
Fix:
    Put gwl-monthly.grouped.csv in the current working directory, or edit INPUT_CSV.

Problem:
    KeyError: Missing required columns
Fix:
    Ensure the input CSV contains STATION, GROUP, MSMT_DATE, and WSE.

Problem:
    No valid windows were found
Fix:
    The dataset may not contain fully observed 12- or 24-month blocks after trimming.
    Try lowering the required gap lengths, for example GAP_LENGTHS = [6, 12].

Problem:
    Neighbor regression has many missing results
Fix:
    This is expected when groups are small or donor wells do not overlap enough with the
    target station. The method records this as no_valid_donor_or_overlap.

Problem:
    The script runs slowly
Fix:
    Reduce KNN_VARIANTS, set MAKE_SUMMARY_FIGS = False, or test only one GAP_LENGTHS
    value while debugging.

What this script does NOT claim
-------------------------------
This script does not prove that KNN is universally optimal.
It does not provide formal probabilistic prediction intervals.
It does not prove that PLSS grouping is always hydrologically perfect.
It does not replace site-specific hydrogeologic judgment.

Instead, it provides a transparent, reproducible, reviewer-responsive validation layer
for checking how GWMonthlyKNN behaves across the full eligible monitoring network under
realistic blocked-gap stress tests.

Recommended citation in repository documentation
------------------------------------------------
In the GitHub README or manuscript Data and Software Availability section, this script can
be described as the full-network masked-gap validation script that generates station-level
and method-level outputs used for the revised manuscript's expanded validation results.

================================================================================
END OF BEGINNER-FRIENDLY GUIDE
================================================================================
"""

import os
import numpy as np
import pandas as pd

from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression

# progress bar (safe)
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(x, **kwargs):
        return x

# ----------------------------
# USER CONFIG
# ----------------------------
INPUT_CSV = "gwl-monthly.grouped.csv"

OUTDIR = "outputs_full_network_validation"
os.makedirs(OUTDIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# Window lengths to test
GAP_LENGTHS = [12, 24]  # months

# Deterministic selection: choose the candidate window with MAX std(WSE) to stress-test
WINDOW_SELECTION = "max_std"  # ("max_std" recommended)

# Minimum observed months required outside the withheld window to fit baselines robustly
MIN_TRAIN_OBS = 24


# Minimum observed months outside the withheld window for a station to be eligible
MIN_OUTSIDE_OBS = 12
# KNN variants (keep small for runtime; expand if desired)
# - These will all run on the same selected windows for each station.
KNN_VARIANTS = [
    {"name": "KNN_k5_w3_dist", "k": 5, "w": 3, "weights": "distance"},
    {"name": "KNN_k3_w3_dist", "k": 3, "w": 3, "weights": "distance"},
    {"name": "KNN_k8_w3_dist", "k": 8, "w": 3, "weights": "distance"},
    # Uncomment for deeper sensitivity:
    # {"name": "KNN_k5_w6_dist", "k": 5, "w": 6, "weights": "distance"},
    # {"name": "KNN_k5_w3_unif", "k": 5, "w": 3, "weights": "uniform"},
]

# Baselines to include
RUN_BASELINES = True

# Figures (optional)
MAKE_SUMMARY_FIGS = True


# ----------------------------
# Utilities
# ----------------------------
def month_id(dt):
    """Map datetime to an integer month index (year*12 + month) for contiguity checks."""
    return dt.year * 12 + dt.month


def contiguous_observed_windows(st_df, L):
    """
    Return candidate windows (start_idx, end_idx inclusive) of length L months
    such that:
    - months are consecutive (no missing months in index)
    - WSE is observed (not NaN) for all L months
    """
    st_df = st_df.sort_values("MSMT_DATE").copy()
    mids = st_df["MSMT_DATE"].map(month_id).to_numpy()
    obs = (~st_df["WSE"].isna()).to_numpy()

    cands = []
    n = len(st_df)
    for i in range(n - L + 1):
        j = i + L - 1
        # consecutive months
        if mids[j] - mids[i] != (L - 1):
            continue
        # all observed
        if not obs[i:j + 1].all():
            continue
        cands.append((i, j))
    return cands


def select_window(st_df, L, rule="max_std"):
    cands = contiguous_observed_windows(st_df, L)
    if len(cands) == 0:
        return None

    # Require that the station retains enough observed context outside the withheld window
    total_obs = int(st_df["WSE"].notna().sum())
    eligible = []
    for (i, j) in cands:
        outside_obs = total_obs - L  # window is fully observed by construction
        if outside_obs >= MIN_OUTSIDE_OBS:
            eligible.append((i, j))

    if len(eligible) == 0:
        return None

    if rule == "max_std":
        best = None
        best_val = -np.inf
        srt = st_df.sort_values("MSMT_DATE")
        for (i, j) in eligible:
            vals = srt["WSE"].iloc[i:j + 1].to_numpy()
            v = float(np.nanstd(vals))
            if v > best_val:
                best_val = v
                best = (i, j)
        return best
    else:
        return eligible[len(eligible) // 2]


def rmse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.sqrt(np.nanmean((y - yhat) ** 2)))


def mae(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.nanmean(np.abs(y - yhat)))


def bias(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    return float(np.nanmean(yhat - y))


def pearson_r(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    if len(y) < 2:
        return np.nan
    if np.nanstd(y) == 0 or np.nanstd(yhat) == 0:
        return np.nan
    return float(np.corrcoef(y, yhat)[0, 1])


def r2(y, yhat):
    r = pearson_r(y, yhat)
    return float(r * r) if np.isfinite(r) else np.nan


def nse(y, yhat):
    y = np.asarray(y)
    yhat = np.asarray(yhat)
    den = np.nansum((y - np.nanmean(y)) ** 2)
    if den == 0:
        return np.nan
    num = np.nansum((y - yhat) ** 2)
    return float(1 - num / den)


def compute_metrics(y, yhat):
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)

    mask = np.isfinite(y) & np.isfinite(yhat)
    n_eff = int(mask.sum())

    if n_eff == 0:
        return {"R": np.nan, "R2": np.nan, "RMSE": np.nan, "MAE": np.nan, "Bias": np.nan, "NSE": np.nan, "n_test": 0}

    yy = y[mask]
    pp = yhat[mask]

    out = {}
    out["n_test"] = n_eff

    # Correlation-based
    if n_eff >= 2 and np.nanstd(yy) > 0 and np.nanstd(pp) > 0:
        r = float(np.corrcoef(yy, pp)[0, 1])
        out["R"] = r
        out["R2"] = float(r * r)
    else:
        out["R"] = np.nan
        out["R2"] = np.nan

    # Magnitude-based
    out["RMSE"] = float(np.sqrt(np.nanmean((yy - pp) ** 2)))
    out["MAE"] = float(np.nanmean(np.abs(yy - pp)))
    out["Bias"] = float(np.nanmean(pp - yy))

    den = float(np.nansum((yy - np.nanmean(yy)) ** 2))
    if den == 0:
        out["NSE"] = np.nan
    else:
        num = float(np.nansum((yy - pp) ** 2))
        out["NSE"] = float(1 - num / den)

    return out


# ----------------------------
# Core GWMonthlyKNN imputation (group-wise)
# ----------------------------
def knn_group_impute(group_data, k=5, w=3, weights="distance"):
    """
    Replicates the core behavior of GWMonthlyKNN:
    - Features per station: LAG_1, LAG_2, ROLLING_MEAN(w), ROLLING_STD(w)
    - Pivot indexed by (MONTH, YEAR)
    - IMPORTANT: MONTH and YEAR included in feature matrix (reset_index)
    """
    gd = group_data.copy()

    gd["LAG_1"] = gd.groupby("STATION")["WSE"].shift(1)
    gd["LAG_2"] = gd.groupby("STATION")["WSE"].shift(2)
    gd["ROLLING_MEAN"] = gd.groupby("STATION")["WSE"].transform(
        lambda x: x.rolling(window=w, min_periods=1).mean()
    )
    gd["ROLLING_STD"] = gd.groupby("STATION")["WSE"].transform(
        lambda x: x.rolling(window=w, min_periods=1).std()
    )

    pivot = gd.pivot(
        index=["MONTH", "YEAR"],
        columns="STATION",
        values=["WSE", "LAG_1", "LAG_2", "ROLLING_MEAN", "ROLLING_STD"],
    )
    pivot.columns = ["_".join(col).strip() for col in pivot.columns.values]

    pivot = pivot.reset_index()

    # keep_empty_features avoids dropping all-NaN feature columns (sklearn>=1.2)
    try:
        imputer = KNNImputer(n_neighbors=k, weights=weights, keep_empty_features=True)
    except TypeError:
        imputer = KNNImputer(n_neighbors=k, weights=weights)

    arr = imputer.fit_transform(pivot)

    out = pd.DataFrame(arr, columns=pivot.columns).set_index(["MONTH", "YEAR"])
    return out


# ----------------------------
# Baseline methods
# ----------------------------
def baseline_linear_interpolation(st_df_masked, target_dates):
    s = st_df_masked.set_index("MSMT_DATE")["WSE"].copy()
    s_interp = s.interpolate(method="time")
    return s_interp.reindex(target_dates).to_numpy()


def baseline_seasonal_climatology(st_df_masked, target_dates):
    tmp = st_df_masked.copy()
    tmp["MONTH"] = tmp["MSMT_DATE"].dt.month
    clim = tmp.groupby("MONTH")["WSE"].mean()
    preds = []
    for d in target_dates:
        preds.append(float(clim.get(d.month, np.nan)))
    return np.array(preds)


def baseline_neighbor_regression(group_df_masked, target_station, target_dates):
    g = group_df_masked.copy()
    wide = g.pivot_table(index="MSMT_DATE", columns="STATION", values="WSE", aggfunc="mean").sort_index()

    if target_station not in wide.columns:
        return None

    y = wide[target_station]
    donors = [c for c in wide.columns if c != target_station]
    if len(donors) == 0:
        return None

    best_donor = None
    best_r = -np.inf
    for d in donors:
        pair = pd.concat([y, wide[d]], axis=1).dropna()
        if len(pair) < MIN_TRAIN_OBS:
            continue
        r = np.corrcoef(pair.iloc[:, 0], pair.iloc[:, 1])[0, 1]
        if np.isfinite(r) and abs(r) > best_r:
            best_r = abs(r)
            best_donor = d

    if best_donor is None:
        return None

    pair = pd.concat([y, wide[best_donor]], axis=1).dropna()
    if len(pair) < MIN_TRAIN_OBS:
        return None

    X = pair.iloc[:, 1].to_numpy().reshape(-1, 1)
    Y = pair.iloc[:, 0].to_numpy()

    model = LinearRegression()
    model.fit(X, Y)

    donor_series = wide[best_donor].reindex(target_dates)
    if donor_series.isna().any():
        return None

    preds = model.predict(donor_series.to_numpy().reshape(-1, 1))
    return preds


# ----------------------------
# Load + preprocess data
# ----------------------------
data = pd.read_csv(INPUT_CSV)
data["MSMT_DATE"] = pd.to_datetime(data["MSMT_DATE"])

needed = {"STATION", "GROUP", "MSMT_DATE", "WSE"}
missing_cols = needed - set(data.columns)
if missing_cols:
    raise KeyError(f"Missing required columns in {INPUT_CSV}: {missing_cols}")


def filter_observation_range(group):
    first_obs_index = group["WSE"].first_valid_index()
    last_obs_index = group["WSE"].last_valid_index()
    return group.loc[first_obs_index:last_obs_index]

# Robust observation-window trimming WITHOUT groupby.apply (avoids pandas apply deprecation and STATION loss)
# Keep only records between the first and last observed (non-missing) WSE month for each station.
data = data.sort_values(["STATION", "MSMT_DATE"]).copy()

first_map = data.loc[data["WSE"].notna()].groupby("STATION")["MSMT_DATE"].min()
last_map  = data.loc[data["WSE"].notna()].groupby("STATION")["MSMT_DATE"].max()

data["_first_obs_date"] = data["STATION"].map(first_map)
data["_last_obs_date"]  = data["STATION"].map(last_map)

data = data[(data["MSMT_DATE"] >= data["_first_obs_date"]) & (data["MSMT_DATE"] <= data["_last_obs_date"])].copy()
data = data.drop(columns=["_first_obs_date", "_last_obs_date"])


data["MONTH"] = data["MSMT_DATE"].dt.month
data["YEAR"] = data["MSMT_DATE"].dt.year

station_stats = (
    data.groupby("STATION")
    .agg(
        n_months=("WSE", "size"),
        n_missing=("WSE", lambda x: int(x.isna().sum())),
        first_date=("MSMT_DATE", "min"),
        last_date=("MSMT_DATE", "max"),
        group=("GROUP", lambda x: x.iloc[0]),
    )
    .reset_index()
)
station_stats["missing_frac"] = station_stats["n_missing"] / station_stats["n_months"]
station_stats.to_csv(os.path.join(OUTDIR, "station_missingness_summary.csv"), index=False)

# ----------------------------
# Select one evaluation window per station per gap length
# ----------------------------
window_rows = []
for st in tqdm(station_stats["STATION"].tolist(), desc="Selecting windows", unit="station"):
    st_df = data[data["STATION"] == st].sort_values("MSMT_DATE").copy()
    for L in GAP_LENGTHS:
        sel = select_window(st_df, L, rule=WINDOW_SELECTION)
        if sel is None:
            window_rows.append(
                {"STATION": st, "GROUP": st_df["GROUP"].iloc[0], "gap_len": L, "status": "no_valid_window"}
            )
            continue
        i, j = sel
        start = st_df["MSMT_DATE"].iloc[i]
        end = st_df["MSMT_DATE"].iloc[j]
        window_rows.append(
            {
                "STATION": st,
                "GROUP": st_df["GROUP"].iloc[0],
                "gap_len": L,
                "status": "ok",
                "start_date": start,
                "end_date": end,
            }
        )

windows = pd.DataFrame(window_rows)
windows.to_csv(os.path.join(OUTDIR, "selected_windows_per_station.csv"), index=False)

# ----------------------------
# Run evaluation
# ----------------------------
results = []

valid_windows = windows[windows["status"] == "ok"].copy()
if valid_windows.empty:
    raise RuntimeError("No valid windows were found for evaluation. Check input data continuity/coverage.")

for row in tqdm(valid_windows.itertuples(index=False), total=len(valid_windows), desc="Evaluating stations", unit="test"):
    st = row.STATION
    grp = row.GROUP
    L = int(row.gap_len)
    start = pd.to_datetime(row.start_date)
    end = pd.to_datetime(row.end_date)

    st_df_full = data[data["STATION"] == st].sort_values("MSMT_DATE").copy()
    mask_win = (st_df_full["MSMT_DATE"] >= start) & (st_df_full["MSMT_DATE"] <= end)

    true_vals = st_df_full.loc[mask_win, "WSE"].to_numpy()
    target_dates = st_df_full.loc[mask_win, "MSMT_DATE"].to_list()

    if len(true_vals) != L:
        continue

    gdf = data[data["GROUP"] == grp].copy()
    mask_g = (gdf["STATION"] == st) & (gdf["MSMT_DATE"] >= start) & (gdf["MSMT_DATE"] <= end)
    gdf.loc[mask_g, "WSE"] = np.nan

    # Baselines
    if RUN_BASELINES:
        st_df_masked = st_df_full.copy()
        st_df_masked.loc[mask_win, "WSE"] = np.nan

        pred_lin = baseline_linear_interpolation(st_df_masked, target_dates)
        m_lin = compute_metrics(true_vals, pred_lin)
        results.append(
            dict(method="baseline_linear_interp", variant="", STATION=st, GROUP=grp, gap_len=L,
                 start_date=start, end_date=end, **m_lin)
        )

        pred_clim = baseline_seasonal_climatology(st_df_masked, target_dates)
        m_clim = compute_metrics(true_vals, pred_clim)
        results.append(
            dict(method="baseline_seasonal_climatology", variant="", STATION=st, GROUP=grp, gap_len=L,
                 start_date=start, end_date=end, **m_clim)
        )

        pred_reg = baseline_neighbor_regression(gdf, st, target_dates)
        if pred_reg is not None:
            m_reg = compute_metrics(true_vals, pred_reg)
            results.append(
                dict(method="baseline_neighbor_regression", variant="", STATION=st, GROUP=grp, gap_len=L,
                     start_date=start, end_date=end, **m_reg)
            )
        else:
            results.append(
                dict(method="baseline_neighbor_regression", variant="", STATION=st, GROUP=grp, gap_len=L,
                     start_date=start, end_date=end, R=np.nan, R2=np.nan, RMSE=np.nan, MAE=np.nan,
                     Bias=np.nan, NSE=np.nan, n_test=L, note="no_valid_donor_or_overlap")
            )

    # KNN variants
    for v in KNN_VARIANTS:
        imputed = knn_group_impute(gdf, k=v["k"], w=v["w"], weights=v["weights"])

        col = f"WSE_{st}"
        if col not in imputed.columns:
            continue

        pred = []
        for d in target_dates:
            key = (d.month, d.year)
            pred.append(imputed.loc[key, col] if key in imputed.index else np.nan)
        pred = np.array(pred, dtype=float)

        m_knn = compute_metrics(true_vals, pred)
        results.append(
            dict(method="GWMonthlyKNN", variant=v["name"], k=v["k"], w=v["w"], weights=v["weights"],
                 STATION=st, GROUP=grp, gap_len=L, start_date=start, end_date=end, **m_knn)
        )

res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUTDIR, "full_network_masked_gap_metrics.csv"), index=False)

# ----------------------------
# Summaries (paper-ready)
# ----------------------------
res_df_clean = res_df.dropna(subset=["RMSE"], how="all").copy()

summary = (
    res_df_clean.groupby(["method", "variant", "gap_len"])
    .agg(
        n_tests=("RMSE", "count"),
        RMSE_median=("RMSE", "median"),
        RMSE_IQR=("RMSE", lambda x: float(np.nanpercentile(x, 75) - np.nanpercentile(x, 25))),
        MAE_median=("MAE", "median"),
        Bias_median=("Bias", "median"),
        R2_median=("R2", "median"),
        NSE_median=("NSE", "median"),
    )
    .reset_index()
)

summary.to_csv(os.path.join(OUTDIR, "summary_by_method_gap.csv"), index=False)

# ----------------------------
# Additional summary: win rates vs simple baselines (station-level)
# ----------------------------
# This quantifies how often GWMonthlyKNN improves magnitude-based error metrics (e.g., RMSE)
# relative to interpolation/climatology, addressing reviewer concern beyond correlation metrics.
def compute_win_rates(res_df, target_variant="KNN_k5_w3_dist"):
    out_rows = []
    for L in sorted(res_df["gap_len"].dropna().unique()):
        sub = res_df[res_df["gap_len"] == L].copy()

        # Pull method tables
        knn = sub[(sub["method"] == "GWMonthlyKNN") & (sub["variant"] == target_variant)][["STATION", "RMSE", "MAE"]].rename(
            columns={"RMSE": "RMSE_knn", "MAE": "MAE_knn"}
        )
        lin = sub[sub["method"] == "baseline_linear_interp"][["STATION", "RMSE", "MAE"]].rename(
            columns={"RMSE": "RMSE_lin", "MAE": "MAE_lin"}
        )
        clim = sub[sub["method"] == "baseline_seasonal_climatology"][["STATION", "RMSE", "MAE"]].rename(
            columns={"RMSE": "RMSE_clim", "MAE": "MAE_clim"}
        )

        # Merge where comparable
        m_lin = knn.merge(lin, on="STATION", how="inner").dropna()
        m_clim = knn.merge(clim, on="STATION", how="inner").dropna()

        def frac_better(a, b):
            if len(a) == 0:
                return np.nan
            return float(np.mean(a < b))

        # RMSE win rates
        out_rows.append({
            "gap_len": int(L),
            "comparison": "KNN vs linear interpolation",
            "n_comparable": int(len(m_lin)),
            "frac_RMSE_better": frac_better(m_lin["RMSE_knn"].values, m_lin["RMSE_lin"].values),
            "frac_MAE_better": frac_better(m_lin["MAE_knn"].values, m_lin["MAE_lin"].values),
        })
        out_rows.append({
            "gap_len": int(L),
            "comparison": "KNN vs seasonal climatology",
            "n_comparable": int(len(m_clim)),
            "frac_RMSE_better": frac_better(m_clim["RMSE_knn"].values, m_clim["RMSE_clim"].values),
            "frac_MAE_better": frac_better(m_clim["MAE_knn"].values, m_clim["MAE_clim"].values),
        })

        # Skill scores (median 1 - RMSE_knn/RMSE_baseline)
        if len(m_lin) > 0:
            out_rows.append({
                "gap_len": int(L),
                "comparison": "RMSE skill vs linear interpolation",
                "n_comparable": int(len(m_lin)),
                "median_skill": float(np.nanmedian(1.0 - (m_lin["RMSE_knn"].values / m_lin["RMSE_lin"].values))),
            })
        if len(m_clim) > 0:
            out_rows.append({
                "gap_len": int(L),
                "comparison": "RMSE skill vs seasonal climatology",
                "n_comparable": int(len(m_clim)),
                "median_skill": float(np.nanmedian(1.0 - (m_clim["RMSE_knn"].values / m_clim["RMSE_clim"].values))),
            })

    return pd.DataFrame(out_rows)

win_df = compute_win_rates(res_df_clean, target_variant="KNN_k5_w3_dist")
win_df.to_csv(os.path.join(OUTDIR, "winrate_summary_vs_baselines.csv"), index=False)

# Optional: figures
if MAKE_SUMMARY_FIGS:
    import matplotlib.pyplot as plt

    for L in GAP_LENGTHS:
        sub = res_df_clean[(res_df_clean["gap_len"] == L) & (res_df_clean["RMSE"].notna())].copy()
        if sub.empty:
            continue

        labels = []
        data_box = []

        for m in ["baseline_linear_interp", "baseline_seasonal_climatology", "baseline_neighbor_regression"]:
            vals = sub[sub["method"] == m]["RMSE"].to_numpy()
            if len(vals) > 0:
                labels.append(m)
                data_box.append(vals)

        knn_sub = sub[sub["method"] == "GWMonthlyKNN"]
        for v in KNN_VARIANTS:
            vals = knn_sub[knn_sub["variant"] == v["name"]]["RMSE"].to_numpy()
            if len(vals) > 0:
                labels.append(v["name"])
                data_box.append(vals)

        plt.figure(figsize=(11, 4), dpi=300)
        try:
            plt.boxplot(data_box, tick_labels=labels, showfliers=False)  # Matplotlib >= 3.9
        except TypeError:
            plt.boxplot(data_box, labels=labels, showfliers=False)       # Matplotlib < 3.9
        plt.xticks(rotation=25, ha="right")
        plt.ylabel("RMSE (WSE units)")
        plt.title(f"Full-Network Masked-Gap RMSE Distribution (gap = {L} months)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTDIR, f"rmse_boxplot_gap{L}.pdf"))
        plt.savefig(os.path.join(OUTDIR, f"rmse_boxplot_gap{L}.png"))
        plt.close()

print("DONE.")
print("Outputs written to:", os.path.abspath(OUTDIR))
print("- station_missingness_summary.csv")
print("- selected_windows_per_station.csv")
print("- full_network_masked_gap_metrics.csv")
print("- summary_by_method_gap.csv")
