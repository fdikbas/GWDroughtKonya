# -*- coding: utf-8 -*-
"""
GWMonthlyKNN_full_network_validation_v1.py

Purpose:
- Address reviewer concern: evaluation based on only 8 wells.
- Run masked-gap validation across ALL wells where valid observed 12- and/or 24-month windows exist.
- Compute correlation + magnitude-based error metrics (RMSE/MAE/Bias/NSE).
- Compare against simple baselines (linear interpolation, seasonal climatology, neighboring-well regression).
- Optional sensitivity runs for KNN hyperparameters (k, rolling window w, weights).

Designed for Spyder/Windows: parameterless runfile(...).
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
