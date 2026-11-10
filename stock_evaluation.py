"""

Evaluation script for the proposal "Random versus Walk-Forward Evaluation in Stock Direction Prediction".

Question: how much higher does a model's score look when it is evaluated with random cross-validation

instead of time-ordered procedures, on the same stocks, models, features and test days?



Primary procedures (all scored on the same days, namely the last five sixths of each series)

  random_kfold        : stratified 5-fold cross-validation with shuffling

  blocked_purged      : 5 consecutive blocks; training days within WINDOW + horizon days before and after

                        the test block are removed; training days further away (also later ones) are kept

  walk_forward_purged : expanding window; the series is cut into 6 blocks, the first is training only,

                        each later block is predicted using all earlier observations except the final

                        horizon training observations immediately before the test block



Optional analyses (each is switched on by its own option)

  --controls          mechanism controls, written to controls_by_stock.csv and controls_differences.csv

      walk_forward        : unpurged expanding-window diagnostic

      walk_forward_fixed  : walk-forward with a rolling training window of the size of the first block

      walk_forward_purged : primary purged walk-forward result, repeated in controls

      blocked_small       : walk-forward test blocks, trained on a random subsample

                            (size of the first block) of all other days, earlier and later

      random_small        : random 5-fold, but trained on a random subsample (size of

                            the first block) of each training set

      The controls also compare purged and unpurged walk-forward.

  --placebo N         N placebo runs: the target series is shifted by a random number of days (independent

                      per stock, horizon and run; features stay in place), so no feature can contain

                      information about the target. All three primary procedures are run; output

                      placebo_by_run.csv and placebo_summary.csv.

                      Use --placebo-models to choose models (default: gboost) and --jobs to use several cores.

  --purge-sweep       blocked_purged with purge distances 0, h, WINDOW + h and 2 * (WINDOW + h) days,

                      with the number of training rows and the difference to random_kfold;

                      output purge_sweep_*.csv

  --target single_day target = 1 if the return of day t + h alone (close[t+h] against close[t+h-1])

                      is positive, instead of the return over the whole horizon (identical for horizon 1).

                      Run it with its own --out.

  --seeds 43 44 ...   noise baseline for RQ3 (random procedure rerun with other seeds)



Models (fixed settings, no tuning): logistic regression, histogram gradient boosting.

Target (default): 1 if the close is higher after `horizon` trading days than today, else 0.

Features (data up to day t only): ret_1, ret_2, ret_3, ret_5, vol_20, ma_ratio_20, vol_chg_20

(vol_chg_20 is left out if the series has no volume).



Statistics

  AUC (main) and accuracy (secondary, threshold 0.5, compared with the share of the larger class)

  per stock, model, horizon and procedure, from pooled out-of-sample predictions.

  Paired differences between procedures per stock; Wilcoxon signed-rank test across stocks;

  median difference with a percentile bootstrap interval over stocks (approximate: stocks share the same calendar).

  H1 and H2 (gradient boosting, horizon 5, AUC, two-sided) are adjusted together with the Holm procedure.

  Decision rule (proposal, section 3.6): a hypothesis is supported if the median difference is at least

  --min-effect (default 0.02 AUC) AND the Holm-adjusted p-value is below 0.05.

  All other rows are descriptive and carry no adjusted p-value.



Input: either the combined file (--combined prices_all.csv with --sample main|confirmation|test)

or one CSV per stock in --data-dir (Nasdaq.com export "Date, Close/Last, Volume, Open, High, Low", or any

file with Date, Close or Adj Close, and optionally Volume). The stock name is the file name without extension.

All series are cut to the study period --start to --end (default 2016-11-02 to 2026-10-30).



Usage

  python stock_eval.py --combined prices_all.csv --sample main --horizons 1 5 --controls --out results_main_purged

  python stock_eval.py --combined prices_all.csv --sample main --horizons 1 5 --placebo 100 --jobs 4 --out results_main_purged

  python stock_eval.py --combined prices_all.csv --sample main --horizons 5 --purge-sweep --out results_main_purged

  python stock_eval.py --combined prices_all.csv --sample main --horizons 5 --target single_day --out results_main_single_purged

  python stock_eval.py --combined prices_test.csv --sample test --horizons 1 5 --full --jobs 4 --out results_test

  python stock_eval.py --combined prices_test.csv --sample test --horizons 5 --target single_day --out results_test_single

  python stock_eval.py --demo --placebo 3 --out demo_out

Outputs in --out: data_summary.csv, results_by_stock.csv, differences.csv, hypotheses.csv,

model_ranking.csv, run_info.txt (+ the files of the optional analyses)

"""

import argparse

import hashlib

import platform

import sys

import warnings

from datetime import date

from pathlib import Path



import numpy as np

import pandas as pd

import scipy

import sklearn

from joblib import Parallel, delayed

from scipy import stats

from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, roc_auc_score

from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit

from sklearn.pipeline import make_pipeline

from sklearn.preprocessing import StandardScaler



SEED = 42

N_SPLITS = 5

WINDOW = 20

MIN_ROWS = 500

JUMP_WARNING = 0.40

MIN_SHIFT = 60

MIN_EFFECT = 0.02              # CHANGED: minimum median AUC difference of the decision rule (section 3.6)

STUDY_START = "2016-11-02"     # CHANGED: study period of the proposal (section 3.4)

STUDY_END = "2026-10-30"



# These are the procedures used in the main results and hypotheses.

PROCEDURES = ("random_kfold", "blocked_purged", "walk_forward_purged")



# Optional diagnostic/control procedures. walk_forward is deliberately

# unpurged and is not the primary chronological benchmark.

CONTROL_PROCEDURES = (

    "walk_forward",

    "walk_forward_fixed",

    "walk_forward_purged",

    "blocked_small",

    "random_small",

)



COMPARISONS = (

    ("random_kfold minus walk_forward_purged",

     "random_kfold", "walk_forward_purged"),      # RQ1, H1

    ("blocked_purged minus walk_forward_purged",

     "blocked_purged", "walk_forward_purged"),    # RQ1

    ("random_kfold minus blocked_purged",

     "random_kfold", "blocked_purged"),           # RQ1, H2

)



CONTROL_COMPARISONS = (

    ("(a) random_kfold minus blocked_purged [neighbouring days, large size]",

     "random_kfold", "blocked_purged"),

    ("(b) random_small minus blocked_small [neighbouring days, small size]",

     "random_small", "blocked_small"),

    ("(c) random_kfold minus random_small [training size, with neighbours]",

     "random_kfold", "random_small"),

    ("(d) walk_forward minus walk_forward_fixed [expanding vs fixed window]",

     "walk_forward", "walk_forward_fixed"),

    ("(e) blocked_small minus walk_forward_fixed [later days in training]",

     "blocked_small", "walk_forward_fixed"),

    ("(f) random_small minus walk_forward_fixed [gap at equal training size, RQ2]",

     "random_small", "walk_forward_fixed"),

    ("(g) walk_forward_purged minus walk_forward [last h rows removed]",

     "walk_forward_purged", "walk_forward"),

    ("random_kfold minus walk_forward_purged",

     "random_kfold", "walk_forward_purged"),

)



HYPOTHESES = {

    "H1": ("gboost", 5, "auc",

           "random_kfold minus walk_forward_purged"),

    "H2": ("gboost", 5, "auc",

           "random_kfold minus blocked_purged"),

}





# ----------------------------------------------------------------------------

# Data

# ----------------------------------------------------------------------------

def read_price_csv(path):

    """Read one price file and return close and (if present) volume."""

    df = pd.read_csv(path, thousands=",")

    df.columns = [c.strip().lower() for c in df.columns]

    df = df.rename(columns={"close/last": "close"})

    date_col = "date" if "date" in df.columns else df.columns[0]

    for c in df.columns:

        if c != date_col and not pd.api.types.is_numeric_dtype(df[c]):

            df[c] = pd.to_numeric(

                df[c].astype(str).str.replace(r"[$,]", "", regex=True),

                errors="coerce",

            )

    df[date_col] = pd.to_datetime(df[date_col])

    df = df.sort_values(date_col).drop_duplicates(subset=date_col).set_index(date_col)

    price_col = "adj close" if "adj close" in df.columns else "close"

    out = pd.DataFrame({"close": df[price_col].astype(float)})

    if "volume" in df.columns:

        out["volume"] = df["volume"].astype(float)

    return out.dropna(subset=["close"])





def restrict_period(name, prices, start, end):

    """CHANGED: cut a series to the study period and warn if it does not cover it."""

    if start is None and end is None:

        return prices

    covered_start = start is None or prices.index.min() <= pd.Timestamp(start)

    covered_end = end is None or prices.index.max() >= pd.Timestamp(end)

    if not (covered_start and covered_end):

        print(

            f"  WARNING [{name}]: series runs from {prices.index.min().date()} to "

            f"{prices.index.max().date()} and does not cover the study period"

        )

    return prices.loc[start:end]





def check_prices(name, prices):

    """Print warnings for data problems; never changes the data."""

    jumps = prices["close"].pct_change().abs()

    if jumps.max() > JUMP_WARNING:

        print(

            f"  WARNING [{name}]: one-day move of {jumps.max():.0%} on "

            f"{jumps.idxmax().date()}; check for an unadjusted split"

        )

    if (prices["close"] <= 0).any():

        raise SystemExit(f"[{name}] contains non-positive prices")





def demo_prices(n_stocks=20, n_days=3000):

    """Synthetic prices with volatility clustering and almost unpredictable direction."""

    rng = np.random.default_rng(SEED)

    out = {}

    for k in range(n_stocks):

        log_sigma = np.zeros(n_days)

        for i in range(1, n_days):

            log_sigma[i] = 0.97 * log_sigma[i - 1] + 0.15 * rng.standard_normal()

        r = 0.0003 + 0.01 * np.exp(log_sigma) * rng.standard_normal(n_days)

        close = 100 * np.exp(np.cumsum(r))

        vol = 1e6 * np.exp(0.5 * log_sigma + 0.2 * rng.standard_normal(n_days))

        idx = pd.bdate_range("2010-01-01", periods=n_days)

        out[f"DEMO{k + 1}"] = pd.DataFrame(

            {"close": close, "volume": vol}, index=idx

        )

    return out





def compute_features(prices):

    close = prices["close"]

    r = close.pct_change()

    feats = pd.DataFrame({

        "ret_1": r,

        "ret_2": r.shift(1),

        "ret_3": r.shift(2),

        "ret_5": close.pct_change(5),

        "vol_20": r.rolling(WINDOW).std(),

        "ma_ratio_20": close / close.rolling(WINDOW).mean() - 1,

    })

    if "volume" in prices.columns:

        v = prices["volume"].replace(0, np.nan)

        if v.notna().mean() > 0.9:

            feats["vol_chg_20"] = v / v.rolling(WINDOW).mean() - 1

    return feats





def check_no_lookahead(prices):

    """Features on day t must not change when later prices are removed."""

    cut = len(prices) // 2

    full = compute_features(prices)

    part = compute_features(prices.iloc[:cut])

    a = full.iloc[:cut].to_numpy()

    b = part.to_numpy()

    if not np.allclose(a, b, equal_nan=True):

        raise AssertionError("features use information from later days")





def make_dataset(prices, horizon, target="cumulative"):

    """Create cumulative or single-day-ahead direction labels."""

    close = prices["close"]

    feats = compute_features(prices)

    if target == "single_day":

        fwd = close.shift(-horizon) / close.shift(-(horizon - 1)) - 1

    else:

        fwd = close.shift(-horizon) / close - 1

    data = feats.assign(y=(fwd > 0).astype(int))[fwd.notna()]

    data = data.replace([np.inf, -np.inf], np.nan).dropna()

    return data.drop(columns="y"), data["y"]





# ----------------------------------------------------------------------------

# Models and procedures

# ----------------------------------------------------------------------------

def model_factories(seed=SEED):

    return {

        "logreg": lambda: make_pipeline(

            StandardScaler(), LogisticRegression(max_iter=1000)

        ),

        "gboost": lambda: HistGradientBoostingClassifier(

            max_iter=100, learning_rate=0.05, max_depth=3, random_state=seed

        ),

    }





def fit_predict(factory, X, y, tr, te):

    if len(tr) == 0 or y.iloc[tr].nunique() < 2:

        return np.full(len(te), np.nan)

    model = factory().fit(X.iloc[tr], y.iloc[tr])

    return model.predict_proba(X.iloc[te])[:, 1]





def oof_random(factory, X, y, seed=SEED):

    pred = np.full(len(y), np.nan)

    for tr, te in StratifiedKFold(

        N_SPLITS, shuffle=True, random_state=seed

    ).split(X, y):

        pred[te] = fit_predict(factory, X, y, tr, te)

    return pred





def oof_blocked_purged(factory, X, y, gap, return_train_sizes=False):

    pred = np.full(len(y), np.nan)

    sizes = []

    for tr, te in KFold(N_SPLITS, shuffle=False).split(X):

        lo, hi = te.min(), te.max()

        tr = tr[(tr < lo - gap) | (tr > hi + gap)]

        sizes.append(len(tr))

        pred[te] = fit_predict(factory, X, y, tr, te)

    return (pred, float(np.mean(sizes))) if return_train_sizes else pred





def wf_splits(X):

    return list(TimeSeriesSplit(n_splits=N_SPLITS).split(X))





def oof_walk_forward(factory, X, y):

    """Unpurged expanding-window predictions; used only as a diagnostic control."""

    pred = np.full(len(y), np.nan)

    for tr, te in wf_splits(X):

        pred[te] = fit_predict(factory, X, y, tr, te)

    return pred





def oof_walk_forward_purged(factory, X, y, horizon):

    """

    Primary chronological procedure. For each test block, exclude the final

    'horizon' training observations immediately before the test block.

    This prevents training labels whose forward outcome period overlaps the

    beginning of the test block.

    """

    pred = np.full(len(y), np.nan)

    for _, te in wf_splits(X):

        train_end = max(0, int(te.min()) - int(horizon))

        tr = np.arange(train_end)

        pred[te] = fit_predict(factory, X, y, tr, te)

    return pred





def oof_walk_forward_fixed(factory, X, y):

    pred = np.full(len(y), np.nan)

    splits = wf_splits(X)

    size = len(splits[0][0])

    for _, te in splits:

        lo = te.min()

        pred[te] = fit_predict(

            factory, X, y, np.arange(max(0, lo - size), lo), te

        )

    return pred





def oof_blocked_small(factory, X, y, seed=SEED):

    """Walk-forward test blocks, trained on a random subsample of all other days."""

    pred = np.full(len(y), np.nan)

    splits = wf_splits(X)

    size = len(splits[0][0])

    rng = np.random.default_rng(seed)

    allx = np.arange(len(y))

    for _, te in splits:

        cand = np.setdiff1d(allx, te, assume_unique=True)

        pred[te] = fit_predict(

            factory, X, y, rng.choice(cand, size=size, replace=False), te

        )

    return pred





def oof_random_small(factory, X, y, seed=SEED):

    """Random stratified 5-fold, with training sets subsampled to walk-forward size."""

    pred = np.full(len(y), np.nan)

    size = len(wf_splits(X)[0][0])

    rng = np.random.default_rng(seed)

    for tr, te in StratifiedKFold(

        N_SPLITS, shuffle=True, random_state=seed

    ).split(X, y):

        pred[te] = fit_predict(

            factory, X, y,

            rng.choice(tr, size=min(size, len(tr)), replace=False), te

        )

    return pred





def auc_or_nan(y_true, p):

    ok = ~np.isnan(p)

    return (

        roc_auc_score(y_true[ok], p[ok])

        if ok.sum() > 0 and len(np.unique(y_true[ok])) > 1

        else np.nan

    )





def evaluate_stock(name, X, y, horizon, models=None):

    """

    Evaluate the three primary procedures on a common test-date mask.

    The walk-forward primary result is purged by the forecast horizon.

    """

    gap = WINDOW + horizon

    rows = []

    first_test = wf_splits(X)[0][1].min()

    mask = np.arange(len(y)) >= first_test



    for model_name, factory in model_factories().items():

        if models and model_name not in models:

            continue

        preds = {

            "random_kfold": oof_random(factory, X, y),

            "blocked_purged": oof_blocked_purged(factory, X, y, gap),

            "walk_forward_purged": oof_walk_forward_purged(

                factory, X, y, horizon

            ),

        }

        y_eval = y.to_numpy()[mask]

        p_major = max(y_eval.mean(), 1 - y_eval.mean())



        for proc in PROCEDURES:

            p_eval = preds[proc][mask]

            ok = ~np.isnan(p_eval)

            rows.append({

                "stock": name,

                "horizon": horizon,

                "model": model_name,

                "procedure": proc,

                "accuracy": (

                    accuracy_score(y_eval[ok], p_eval[ok] >= 0.5)

                    if ok.any() else np.nan

                ),

                "auc": auc_or_nan(y_eval, p_eval),

                "majority_baseline": p_major,

                "n_eval": int(ok.sum()),

            })

    return rows





def seed_check(name, X, y, horizon, seeds):

    """Rerun random K-fold only, on the same test dates as the main analysis."""

    first_test = wf_splits(X)[0][1].min()

    mask = np.arange(len(y)) >= first_test

    y_eval = y.to_numpy()[mask]

    rows = []

    for seed in seeds:

        for model_name, factory in model_factories(seed).items():

            p_eval = oof_random(factory, X, y, seed)[mask]

            rows.append({

                "stock": name, "horizon": horizon, "model": model_name,

                "seed": seed, "auc": auc_or_nan(y_eval, p_eval)

            })

    return rows





def seed_ranking(seed_res):

    """Count changes in the better-model identity across random seeds (RQ3 noise baseline)."""

    rows = []

    for horizon, part in seed_res.groupby("horizon"):

        wide = part.pivot_table(

            index=["stock", "seed"], columns="model", values="auc"

        ).reset_index()

        wide["gboost_better"] = wide["gboost"] > wide["logreg"]

        win = wide.pivot(index="stock", columns="seed", values="gboost_better")

        base = win.columns[0]

        for seed in win.columns[1:]:

            rows.append({

                "horizon": horizon,

                "seed": seed,

                "compared_with_seed": base,

                "stocks_with_different_winner": int(

                    (win[seed] != win[base]).sum()

                ),

                "of": len(win),

            })

    return pd.DataFrame(rows)





# ----------------------------------------------------------------------------

# Optional analyses: controls, placebo, purge sweep

# ----------------------------------------------------------------------------

def foldwise_auc(y, pred, folds, mask):

    """Mean AUC of test folds, restricted to the common evaluation mask."""

    vals = []

    for te in folds:

        te = te[mask[te]]

        v = auc_or_nan(y[te], pred[te])

        if not np.isnan(v):

            vals.append(v)

    return float(np.mean(vals)) if vals else np.nan





def evaluate_controls(name, X, y, horizon):

    """Evaluate diagnostic controls, including unpurged walk-forward."""

    gap = WINDOW + horizon

    splits = wf_splits(X)

    first_test = splits[0][1].min()

    mask = np.arange(len(y)) >= first_test

    yv = y.to_numpy()



    rand_folds = [

        te for _, te in StratifiedKFold(

            N_SPLITS, shuffle=True, random_state=SEED

        ).split(X, y)

    ]

    block_folds = [te for _, te in KFold(N_SPLITS, shuffle=False).split(X)]

    wf_folds = [te for _, te in splits]

    folds = {

        "random_kfold": rand_folds,

        "random_small": rand_folds,

        "blocked_purged": block_folds,

        "walk_forward": wf_folds,

        "walk_forward_fixed": wf_folds,

        "walk_forward_purged": wf_folds,

        "blocked_small": wf_folds,

    }



    rows = []

    for model_name, factory in model_factories().items():

        preds = {

            "random_kfold": oof_random(factory, X, y),

            "blocked_purged": oof_blocked_purged(factory, X, y, gap),

            "walk_forward": oof_walk_forward(factory, X, y),

            "walk_forward_fixed": oof_walk_forward_fixed(factory, X, y),

            "walk_forward_purged": oof_walk_forward_purged(

                factory, X, y, horizon

            ),

            "blocked_small": oof_blocked_small(factory, X, y),

            "random_small": oof_random_small(factory, X, y),

        }

        for proc, p in preds.items():

            rows.append({

                "stock": name,

                "horizon": horizon,

                "model": model_name,

                "procedure": proc,

                "auc_pooled": auc_or_nan(yv[mask], p[mask]),

                "auc_foldwise_mean": foldwise_auc(

                    yv, p, folds[proc], mask

                ),

                "n_eval": int((~np.isnan(p[mask])).sum()),

            })

    return rows





def control_differences(res, n_boot):

    rng = np.random.default_rng(SEED)

    rows = []

    for (horizon, model_name), part in res.groupby(["horizon", "model"]):

        for metric in ("auc_pooled", "auc_foldwise_mean"):

            wide = part.pivot(

                index="stock", columns="procedure", values=metric

            )

            for label, a, b in CONTROL_COMPARISONS:

                d = (wide[a] - wide[b]).dropna()

                if len(d) >= 6:

                    lo, hi = bootstrap_median_ci(d, n_boot, rng)

                    p = (

                        stats.wilcoxon(d).pvalue

                        if (d != 0).any() else np.nan

                    )

                else:

                    lo, hi, p = np.nan, np.nan, np.nan

                rows.append({

                    "horizon": horizon, "model": model_name, "metric": metric,

                    "comparison": label, "n_stocks": len(d),

                    "median_diff": d.median(), "ci_low": lo, "ci_high": hi,

                    "n_positive": int((d > 0).sum()), "wilcoxon_p": p,

                })

    return pd.DataFrame(rows)





def placebo_job(name, X, y, horizon, run, stock_index, models):

    """Shift the target relative to features, then evaluate the primary procedures."""

    rng = np.random.default_rng([SEED, stock_index, horizon, run])

    shift = int(rng.integers(MIN_SHIFT, len(y) - MIN_SHIFT + 1))

    y_p = pd.Series(np.roll(y.to_numpy(), shift), index=y.index)

    rows = evaluate_stock(name, X, y_p, horizon, models=models)

    for r in rows:

        r["run"] = run

        r["shift_days"] = shift

    return rows





def placebo_summary(pl):

    """Summarise placebo AUCs and primary procedure contrasts over runs."""

    mean_run = pl.groupby(

        ["horizon", "model", "procedure", "run"]

    )["auc"].mean().reset_index()

    rows = []

    for (horizon, model_name, proc), g in mean_run.groupby(

        ["horizon", "model", "procedure"]

    ):

        v = g["auc"].to_numpy()

        rows.append({

            "horizon": horizon, "model": model_name,

            "quantity": f"mean AUC {proc}", "runs": len(v),

            "mean": v.mean(), "p2.5": np.percentile(v, 2.5),

            "p97.5": np.percentile(v, 97.5),

            "share_runs_above_0.52": float((v > 0.52).mean()),

        })



    wide = pl.pivot_table(

        index=["horizon", "model", "run", "stock"],

        columns="procedure", values="auc"

    ).reset_index()



    contrasts = (

        ("random_kfold minus walk_forward_purged",

         "random_kfold", "walk_forward_purged"),

        ("random_kfold minus blocked_purged",

         "random_kfold", "blocked_purged"),

    )

    for label, a, b in contrasts:

        wide[label] = wide[a] - wide[b]

        per_run = wide.groupby(

            ["horizon", "model", "run"]

        )[label].mean().reset_index()

        for (horizon, model_name), g in per_run.groupby(

            ["horizon", "model"]

        ):

            v = g[label].to_numpy()

            rows.append({

                "horizon": horizon, "model": model_name,

                "quantity": f"mean difference {label}", "runs": len(v),

                "mean": v.mean(), "p2.5": np.percentile(v, 2.5),

                "p97.5": np.percentile(v, 97.5),

                "share_runs_above_0.52": np.nan,

            })

    return pd.DataFrame(rows)





def purge_sweep_stock(name, X, y, horizon):

    """blocked_purged with several purge distances, on the common test days."""

    first_test = wf_splits(X)[0][1].min()

    mask = np.arange(len(y)) >= first_test

    yv = y.to_numpy()

    gaps = {

        "0": 0,

        "h": horizon,

        "WINDOW+h": WINDOW + horizon,

        "2*(WINDOW+h)": 2 * (WINDOW + horizon),

    }

    rows = []

    for model_name, factory in model_factories().items():

        p_rand = oof_random(factory, X, y)

        rows.append({

            "stock": name, "horizon": horizon, "model": model_name,

            "purge_rule": "random_kfold", "purge_days": np.nan,

            "train_rows": len(y) * (N_SPLITS - 1) / N_SPLITS,

            "auc": auc_or_nan(yv[mask], p_rand[mask]),

        })

        for rule, gap in gaps.items():

            p, size = oof_blocked_purged(

                factory, X, y, gap, return_train_sizes=True

            )

            rows.append({

                "stock": name, "horizon": horizon, "model": model_name,

                "purge_rule": rule, "purge_days": gap,

                "train_rows": size, "auc": auc_or_nan(yv[mask], p[mask]),

            })

    return rows





def purge_sweep_summary(sw, n_boot):

    rng = np.random.default_rng(SEED)

    rows = []

    for (horizon, model_name), part in sw.groupby(["horizon", "model"]):

        auc = part.pivot(index="stock", columns="purge_rule", values="auc")

        for rule in ("0", "h", "WINDOW+h", "2*(WINDOW+h)"):

            d = (auc["random_kfold"] - auc[rule]).dropna()

            lo, hi = (

                bootstrap_median_ci(d, n_boot, rng)

                if len(d) >= 6 else (np.nan, np.nan)

            )

            sel = part[part.purge_rule == rule]

            rows.append({

                "horizon": horizon, "model": model_name,

                "purge_rule": rule,

                "purge_days": sel["purge_days"].iloc[0],

                "mean_train_rows": sel["train_rows"].mean(),

                "mean_auc": auc[rule].mean(),

                "mean_auc_random_kfold": auc["random_kfold"].mean(),

                "median_random_minus_purged": d.median(),

                "ci_low": lo, "ci_high": hi,

                "stocks_positive": int((d > 0).sum()),

                "n_stocks": len(d),

            })

    return pd.DataFrame(rows)





# ----------------------------------------------------------------------------

# Statistics

# ----------------------------------------------------------------------------

def holm_adjust(p_values):

    """Holm step-down adjusted p-values."""

    p = np.asarray(p_values, dtype=float)

    order = np.argsort(p)

    m = len(p)

    adjusted = np.empty(m)

    running = 0.0

    for rank, idx in enumerate(order):

        running = max(running, (m - rank) * p[idx])

        adjusted[idx] = min(1.0, running)

    return adjusted





def bootstrap_median_ci(d, n_boot, rng, level=0.95):

    d = np.asarray(d, dtype=float)

    meds = np.median(

        d[rng.integers(0, len(d), size=(n_boot, len(d)))], axis=1

    )

    lo, hi = np.percentile(

        meds, [(1 - level) / 2 * 100, (1 + level) / 2 * 100]

    )

    return lo, hi





def paired_differences(res, n_boot):

    rng = np.random.default_rng(SEED)

    rows = []

    for (horizon, model_name), part in res.groupby(["horizon", "model"]):

        for metric in ("auc", "accuracy"):

            wide = part.pivot(

                index="stock", columns="procedure", values=metric

            )

            for label, a, b in COMPARISONS:

                d = (wide[a] - wide[b]).dropna()

                if len(d) >= 6 and (d != 0).any():

                    p = stats.wilcoxon(d).pvalue

                    lo, hi = bootstrap_median_ci(d, n_boot, rng)

                else:

                    p, lo, hi = np.nan, np.nan, np.nan

                rows.append({

                    "horizon": horizon, "model": model_name,

                    "metric": metric, "comparison": label,

                    "n_stocks": len(d), "mean_diff": d.mean(),

                    "median_diff": d.median(),

                    "median_ci_low": lo, "median_ci_high": hi,

                    "stocks_positive": int((d > 0).sum()),

                    "wilcoxon_p": p,

                })

    return pd.DataFrame(rows)





def test_hypotheses(diffs, min_effect=MIN_EFFECT):

    """

    H1 and H2. CHANGED: decision rule of the proposal (section 3.6):

    median difference of at least min_effect AND Holm-adjusted p below 0.05.

    """

    rows = []

    for name, (model_name, horizon, metric, label) in HYPOTHESES.items():

        sel = diffs[

            (diffs.model == model_name)

            & (diffs.horizon == horizon)

            & (diffs.metric == metric)

            & (diffs.comparison == label)

        ]

        if sel.empty:

            continue

        rows.append({"hypothesis": name, **sel.iloc[0].to_dict()})

    hyp = pd.DataFrame(rows)

    if hyp.empty:

        return hyp

    hyp["holm_p"] = holm_adjust(hyp["wilcoxon_p"].fillna(1.0))

    hyp["min_effect"] = min_effect

    hyp["supported"] = (

        (hyp["median_diff"] >= min_effect) & (hyp["holm_p"] < 0.05)

    )

    return hyp





def model_ranking(res):

    """RQ3: model winner by procedure and changes in winner across procedures."""

    rows = []

    for horizon, part in res.groupby("horizon"):

        wide = part.pivot_table(

            index=["stock", "procedure"], columns="model", values="auc"

        ).reset_index()

        if not {"gboost", "logreg"} <= set(wide.columns):

            continue

        wide["gboost_better"] = wide["gboost"] > wide["logreg"]

        for proc in PROCEDURES:

            w = wide[wide.procedure == proc]

            rows.append({

                "horizon": horizon,

                "item": f"gboost has higher AUC under {proc}",

                "stocks": int(w["gboost_better"].sum()),

                "of": len(w),

            })

        win = wide.pivot(

            index="stock", columns="procedure", values="gboost_better"

        )

        for label, a, b in COMPARISONS:

            rows.append({

                "horizon": horizon,

                "item": f"winner differs: {a} vs {b}",

                "stocks": int((win[a] != win[b]).sum()),

                "of": len(win),

            })

    return pd.DataFrame(rows)





# ----------------------------------------------------------------------------

# Output

# ----------------------------------------------------------------------------

def write_run_info(out, args, n_stocks, combined_hash=None):

    src = (

        f"{args.combined} (sample {args.sample}, SHA-256 {combined_hash})"

        if combined_hash else ("synthetic demo data" if args.demo else args.data_dir)

    )

    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()  # CHANGED: frozen analysis

    lines = [

        f"date of run: {date.today().isoformat()}",

        f"python: {platform.python_version()} ({platform.platform()})",

        f"scikit-learn: {sklearn.__version__}",

        f"pandas: {pd.__version__}",

        f"numpy: {np.__version__}",

        f"scipy: {scipy.__version__}",

        f"script: {Path(__file__).name} (SHA-256 {script_hash})",

        f"seed: {SEED}",

        f"primary procedures: {', '.join(PROCEDURES)}",

        "unpurged walk_forward: diagnostic control only (requires --controls)",

        f"extra seeds (random procedure only): {args.seeds if args.seeds else 'none'}",

        f"horizons: {args.horizons}",

        f"target: {args.target}",

        f"study period: {args.start} to {args.end}",

        f"decision rule H1/H2: median difference >= {args.min_effect} and Holm-adjusted p < 0.05",

        f"bootstrap resamples: {args.n_boot}",

        f"full planned analysis (--full): {args.full}",

        f"controls: {args.controls}; purge sweep: {args.purge_sweep}; placebo runs: {args.placebo}"

        + (

            f" (models {args.placebo_models}, shift {MIN_SHIFT} to n-{MIN_SHIFT} days)"

            if args.placebo else ""

        ),

        f"stocks: {n_stocks}",

        f"input: {src}",

        f"command: {' '.join(sys.argv)}",

    ]

    (out / "run_info.txt").write_text("\n".join(lines) + "\n")





def print_report(res, diffs, hyp, ranking):

    pd.set_option("display.width", 200)

    print("\n=== Mean over stocks (same test days for all procedures) ===")

    print(

        res.groupby(["horizon", "model", "procedure"])[

            ["auc", "accuracy", "majority_baseline"]

        ].mean().round(4).to_string()

    )

    print("\n=== AUC differences across stocks (positive = first procedure looks better) ===")

    show = diffs[diffs.metric == "auc"][

        ["horizon", "model", "comparison", "mean_diff", "median_diff",

         "median_ci_low", "median_ci_high", "stocks_positive", "wilcoxon_p"]

    ]

    print(show.to_string(

        index=False, float_format=lambda x: f"{x:.4g}"

    ))

    print("\n=== Hypotheses (Holm-adjusted over H1 and H2; minimum median difference applies) ===")

    if hyp.empty:

        print("not computed (needs gradient boosting at horizon 5 and at least 6 stocks)")

    else:

        print(hyp[

            ["hypothesis", "comparison", "median_diff", "median_ci_low",

             "median_ci_high", "wilcoxon_p", "holm_p", "min_effect", "supported"]

        ].to_string(index=False, float_format=lambda x: f"{x:.4g}"))

    print("\n=== Model ranking (RQ3) ===")

    print(ranking.to_string(index=False))





def main():

    ap = argparse.ArgumentParser(

        description=__doc__,

        formatter_class=argparse.RawDescriptionHelpFormatter,

    )

    ap.add_argument("--data-dir", default="data", help="folder with one CSV per stock")

    ap.add_argument("--combined", help="combined file (prices_all.csv); use together with --sample")

    ap.add_argument("--sample", help="with --combined: which sample to evaluate (main, confirmation or test)")

    ap.add_argument("--horizons", type=int, nargs="+", default=[1, 5])

    ap.add_argument("--n-boot", type=int, default=1000, help="bootstrap resamples for the median interval")

    ap.add_argument(

        "--seeds", type=int, nargs="*", default=[],

        help="extra seeds for the random-procedure noise baseline, e.g. --seeds 43 44 45 46"

    )

    ap.add_argument("--controls", action="store_true", help="run diagnostic mechanism controls")

    ap.add_argument("--placebo", type=int, default=0, metavar="N", help="number of placebo runs")

    ap.add_argument("--placebo-models", nargs="+", default=["gboost"], choices=["gboost", "logreg"])

    ap.add_argument("--purge-sweep", action="store_true", help="blocked_purged with several purge distances")

    ap.add_argument("--target", choices=["cumulative", "single_day"], default="cumulative")

    ap.add_argument("--jobs", type=int, default=1, help="parallel processes for placebo runs")

    ap.add_argument("--min-effect", type=float, default=MIN_EFFECT,

                    help="minimum median AUC difference for H1/H2 (decision rule, default 0.02)")

    ap.add_argument("--start", default=STUDY_START, help="first day of the study period (YYYY-MM-DD)")

    ap.add_argument("--end", default=STUDY_END, help="last day of the study period (YYYY-MM-DD)")

    ap.add_argument("--out", default="results")

    ap.add_argument("--demo", action="store_true", help="use synthetic series (sanity check)")

    ap.add_argument("--full", action="store_true",

                    help="planned confirmatory run: seeds 43-46, controls, purge sweep and 100 placebo runs")

    args = ap.parse_args()

    if args.full:                                  # CHANGED: one switch for the complete planned analysis

        args.seeds = args.seeds or [43, 44, 45, 46]

        args.controls = True

        args.purge_sweep = True

        args.placebo = args.placebo or 100



    if any(h < 1 for h in args.horizons):

        raise SystemExit("All horizons must be positive integers.")

    if args.placebo < 0:

        raise SystemExit("--placebo must be zero or a positive integer.")

    if args.jobs == 0:

        raise SystemExit("--jobs cannot be zero.")



    warnings.filterwarnings("ignore")

    combined_hash = None



    if args.demo:

        prices = demo_prices()

        args.start, args.end = None, None       # synthetic data have their own calendar

    elif args.combined:

        if not args.sample:

            raise SystemExit("--combined needs --sample (for example --sample main)")

        table = pd.read_csv(args.combined, parse_dates=["date"])

        required = {"date", "sample", "stock", "close"}

        missing = required - set(table.columns)

        if missing:

            raise SystemExit(f"Combined file is missing required columns: {sorted(missing)}")

        table = table[table["sample"] == args.sample]

        if table.empty:

            raise SystemExit(f"No rows for sample '{args.sample}' in {args.combined}")

        prices = {}

        for name, group in table.groupby("stock"):

            cols = ["close"] + (["volume"] if "volume" in group.columns else [])

            prices[name] = (

                group.set_index("date").sort_index()[cols]

                .dropna(axis=1, how="all")

            )

        combined_hash = hashlib.sha256(

            Path(args.combined).read_bytes()

        ).hexdigest()

    else:

        files = sorted(Path(args.data_dir).glob("*.csv"))

        if not files:

            raise SystemExit(f"No CSV files found in {args.data_dir}")

        prices = {f.stem: read_price_csv(f) for f in files}



    print(f"{len(prices)} stocks; horizons {args.horizons}; target {args.target}; "

          f"period {args.start} to {args.end}")

    out = Path(args.out)

    out.mkdir(parents=True, exist_ok=True)



    datasets = {}

    summary_rows = []

    all_rows = []

    seed_rows = []

    control_rows = []

    sweep_rows = []



    for name, df in prices.items():

        df = restrict_period(name, df, args.start, args.end)   # CHANGED

        check_prices(name, df)

        if len(df) < MIN_ROWS:

            print(f"  [{name}] only {len(df)} rows, skipped")

            continue

        check_no_lookahead(df)



        for horizon in args.horizons:

            X, y = make_dataset(df, horizon, args.target)

            if len(X) < MIN_ROWS:

                print(f"  [{name}, h={horizon}] only {len(X)} usable rows, skipped")

                continue

            datasets[(name, horizon)] = (X, y)

            summary_rows.append({

                "stock": name, "horizon": horizon, "rows": len(X),

                "start": X.index.min().date(), "end": X.index.max().date(),

                "share_up_days": round(float(y.mean()), 4),

            })

            all_rows += evaluate_stock(name, X, y, horizon)



            if args.seeds:

                seed_rows += seed_check(

                    name, X, y, horizon, [SEED] + list(args.seeds)

                )

            if args.controls:

                control_rows += evaluate_controls(name, X, y, horizon)

            if args.purge_sweep:

                sweep_rows += purge_sweep_stock(name, X, y, horizon)



        print(f"  [{name}] done")



    if not all_rows:

        raise SystemExit("No stock series met the minimum usable-row requirement.")



    res = pd.DataFrame(all_rows)

    diffs = paired_differences(res, args.n_boot)

    hyp = test_hypotheses(diffs, args.min_effect)

    ranking = model_ranking(res)

    print_report(res, diffs, hyp, ranking)



    pd.DataFrame(summary_rows).to_csv(out / "data_summary.csv", index=False)

    res.to_csv(out / "results_by_stock.csv", index=False)

    diffs.to_csv(out / "differences.csv", index=False)

    hyp.to_csv(out / "hypotheses.csv", index=False)

    ranking.to_csv(out / "model_ranking.csv", index=False)



    if seed_rows:

        seed_res = pd.DataFrame(seed_rows)

        seed_rank = seed_ranking(seed_res)

        seed_res.to_csv(out / "seed_check.csv", index=False)

        seed_rank.to_csv(out / "seed_ranking.csv", index=False)

        print("\nNoise baseline for RQ3 (random procedure only, other seeds vs seed 42):")

        print(seed_rank.to_string(index=False))



    if control_rows:

        cres = pd.DataFrame(control_rows)

        cdiff = control_differences(cres, args.n_boot)

        cres.to_csv(out / "controls_by_stock.csv", index=False)

        cdiff.to_csv(out / "controls_differences.csv", index=False)

        print("\n=== Controls: mean AUC over stocks ===")

        print(

            cres.groupby(["horizon", "model", "procedure"])[

                ["auc_pooled", "auc_foldwise_mean"]

            ].mean().round(4).to_string()

        )

        print("\n=== Controls: contrasts (pooled AUC) ===")

        print(

            cdiff[cdiff.metric == "auc_pooled"].drop(columns="metric").to_string(

                index=False, float_format=lambda x: f"{x:.4g}"

            )

        )



    if sweep_rows:

        sw = pd.DataFrame(sweep_rows)

        sws = purge_sweep_summary(sw, args.n_boot)

        sw.to_csv(out / "purge_sweep_by_stock.csv", index=False)

        sws.to_csv(out / "purge_sweep_summary.csv", index=False)

        print("\n=== Purge-distance sweep (blocked_purged) ===")

        print(sws.to_string(index=False, float_format=lambda x: f"{x:.4g}"))



    if args.placebo:

        names = sorted({k[0] for k in datasets})

        jobs = [

            (n, *datasets[(n, h)], h, run, names.index(n))

            for run in range(args.placebo)

            for n in names

            for h in args.horizons

            if (n, h) in datasets

        ]

        print(

            f"\nPlacebo: {args.placebo} runs x {len(names)} stocks x "

            f"{len(args.horizons)} horizons (models {args.placebo_models}) = "

            f"{len(jobs)} jobs, {args.jobs} process(es) ..."

        )

        chunks = Parallel(

            n_jobs=args.jobs, verbose=5 if args.jobs > 1 else 0

        )(

            delayed(placebo_job)(

                n, X, y, h, run, i, args.placebo_models

            )

            for n, X, y, h, run, i in jobs

        )

        pl = pd.DataFrame([r for chunk in chunks for r in chunk])

        if pl.empty:

            raise SystemExit("Placebo evaluation produced no results.")

        pls = placebo_summary(pl)

        pl.to_csv(out / "placebo_by_run.csv", index=False)

        pls.to_csv(out / "placebo_summary.csv", index=False)

        print("\n=== Placebo: distribution over runs of mean AUC across stocks ===")

        print(pls.to_string(index=False, float_format=lambda x: f"{x:.4f}"))



    write_run_info(out, args, res["stock"].nunique(), combined_hash)

    print(f"\nSaved results in {out}/")





if __name__ == "__main__":

    main()