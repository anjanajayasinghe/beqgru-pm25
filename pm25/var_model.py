"""Vector autoregression baseline (Section 2.4.1)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.unitroot import PhillipsPerron
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.stats.diagnostic import acorr_ljungbox, het_white
from statsmodels.tsa.api import VAR
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.vecm import coint_johansen

import config as C


def decompose(df: pd.DataFrame, period=C.STL_PERIOD):
    """Robust STL decomposition of every column: (residual, trend, seasonal) frames."""
    parts = {k: pd.DataFrame(index=df.index) for k in ("resid", "trend", "seasonal")}
    for col in df.columns:
        res = STL(df[col], period=period, robust=True).fit()
        parts["resid"][col], parts["trend"][col], parts["seasonal"][col] = res.resid, res.trend, res.seasonal
    return parts["resid"], parts["trend"], parts["seasonal"]


def unit_root_tests(df: pd.DataFrame) -> pd.DataFrame:
    """ADF (with trend), Phillips–Perron and seasonal-lag ADF p-values for every column."""
    rows = []
    for col in df.columns:
        s = df[col].dropna()
        rows.append({
            "variable": col,
            "ADF p": adfuller(s, regression="ct")[1],
            "PP p": PhillipsPerron(s).pvalue,
            "ADF p (lag 24)": adfuller(s, maxlag=24, autolag=None)[1],
            "ADF p (lag 48)": adfuller(s, maxlag=48, autolag=None)[1],
        })
    out = pd.DataFrame(rows)
    out["stationary"] = (out[["ADF p", "PP p"]] < 0.05).all(axis=1)
    return out


def johansen_with_target(df: pd.DataFrame, target=C.TARGET) -> pd.DataFrame:
    """Pairwise Johansen trace test of each predictor with PM2.5 (r = 0 at 5%)."""
    rows = []
    for col in df.columns.drop(target):
        res = coint_johansen(df[[target, col]].dropna(), det_order=0, k_ar_diff=1)
        rows.append({"variable": col, "trace": res.lr1[0], "crit_5%": res.cvt[0, 1],
                     "cointegrated": bool(res.lr1[0] > res.cvt[0, 1])})
    return pd.DataFrame(rows)


def joint_stationarity(df: pd.DataFrame, target=C.TARGET) -> pd.DataFrame:
    """Largest eigenvalue modulus of a bivariate VAR(1) of each predictor with PM2.5."""
    rows = []
    for col in df.columns.drop(target):
        fit = VAR(df[[target, col]].dropna()).fit(maxlags=1)
        ev = np.abs(np.linalg.eigvals(fit.coefs[0])).max()
        rows.append({"variable": col, "max_eigenvalue": ev, "stationary": bool(ev < 1)})
    return pd.DataFrame(rows)


def granger_predictors(df: pd.DataFrame, target=C.TARGET, maxlag=None):
    """Predictors that Granger-cause PM2.5 at 5% for at least one lag up to the AIC lag."""
    maxlag = int(maxlag or max(1, VAR(df).select_order(maxlags=C.GRANGER_MAXLAG).aic))
    keep, pvals = [], {}
    for col in df.columns.drop(target):
        try:                                   # statsmodels 0.14 (paper) accepts `verbose`; 0.15 removed it
            res = grangercausalitytests(df[[target, col]], maxlag, verbose=False)
        except TypeError:
            res = grangercausalitytests(df[[target, col]], maxlag)
        p = [res[l][0]["ssr_chi2test"][1] for l in range(1, maxlag + 1)]
        pvals[col] = min(p)
        if min(p) < 0.05:
            keep.append(col)
    return keep, pd.Series(pvals, name="min p"), maxlag


def select_lag(train: pd.DataFrame, maxlags=C.VAR_MAXLAGS) -> int:
    """Lag favoured by most of AIC, BIC, FPE and HQIC."""
    sel = VAR(train).select_order(maxlags=maxlags)
    votes = [sel.aic, sel.bic, sel.fpe, sel.hqic]
    return int(max(set(votes), key=votes.count))


def diagnostics(fit, train: pd.DataFrame) -> pd.DataFrame:
    """Ljung–Box, White and Jarque–Bera tests on each residual series."""
    resid = fit.resid
    exog = sm.add_constant(train.loc[resid.index])
    rows = []
    for col in resid.columns:
        rows.append({
            "series": col,
            "Ljung-Box p": float(acorr_ljungbox(resid[col], lags=[fit.k_ar])["lb_pvalue"].iloc[0]),
            "White p": float(het_white(resid[col], exog)[1]),
            "Jarque-Bera p": float(stats.jarque_bera(resid[col])[1]),
        })
    return pd.DataFrame(rows)


def run(df: pd.DataFrame, station: str, steps: int = 48):
    """Fit the VAR on all hours before the final `steps` hours and forecast them.

    Returns the forecast table on the original scale (trend + hour-of-day seasonal
    profile + VAR residual forecast) and a dictionary of test results.
    """
    resid, trend, seasonal = decompose(df)
    predictors, granger_p, g_lag = granger_predictors(resid.copy())
    if not predictors:
        print(f"{station}: no predictor Granger-causes PM2.5 at 5%; the VAR uses all predictors instead")
        predictors = list(resid.columns.drop(C.TARGET))
    cols = [C.TARGET] + predictors
    train = resid[cols].iloc[:-steps]
    lag = C.VAR_LAG.get(station) or select_lag(train)
    fit = VAR(train).fit(lag)

    resid_fc = fit.forecast(train.values[-fit.k_ar:], steps=steps)[:, 0]
    t = np.arange(len(df)).reshape(-1, 1)
    poly = PolynomialFeatures(C.TREND_DEGREE)
    trend_model = LinearRegression().fit(poly.fit_transform(t[:-steps]), trend[C.TARGET].iloc[:-steps])
    trend_fc = trend_model.predict(poly.transform(t[-steps:]))
    hour = np.arange(len(df)) % C.STL_PERIOD
    profile = pd.Series(seasonal[C.TARGET].iloc[:-steps].to_numpy()).groupby(hour[:-steps]).mean()
    seas_fc = profile.reindex(hour[-steps:]).to_numpy()

    frame = pd.DataFrame({"datetime": df.index[-steps:], "y_true": df[C.TARGET].iloc[-steps:].to_numpy(),
                          "y_pred": resid_fc + trend_fc + seas_fc})
    tests = {
        "unit_roots": unit_root_tests(resid),
        "johansen": johansen_with_target(resid),
        "joint_stationarity": joint_stationarity(resid),
        "granger_min_p": granger_p.reset_index().rename(columns={"index": "variable"}),
        "diagnostics": diagnostics(fit, train),
        "summary": pd.DataFrame([{"granger_lag": g_lag, "var_lag": lag, "predictors": ", ".join(predictors)}]),
    }
    return frame, tests
