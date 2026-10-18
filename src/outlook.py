"""Crop-yield outlook: crop + location + season  ->  likelihood of a Good / Average / Poor yield.

The model itself is trained in notebooks/02_yield_likelihood_model.ipynb and saved to models/yield_outlook.joblib.
This module holds the feature builder (shared by training and prediction) and the user-facing predictor.
"""
from __future__ import annotations

import json
import pathlib

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
CLASSES = ["Poor", "Average", "Good"]
# Illustrative yield-loss fraction on an affected plant (assumptions for the demo, not fitted from data)
DISEASE_SEVERITY = {"Gray leaf spot": 0.25, "Northern leaf blight": 0.20, "Common rust": 0.08}
SCENARIOS = {
    "normal": "typical season (district average)",
    "dry": "drier than usual (average of the driest third of past seasons)",
    "wet": "wetter than usual (average of the wettest third of past seasons)",
}


# --------------------------------------------------------------------------- features
def build_X(df, normals, fcols, crops, states):
    """Model inputs for rows of (crop, state, district, season, lat, lon, <fcols>)."""
    keys = ["state", "district", "season"]
    n = df[keys].merge(normals, on=keys, how="left")
    X = df[fcols].reset_index(drop=True).copy()
    for c in fcols:                                   # how unusual is this season for this district?
        X["anom_" + c] = X[c].values - n[c].values
    X["lat"], X["lon"] = df["lat"].values, df["lon"].values
    if "past_log_yield" in df:                        # district's own earlier yield level for this crop
        X["past_log_yield"] = df["past_log_yield"].values
    for c in crops:
        X["crop_" + c] = (df["crop"].values == c).astype(int)
    for s in states:
        X["state_" + s] = (df["state"].values == s).astype(int)
    return X


def probs_from_regression(mu_log, sigma, poor_below, good_above):
    """P(Poor/Average/Good) if trend-adjusted log-yield ~ Normal(mu_log, sigma)."""
    lo, hi = np.log(poor_below), np.log(good_above)
    p_poor = norm.cdf((lo - mu_log) / sigma)
    p_good = 1 - norm.cdf((hi - mu_log) / sigma)
    return np.array([p_poor, 1 - p_poor - p_good, p_good])


# --------------------------------------------------------------------------- predictor
class YieldOutlook:
    def __init__(self, bundle_path=ROOT / "models" / "yield_outlook.joblib"):
        b = joblib.load(bundle_path)
        self.__dict__.update(b)           # clf, reg, fcols, crops, states, normals, feat, coords, meta, sigma, crop_season, ...

    # -- what can the user choose?
    def options(self):
        out = self.coords.sort_values(["state", "district"])
        return dict(crops=self.crops, crop_season=self.crop_season,
                    locations={s: g.district.tolist() for s, g in out.groupby("state")},
                    seasons=["Kharif", "Rabi"], scenarios=SCENARIOS)

    def _conditions(self, state, district, season, year, scenario):
        """Seasonal weather+satellite conditions: a real past season, or a data-driven scenario."""
        f = self.feat[(self.feat.state == state) & (self.feat.district == district) & (self.feat.season == season)]
        if f.empty:
            raise ValueError(f"No climate record for {district}, {state} in {season}.")
        if year is not None and (f.year == year).any():
            return f[f.year == year].iloc[0][self.fcols].astype(float), f"actual {season} {year} season conditions"
        hist = f[f.year <= self.meta["train_end"]]
        if scenario == "normal":
            return hist[self.fcols].mean(), SCENARIOS["normal"]
        order = hist.sort_values("rain_total")
        k = max(2, len(order) // 3)
        sel = order.head(k) if scenario == "dry" else order.tail(k)
        return sel[self.fcols].mean(), SCENARIOS[scenario]

    def predict(self, crop, state, district, season, year=None, scenario="normal", yield_loss=0.0):
        """Main entry point.  yield_loss (0-1) is an optional disease-loss fraction (notebook 04)."""
        if crop not in self.crop_season:
            raise ValueError(f"Unknown crop {crop!r}. Choose from {self.crops}")
        if self.crop_season[crop] != season:
            raise ValueError(f"{crop} is a {self.crop_season[crop]} crop here, not a {season} crop.")
        loc = self.coords[(self.coords.state == state) & (self.coords.district == district)]
        if loc.empty:
            raise ValueError(f"Unknown location {district}, {state}.")
        cond, desc = self._conditions(state, district, season, year, scenario)
        h = self.hist[(self.hist.crop == crop) & (self.hist.district == district)]
        past = float(h.past_log_yield.iloc[0]) if len(h) else float(self.crop_mean_log[crop])
        row = pd.DataFrame([dict(crop=crop, state=state, district=district, season=season, past_log_yield=past,
                                 lat=loc.lat.iloc[0], lon=loc.lon.iloc[0], **cond.to_dict())])
        X = build_X(row, self.normals, self.fcols, self.crops, self.states)[self.feature_names]

        t = self.meta["thresholds"][crop]
        mu = float(self.reg.predict(X)[0])
        tf = np.exp(self.meta["trend"][crop] * (self.ref_year - 2001))        # technology level of recent years
        if yield_loss > 0:                                    # disease-adjusted: shift the whole distribution
            mu += np.log(1 - yield_loss)
            p = probs_from_regression(mu, self.sigma, t["poor_below"], t["good_above"])
            method = "regression distribution (disease-adjusted)"
        else:
            p = self.clf.predict_proba(X)[0]
            method = "classifier"
        q10, q90 = np.exp(mu + norm.ppf([.1, .9]) * self.sigma) * tf
        return dict(crop=crop, state=state, district=district, season=season, conditions=desc,
                    p_poor=float(p[0]), p_average=float(p[1]), p_good=float(p[2]),
                    verdict=CLASSES[int(np.argmax(p))], method=method,
                    expected_yield=float(np.exp(mu) * tf), yield_low=float(q10), yield_high=float(q90),
                    good_threshold=float(t["good_above"] * tf), poor_threshold=float(t["poor_below"] * tf),
                    drivers=self._drivers(X, cond, state, district, season))

    def _drivers(self, X, cond, state, district, season, top=3):
        """Plain-language reasons: which seasonal conditions differ most from normal (in std-devs)."""
        n = self.normals[(self.normals.state == state) & (self.normals.district == district) & (self.normals.season == season)]
        if n.empty:
            return []
        sd = self.feat[(self.feat.season == season) & (self.feat.state == state)][self.fcols].std()
        z = ((cond[self.fcols] - n.iloc[0][self.fcols]) / sd.replace(0, np.nan)).dropna()
        z = z.reindex(z.abs().sort_values(ascending=False).index)
        names = {"rain_total": "total rainfall", "rain_early": "early-season rain", "rain_mid": "mid-season rain",
                 "rain_late": "late-season rain", "dry_spell": "longest dry spell", "tmean": "mean temperature",
                 "tmax": "daytime temperature", "heat_days": "number of hot days", "ndvi_max": "peak greenness (NDVI)",
                 "ndvi_mean": "average greenness (NDVI)", "evi_max": "peak greenness (EVI)", "ndvi_early": "early crop greenness",
                 "humidity": "humidity", "solar": "sunshine", "rain_days": "number of rainy days", "tmin": "night temperature",
                 "cold_days": "number of cold days", "evi_mean": "average greenness (EVI)", "ndvi_peak_day": "timing of peak growth"}
        return [f"{names.get(k, k)} {'above' if v > 0 else 'below'} normal ({v:+.1f} sd)" for k, v in z.head(top).items() if abs(v) >= 0.3]

    # -- friendly text
    def report(self, **kw):
        r = self.predict(**kw)
        s = [f"{r['crop']}  |  {r['district']}, {r['state']}  |  {r['season']}   [{r['conditions']}]",
             f"  P(Good) = {r['p_good']:.0%}   P(Average) = {r['p_average']:.0%}   P(Poor) = {r['p_poor']:.0%}   ->  most likely: {r['verdict']}",
             f"  Expected yield ~ {r['expected_yield']:,.0f} kg/ha  (80% range {r['yield_low']:,.0f}-{r['yield_high']:,.0f}); "
             f"'Good' means above ~ {r['good_threshold']:,.0f} kg/ha"]
        if r["drivers"]:
            s.append("  Because: " + "; ".join(r["drivers"]))
        return "\n".join(s)
