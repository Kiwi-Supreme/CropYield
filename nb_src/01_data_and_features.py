# %% [markdown]
# # 01 · Data, Seasonal Features and "Good / Average / Poor" Labels
#
# ## The question this project answers
# > **Input:** a *crop*, a *location* (state + district) and a *season* (Kharif / Rabi).
# > **Output:** the **likelihood that the crop yield will be Good, Average or Poor**, plus an expected yield in kg/ha.
#
# To learn that, the model needs history: for many crop–district–season combinations, *what was the yield* and *what were the
# growing conditions* (weather + satellite vegetation) that season. This notebook builds that table.
#
# | Data | Source | Used for |
# |---|---|---|
# | District crop yields 2001-2017 (13 crops) | ICRISAT / Tata-Cornell District Level Database | Target: how good was the yield |
# | Daily weather (rain, temperature, humidity, solar) | NASA POWER API | Seasonal weather features |
# | NDVI, 16-day, 250 m | MODIS MOD13Q1 via ORNL DAAC | Seasonal vegetation features |
#
# > **Why MODIS instead of Sentinel-2?** The proposal names Sentinel-2, but Sentinel-2 only starts in 2015, so it cannot give a
# > 2001-2017 history for training. MODIS gives the same NDVI vegetation index back to 2000 and needs no account.

# %%
import sys, json, inspect, warnings, pathlib
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns

ROOT = pathlib.Path("..").resolve()
sys.path.insert(0, str(ROOT / "src"))
import collect                                  # data-download helpers (cached on disk)
PROC = ROOT / "data" / "processed"; FIG = ROOT / "figures"
PROC.mkdir(exist_ok=True); FIG.mkdir(exist_ok=True)
sns.set_theme(style="whitegrid", context="notebook")

TRAIN_END = 2013          # train on 2001-2013, test on 2014-2017 (chronological ~76/24 split, no leakage across time)

# %% [markdown]
# ## 1. Crop yields (the target)
# The ICRISAT table has one row per district-year with a column per crop. We reshape it to one row per
# *crop × district × year*, keep 13 crops that are clearly Kharif or Rabi, and drop missing / zero yields.

# %%
icrisat = pd.read_csv(ROOT / "data" / "raw" / "icrisat.csv")
print(icrisat.shape, "| years", icrisat.Year.min(), "-", icrisat.Year.max())
print(inspect.getsource(collect.select_districts))
districts = collect.select_districts(icrisat, per_state=5)
coords = collect.geocode(districts)
print(len(coords), "districts in", coords.state.nunique(), "states")
coords.head()

# %%
recs = []
for crop, (col, season) in collect.CROPS.items():
    area_col = col.replace("YIELD (Kg per ha)", "AREA (1000 ha)")
    t = icrisat[["State Name", "Dist Name", "Year", col, area_col]].rename(
        columns={"State Name": "state", "Dist Name": "district", "Year": "year", col: "yield_kg_ha", area_col: "area_kha"})
    t["crop"], t["season"] = crop, season
    recs.append(t)
yields = pd.concat(recs)
yields = yields[(yields.year >= 2001) & (yields.yield_kg_ha > 0)]
yields = yields.merge(coords, on=["state", "district"])           # keep only the selected districts
print(yields.shape)
yields.groupby(["season", "crop"]).agg(rows=("yield_kg_ha", "size"), mean_yield=("yield_kg_ha", "mean"),
                                         districts=("district", "nunique")).round(0)

# %% [markdown]
# ## 2. Daily weather from NASA POWER
# One API call per district gives daily temperature, rainfall, humidity and solar radiation for 2001-2018.
# (Already downloaded and cached, so re-running is instant.)

# %%
print(inspect.getsource(collect.fetch_power_one))
weather = collect.fetch_weather(coords)
print(weather.shape); weather.head(3)

# %%
w = weather.copy(); w["year"] = w.date.dt.year; w["month"] = w.date.dt.month
annual = w.groupby(["year"]).agg(rain=("PRECTOTCORR", "sum"), tmean=("T2M", "mean")).reset_index()
annual = annual[annual.year <= 2017]
# average across districts of the annual total rainfall (each district's own total)
rain_y = w[w.year <= 2017].groupby(["state", "district", "year"]).PRECTOTCORR.sum().groupby("year").mean()
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
rain_y.plot(ax=ax[0], marker="o"); ax[0].set_title("Mean annual rainfall, study districts (mm)")
annual.set_index("year").tmean.plot(ax=ax[1], marker="o", color="tab:red"); ax[1].set_title("Mean temperature trend (°C)")
mclim = w.groupby("month").agg(rain=("PRECTOTCORR", "mean"), tmax=("T2M_MAX", "mean"))
(mclim.rain * 30).plot.bar(ax=ax[2], color="tab:blue", alpha=.7); ax[2].set_title("Monthly rainfall climatology (mm)  → Kharif=Jun-Oct, Rabi=Nov-Mar")
plt.tight_layout(); plt.savefig(FIG / "weather_trends.png", dpi=150); plt.show()

# %% [markdown]
# ## 3. Satellite vegetation: NDVI (MODIS, 16-day composites, ~3 km box around each district centre)

# %%
print(inspect.getsource(collect.fetch_ndvi_one))
ndvi = collect.load_ndvi(coords)                       # districts whose NDVI series is downloaded (cached)
have = ndvi[["state", "district"]].drop_duplicates()
coords = coords.merge(have, on=["state", "district"])  # study districts = those with weather AND satellite data
yields = yields.merge(have, on=["state", "district"])
print(ndvi.shape, "->", len(coords), "study districts"); ndvi.head(3)

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
cov = yields.pivot_table(index="crop", columns="state", values="yield_kg_ha", aggfunc="count").fillna(0)
sns.heatmap(cov, cmap="YlGn", ax=ax[0], cbar_kws={"label": "district-years"})
ax[0].set_title("Data coverage: crop × state")
sns.scatterplot(data=coords, x="lon", y="lat", hue="state", s=70, ax=ax[1], legend=False)
for _, r in coords.iterrows(): ax[1].text(r.lon + .1, r.lat + .1, r.district.split(" /")[0][:9], fontsize=5)
ax[1].set_title(f"{len(coords)} study districts across India")
plt.tight_layout(); plt.savefig(FIG / "coverage_and_map.png", dpi=150); plt.show()

# %%
ex = ndvi[ndvi.district.isin(coords.district.iloc[[0, len(coords) // 3, 2 * len(coords) // 3]])].copy()
ex["doy"] = ex.date.dt.dayofyear; ex["label"] = ex.district
fig, ax = plt.subplots(figsize=(9, 3.8))
for (lbl), g in ex[ex.date.dt.year.between(2012, 2016)].groupby("label"):
    g = g.sort_values("date"); ax.plot(g.date, g.ndvi, label=lbl, lw=1.2)
ax.set_title("NDVI through the year (2012-16): peaks = crop growth seasons"); ax.legend(fontsize=7, ncol=3); ax.set_ylabel("NDVI")
plt.tight_layout(); plt.savefig(FIG / "ndvi_curves.png", dpi=150); plt.show()

# %% [markdown]
# ## 4. Turn daily data into **one row per district × season × year**
# Season windows (the crop calendar):
# * **Kharif** = 1 June - 31 October of year *Y*
# * **Rabi** = 1 November of *Y* - 31 March of *Y+1*  (ICRISAT labels this agricultural year *Y*)
#
# Weather features: total rain, rain in each third of the season (early / mid / late), rainy days, longest dry spell,
# mean / max / min temperature, hot days, cold days, humidity, solar radiation.
# Satellite features: mean / peak / minimum NDVI, when the peak happens, and early-season greenness.

# %%
def tag_season(df):
    m, y = df.date.dt.month, df.date.dt.year
    df = df.copy()
    df["season"] = np.select([m.between(6, 10), m >= 11, m <= 3], ["Kharif", "Rabi", "Rabi"], default=None)
    df["year"] = np.where((df.season == "Rabi") & (m <= 3), y - 1, y)
    start = pd.to_datetime(np.where(df.season == "Kharif", df.year.astype(str) + "-06-01", df.year.astype(str) + "-11-01"))
    df["day"] = (df.date - start).dt.days
    df["third"] = (df.day // 51).clip(upper=2)
    return df.dropna(subset=["season"])

def longest_dry_spell(p):
    run = best = 0
    for v in (p.values < 2.5):
        run = run + 1 if v else 0
        best = max(best, run)
    return best

def weather_features(weather):
    w = tag_season(weather)
    keys = ["state", "district", "season", "year"]
    g = w.groupby(keys)
    f = g.agg(n_days=("day", "size"),
              rain_total=("PRECTOTCORR", "sum"), rain_days=("PRECTOTCORR", lambda s: (s > 2.5).sum()),
              tmean=("T2M", "mean"), tmax=("T2M_MAX", "mean"), tmin=("T2M_MIN", "mean"),
              heat_days=("T2M_MAX", lambda s: (s > 35).sum()), cold_days=("T2M_MIN", lambda s: (s < 8).sum()),
              humidity=("RH2M", "mean"), solar=("ALLSKY_SFC_SW_DWN", "mean"))
    f["dry_spell"] = g.PRECTOTCORR.apply(longest_dry_spell)
    thirds = w.pivot_table(index=keys, columns="third", values="PRECTOTCORR", aggfunc="sum")
    thirds.columns = ["rain_early", "rain_mid", "rain_late"]
    f = f.join(thirds).reset_index()
    return f[f.n_days >= 140].drop(columns="n_days")          # drop incomplete seasons

def satellite_features(ndvi):
    s = tag_season(ndvi)
    keys = ["state", "district", "season", "year"]
    g = s.groupby(keys)
    f = g.agg(ndvi_mean=("ndvi", "mean"), ndvi_max=("ndvi", "max"), ndvi_min=("ndvi", "min"), n=("ndvi", "size"))
    f["ndvi_peak_day"] = s.loc[g.ndvi.idxmax()].set_index(keys).day
    f["ndvi_early"] = s[s.third == 0].groupby(keys).ndvi.mean()
    return f[f.n >= 6].drop(columns="n").reset_index()

wf, sf = weather_features(weather), satellite_features(ndvi)
feat = wf.merge(sf, on=["state", "district", "season", "year"], how="inner")
print("weather rows", len(wf), "| satellite rows", len(sf), "| merged", len(feat))
feat.head()

# %% [markdown]
# ## 5. Join yields + features, and define the label
#
# **"Good" must mean good for that crop**, and yields creep upward over the years from better seeds and inputs, so we:
# 1. **Remove the technology trend** per crop: fit log-yield vs year on the *training* years, then divide each yield by its trend.
# 2. Split each crop's trend-adjusted yields (training years only) into **terciles**:
#    bottom third = **Poor**, middle = **Average**, top third = **Good**.
#
# Because the terciles are computed across *all districts* for a crop, location matters (Punjab wheat is usually Good, central
# India wheat usually less so) as well as the season's weather. Thresholds are computed on training years only, so nothing from
# the test years leaks in.

# %%
data = yields.merge(feat, on=["state", "district", "season", "year"], how="inner")
train_mask = data.year <= TRAIN_END

trend = {}
for crop, g in data[train_mask].groupby("crop"):
    med = g.groupby("year").yield_kg_ha.median()
    slope = np.polyfit(med.index - 2001, np.log(med.values), 1)[0]
    trend[crop] = float(slope)
data["trend_factor"] = np.exp(data.crop.map(trend) * (data.year - 2001))
data["adj_yield"] = data.yield_kg_ha / data.trend_factor

thr = (data[train_mask].groupby("crop").adj_yield.quantile([1 / 3, 2 / 3]).unstack()
       .rename(columns={1 / 3: "poor_below", 2 / 3: "good_above"}))
data = data.join(thr, on="crop")
data["label"] = np.select([data.adj_yield <= data.poor_below, data.adj_yield >= data.good_above], [0, 2], default=1)
data["label_name"] = data.label.map({0: "Poor", 1: "Average", 2: "Good"})
print(data.shape, "rows |", data.crop.nunique(), "crops |", data.district.nunique(), "districts")
print("\nTrend growth per year (%):\n", (pd.Series(trend) * 100).round(2).to_string())
thr.round(0)

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
order = data.groupby("crop").yield_kg_ha.median().sort_values().index
sns.boxplot(data=data, y="crop", x="yield_kg_ha", order=order, ax=ax[0], color="#9ecb9e", fliersize=1)
ax[0].set_title("Yield distribution by crop (kg/ha)")
lab = data.groupby(["crop", "label_name"]).size().unstack().loc[order, ["Poor", "Average", "Good"]]
lab.plot.barh(stacked=True, ax=ax[1], color=["#d9534f", "#f0ad4e", "#5cb85c"]); ax[1].set_title("Label balance (all years)")
ex = data[data.crop == "Wheat"].groupby("year").yield_kg_ha.median()
ex.plot(ax=ax[2], marker="o", label="median wheat yield"); ax[2].set_title("Upward yield trend (why we detrend)"); ax[2].legend()
plt.tight_layout(); plt.savefig(FIG / "yield_distribution_labels.png", dpi=150); plt.show()

# %% [markdown]
# ## 6. Which conditions go with good yields? (quick EDA)
# Correlation of each seasonal feature with the **trend-adjusted yield**, separately for Kharif and Rabi.

# %%
fcols = ["rain_total", "rain_early", "rain_mid", "rain_late", "rain_days", "dry_spell", "tmean", "tmax", "tmin", "heat_days",
         "cold_days", "humidity", "solar", "ndvi_mean", "ndvi_max", "ndvi_min", "ndvi_peak_day", "ndvi_early"]
# within each crop-district, compare against that district's own average -> isolates the weather effect
d2 = data.copy()
for c in fcols + ["adj_yield"]:
    d2[c] = d2[c] - d2.groupby(["crop", "district"])[c].transform("mean")
corr = pd.DataFrame({s: d2[d2.season == s][fcols].corrwith(d2[d2.season == s].adj_yield) for s in ["Kharif", "Rabi"]})
fig, ax = plt.subplots(figsize=(5.8, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
ax.set_title("Season anomaly vs yield anomaly\n(correlation, within each crop-district)")
plt.tight_layout(); plt.savefig(FIG / "feature_correlation.png", dpi=150); plt.show()

# %% [markdown]
# The signals are **modest** (|r| ≤ 0.14) but point the way agronomy expects: cooler, wetter and more humid Kharif seasons, and
# sunnier, greener (higher NDVI) Rabi seasons, go with better yields. Within one district, weather explains only a small share of
# year-to-year differences, which is why the district's own history turns out to matter most in notebook 02.

# %% [markdown]
# ## 7. "Normal" conditions per district-season
# At prediction time a farmer asks about the *coming* season, whose weather is not known yet. So we also store each district's
# **normal** (average over the training years) conditions. These (a) tell the model the district's climate and
# (b) are the starting point for a "what if it's a dry / normal / wet season" scenario.

# %%
normals = (feat[feat.year <= TRAIN_END].groupby(["state", "district", "season"])[fcols].mean().reset_index())
print(normals.shape); normals.head()

# %%
data.to_csv(PROC / "model_table.csv", index=False)
normals.to_csv(PROC / "normals.csv", index=False)
feat.to_csv(PROC / "season_features.csv", index=False)
coords.to_csv(PROC / "coords.csv", index=False)
json.dump(dict(trend=trend, thresholds=thr.to_dict(orient="index"), train_end=TRAIN_END, fcols=fcols),
          open(PROC / "label_meta.json", "w"), indent=1)
print("saved:", [p.name for p in PROC.iterdir()])
print(f"\nmodel table: {len(data):,} rows | train {int(train_mask.sum()):,} | test {int((~train_mask).sum()):,}")
