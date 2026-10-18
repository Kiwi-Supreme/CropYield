"""Data collection helpers (cached to data/raw).

Used by notebooks/01_data_collection.ipynb. Every function is idempotent:
if the cached file exists it is re-used, so re-running the notebook never
re-downloads anything.
"""
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

# crop -> (ICRISAT yield column, season in which it is grown)
CROPS = {
    "Rice":                 ("RICE YIELD (Kg per ha)", "Kharif"),
    "Maize":                ("MAIZE YIELD (Kg per ha)", "Kharif"),
    "Sorghum (Kharif)":     ("KHARIF SORGHUM YIELD (Kg per ha)", "Kharif"),
    "Pearl Millet":         ("PEARL MILLET YIELD (Kg per ha)", "Kharif"),
    "Groundnut":            ("GROUNDNUT YIELD (Kg per ha)", "Kharif"),
    "Soybean":              ("SOYABEAN YIELD (Kg per ha)", "Kharif"),
    "Cotton":               ("COTTON YIELD (Kg per ha)", "Kharif"),
    "Pigeonpea":            ("PIGEONPEA YIELD (Kg per ha)", "Kharif"),
    "Wheat":                ("WHEAT YIELD (Kg per ha)", "Rabi"),
    "Chickpea":             ("CHICKPEA YIELD (Kg per ha)", "Rabi"),
    "Mustard":              ("RAPESEED AND MUSTARD YIELD (Kg per ha)", "Rabi"),
    "Sorghum (Rabi)":       ("RABI SORGHUM YIELD (Kg per ha)", "Rabi"),
    "Barley":               ("BARLEY YIELD (Kg per ha)", "Rabi"),
}

# a few ICRISAT spellings the geocoder does not know
NAME_FIX = {
    "Buland Shahar": "Bulandshahr", "Rae-Bareily": "Raebareli", "Mirzpur": "Mirzapur",
    "Amarawati": "Amravati", "Ahmednagar": "Ahmednagar", "Kheri": "Lakhimpur Kheri",
    "Allahabad": "Prayagraj", "Faizabad": "Ayodhya", "Hoshangabad": "Narmadapuram",
}
STATE_FIX = {"Orissa": "Odisha"}

# major agricultural states kept in the study (hilly / minor states dropped)
STATES = ["Andhra Pradesh", "Bihar", "Chhattisgarh", "Gujarat", "Haryana", "Karnataka",
          "Madhya Pradesh", "Maharashtra", "Orissa", "Punjab", "Rajasthan", "Tamil Nadu",
          "Telangana", "Uttar Pradesh", "West Bengal"]

# hand-checked approximate (lat, lon) for ICRISAT names the geocoder cannot resolve
COORD_FIX = {
    "Ananthapur": (14.68, 77.60), "Kadapa YSR": (14.47, 78.82), "S.P.S. Nellore": (14.44, 79.99),
    "East Godavari": (17.00, 82.20), "Shahabad (now part of Bhojpur district)": (25.56, 84.66),
    "Champaran": (26.65, 84.92), "Mungair": (25.38, 86.47), "Panchmahal": (22.78, 73.61),
    "Vadodara / Baroda": (22.31, 73.18), "Mahendragarh / Narnaul": (28.04, 76.11),
    "Gurgaon": (28.46, 77.03), "Gulbarga / Kalaburagi": (17.33, 76.83),
    "Bijapur / Vijayapura": (16.83, 75.71), "Khargone / West Nimar": (21.82, 75.61),
    "Phulbani ( Kandhamal )": (20.47, 84.23), "Bhatinda": (30.21, 74.95),
    "Roopnagar / Ropar": (30.97, 76.53), "Swami Madhopur": (26.02, 76.35),
    "Tiruchirapalli / Trichy": (10.80, 78.69), "Ramananthapuram": (9.37, 78.83),
    "Mahabubnagar": (16.74, 77.99), "Birbhum": (23.91, 87.53), "Nadia": (23.40, 88.50),
    "24 Parganas": (22.72, 88.48),
}


def select_districts(icrisat, per_state=5, min_year=2001):
    """Pick the districts with the most complete multi-crop yield records."""
    d = icrisat[(icrisat.Year >= min_year) & icrisat["State Name"].isin(STATES)]
    cols = [v[0] for v in CROPS.values()]
    d = d.assign(n=(d[cols] > 0).sum(axis=1))
    g = d.groupby(["State Name", "Dist Name"]).n.sum().reset_index()
    g = g.sort_values(["State Name", "n"], ascending=[True, False])
    g = g.groupby("State Name").head(per_state)
    g = g[g.n > 0]
    return g.drop(columns="n").reset_index(drop=True)


def geocode(districts):
    path = RAW / "district_coords.csv"
    if path.exists():
        return pd.read_csv(path)
    rows = []
    for _, r in districts.iterrows():
        name = NAME_FIX.get(r["Dist Name"], r["Dist Name"])
        state = STATE_FIX.get(r["State Name"], r["State Name"])
        lat = lon = None
        if r["Dist Name"] in COORD_FIX:
            lat, lon = COORD_FIX[r["Dist Name"]]
            rows.append(dict(state=r["State Name"], district=r["Dist Name"], lat=lat, lon=lon))
            continue
        for q in (name, name + " district"):
            res = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                               params=dict(name=q, count=20, country_code="IN", language="en"),
                               timeout=30).json().get("results", [])
            hit = [x for x in res if state.lower() in (x.get("admin1") or "").lower()]
            if hit:
                lat, lon = hit[0]["latitude"], hit[0]["longitude"]
                break
            time.sleep(0.2)
        rows.append(dict(state=r["State Name"], district=r["Dist Name"], lat=lat, lon=lon))
    out = pd.DataFrame(rows)
    out.to_csv(path, index=False)
    return out


def fetch_power_one(lat, lon, start="20010101", end="20180430"):
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    params = dict(parameters="T2M,T2M_MAX,T2M_MIN,PRECTOTCORR,RH2M,ALLSKY_SFC_SW_DWN",
                  community="AG", longitude=lon, latitude=lat, start=start, end=end, format="JSON")
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=180)
            r.raise_for_status()
            p = r.json()["properties"]["parameter"]
            df = pd.DataFrame(p)
            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            return df.replace(-999, np.nan)
        except Exception:
            time.sleep(5 * (attempt + 1))
    return None


def fetch_weather(coords):
    """Daily NASA POWER weather for every district -> data/raw/weather_daily.parquet"""
    path = RAW / "weather_daily.csv.gz"
    if path.exists():
        return pd.read_csv(path, parse_dates=["date"])

    def job(r):
        df = fetch_power_one(r.lat, r.lon)
        if df is None:
            return None
        df = df.reset_index().rename(columns={"index": "date"})
        df.insert(0, "district", r.district)
        df.insert(0, "state", r.state)
        return df

    with ThreadPoolExecutor(3) as ex:
        parts = [p for p in ex.map(job, coords.itertuples()) if p is not None]
    out = pd.concat(parts, ignore_index=True)
    out.to_csv(path, index=False, compression="gzip")
    return out


MODIS = "https://modis.ornl.gov/rst/api/v1/MOD13Q1"
NDVI_DIR = RAW / "ndvi_parts"


def fetch_ndvi_one(lat, lon, km=1):
    """NDVI (16-day composite, 250 m) averaged over a ~3x3 km box around the district centre."""
    dates = requests.get(f"{MODIS}/dates", params=dict(latitude=lat, longitude=lon), timeout=60).json()["dates"]
    dates = [d["modis_date"] for d in dates if "A2001001" <= d["modis_date"] <= "A2018100"]
    rows = []
    for i in range(0, len(dates), 10):                    # the API allows 10 composites per call
        chunk = dates[i:i + 10]
        for attempt in range(5):
            try:
                r = requests.get(f"{MODIS}/subset", params=dict(
                    latitude=lat, longitude=lon, band="250m_16_days_NDVI", startDate=chunk[0], endDate=chunk[-1],
                    kmAboveBelow=km, kmLeftRight=km), timeout=180)
                r.raise_for_status()
                for s in r.json()["subset"]:
                    v = np.array(s["data"], dtype=float)
                    v = v[v > -2000]                      # drop fill values
                    if len(v):
                        rows.append((s["calendar_date"], v.mean() * 1e-4))
                break
            except Exception:
                time.sleep(4 * (attempt + 1))
        else:
            return None                                   # a chunk kept failing: do not cache a partial series
    df = pd.DataFrame(rows, columns=["date", "ndvi"])
    df["date"] = pd.to_datetime(df["date"])
    return df


def priority_order(coords):
    """Best-covered district of every state first, then the second best, ... (so a partial download is still balanced)."""
    c = coords.copy()
    c["rank"] = c.groupby("state").cumcount()
    return c.sort_values(["rank", "state"]).reset_index(drop=True)


def _ndvi_file(state, district):
    return NDVI_DIR / f"{state}__{district}".replace(" ", "_").replace("/", "-").replace("(", "").replace(")", "")         .replace(".", "") .__add__(".csv")


def fetch_ndvi(coords, workers=5, max_districts=None):
    """Resumable: each finished district is cached on its own, so an interrupted run loses nothing."""
    NDVI_DIR.mkdir(exist_ok=True)
    todo = [r for r in priority_order(coords).itertuples() if not _ndvi_file(r.state, r.district).exists()]
    if max_districts:
        todo = todo[:max_districts]

    def job(r):
        df = fetch_ndvi_one(r.lat, r.lon)
        if df is not None:
            df.to_csv(_ndvi_file(r.state, r.district), index=False)
        return r.district, df is not None

    with ThreadPoolExecutor(workers) as ex:
        for name, ok in ex.map(job, todo):
            print("ndvi", name, "ok" if ok else "FAILED", flush=True)


def load_ndvi(coords):
    """All districts whose NDVI series has been downloaded."""
    parts = []
    for r in coords.itertuples():
        f = _ndvi_file(r.state, r.district)
        if f.exists():
            d = pd.read_csv(f, parse_dates=["date"])
            d.insert(0, "district", r.district); d.insert(0, "state", r.state)
            parts.append(d)
    return pd.concat(parts, ignore_index=True)
