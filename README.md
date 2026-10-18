# Crop Yield Outlook — *crop + location + season → likelihood of a good yield*

Predictive Analysis project. **Main focus:** crop-yield prediction. **Supporting component:** leaf-disease identification.

## What it does
The user gives three inputs:

1. **Crop** (13 crops: rice, maize, wheat, chickpea, cotton, …)
2. **Location** (state + district)
3. **Season** (Kharif / Rabi)

and gets the **probability that the yield will be Good, Average or Poor**, an expected yield in kg/ha with a range, and the main reasons.
Optional: choose a dry/wet scenario, replay a past year, or (maize) add leaf photos to adjust the outlook for disease.

## Notebooks (run in order)
| Notebook | Content |
|---|---|
| `notebooks/01_data_and_features.ipynb` | Data (ICRISAT yields, NASA POWER weather, MODIS NDVI), seasonal features, Good/Average/Poor labels, EDA |
| `notebooks/02_yield_likelihood_model.ipynb` | **Main model.** Baseline vs Logistic Regression vs Random Forest vs XGBoost, tuning, evaluation, MAE/RMSE/R², predictor demo |
| `notebooks/03_leaf_disease_detection.ipynb` | Supporting CNN (MobileNetV2 transfer learning) on PlantVillage maize leaves |
| `notebooks/04_integrated_outlook.ipynb` | How the two connect: satellite early-warning + disease-adjusted outlook |

Notebooks are generated from the percent-format scripts in `nb_src/` (`python tools/py2nb.py nb_src/<file>.py --run`).

## Try it
```bash
pip install -r requirements.txt
python server.py              # simple web page at http://localhost:8000
streamlit run app.py          # alternative Streamlit version
```
Shareable link example: `http://localhost:8000/?crop=Soybean&state=Madhya%20Pradesh&district=Dhar&scenario=dry`
or in Python:
```python
import sys; sys.path.insert(0, "src")
from outlook import YieldOutlook
yo = YieldOutlook()
print(yo.report(crop="Wheat", state="Punjab", district="Bhatinda", season="Rabi", scenario="dry"))
```

## Presentation
`presentation/Crop_Yield_Outlook.pptx` (16 slides, rebuilt from the saved results with `python tools/build_deck.py`).

## Data
| Data | Source | Notes |
|---|---|---|
| Yields 2001-2017, district level | ICRISAT / Tata-Cornell District Level Database | the target; copy in `data/raw/icrisat.csv` |
| Daily weather | NASA POWER API | rain, temperature, humidity, solar |
| NDVI, 16-day | MODIS MOD13Q1 via ORNL DAAC web service | used instead of Sentinel-2 (starts 2015, too short for training) |
| Leaf images | PlantVillage (maize classes) | 3,852 images |

Downloads are cached by `src/collect.py`; notebook 01 re-uses the cache.

**PlantVillage images are not in the repo** (60 MB). To run notebook 03, fetch the maize classes into `data/plantvillage_corn/`:
```bash
git clone --depth 1 --filter=blob:none --sparse https://github.com/spMohanty/PlantVillage-Dataset.git pv
cd pv && git sparse-checkout set --no-cone "raw/color/Corn_(maize)___*" && cd ..
mkdir -p data/plantvillage_corn && cp -r pv/raw/color/* data/plantvillage_corn/
```

## Method in brief
* One row per **crop × district × season × year**. Features: seasonal rainfall (total, by stage, dry spell), temperature, humidity, solar, NDVI (mean / peak / timing), anomalies vs the district normal, location, crop.
* **Label:** yield with the technology trend removed, split into terciles per crop (Poor / Average / Good), using training years only.
* **Split:** train 2001-2013, test 2014-2017 (chronological, no leakage).
* **Scenarios** (typical / dry / wet) use real analogue seasons of that district, not hand-set multipliers.

## Limitations
* District-level yields and a ~3 km NDVI box per district, not field level.
* Season conditions are not known in advance; "typical/dry/wet" scenarios stand in for a forecast.
* Season is tied to the crop (e.g. wheat is Rabi); only the crop-season pairs present in the data are offered.
* PlantVillage photos are lab-style; field accuracy will be lower. The disease-to-yield-loss factors in notebook 04 are assumptions for the demo.
