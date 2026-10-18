"""Build presentation/Crop_Yield_Outlook.pptx from the saved results (numbers are read from data/processed, never typed in)."""
import json, pathlib
import pandas as pd
from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION, XL_LABEL_POSITION
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROC, FIG = ROOT / "data" / "processed", ROOT / "figures"
OUT = ROOT / "presentation"; OUT.mkdir(exist_ok=True)

S = json.load(open(PROC / "model_summary.json"))
I = json.load(open(PROC / "integration_stats.json"))
L = json.load(open(ROOT / "models" / "leaf_disease_metrics.json"))
R = pd.read_csv(PROC / "model_results.csv")

GREEN = RGBColor(0x1F, 0x4D, 0x2B); INK = RGBColor(0x22, 0x22, 0x22); GREY = RGBColor(0x66, 0x66, 0x66)
LIGHT = RGBColor(0xF1, 0xF4, 0xF0); MID = RGBColor(0x2F, 0x6B, 0x3A); WHITE = RGBColor(255, 255, 255)
RED, AMB, GRN = RGBColor(0xB5, 0x47, 0x3E), RGBColor(0xD9, 0xA4, 0x41), RGBColor(0x4C, 0x9A, 0x5B)
FONT = "Calibri"

prs = Presentation(); prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
BLANK = prs.slide_layouts[6]
n_slide = [0]


def tb(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, space=6):
    """paras: list of str or (str, dict(size, bold, color))"""
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True; tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05); tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, p in enumerate(paras if isinstance(paras, list) else [paras]):
        txt, o = (p, {}) if isinstance(p, str) else p
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.alignment = align; para.space_after = Pt(o.get("space", space))
        r = para.add_run(); r.text = txt
        r.font.name = FONT; r.font.size = Pt(o.get("size", size)); r.font.bold = o.get("bold", bold)
        r.font.color.rgb = o.get("color", color)
        if o.get("link"):
            r.hyperlink.address = o["link"]; r.font.underline = True
    return box


def bullets(slide, x, y, w, h, items, size=16, space=8):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = box.text_frame; tf.word_wrap = True
    tf.margin_left = tf.margin_right = Inches(0.05)
    for i, it in enumerate(items):
        para = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        para.space_after = Pt(space)
        pPr = para._p.get_or_add_pPr(); pPr.set("marL", "285750"); pPr.set("indent", "-285750")
        for tag in ("a:buNone",):
            pass
        bu = pPr.makeelement("{http://schemas.openxmlformats.org/drawingml/2006/main}buChar", {"char": "•"})
        pPr.append(bu)
        head, _, rest = it.partition("||")
        if rest:
            r = para.add_run(); r.text = head; r.font.bold = True; r.font.size = Pt(size); r.font.name = FONT; r.font.color.rgb = INK
            r = para.add_run(); r.text = rest
        else:
            r = para.add_run(); r.text = head
        r.font.size = Pt(size); r.font.name = FONT; r.font.color.rgb = INK
    return box


def new_slide(title, subtitle=None, notes=None):
    s = prs.slides.add_slide(BLANK); n_slide[0] += 1
    tb(s, 0.6, 0.38, 12.1, 0.7, title, size=30, color=GREEN, bold=True)
    if subtitle:
        tb(s, 0.6, 1.08, 12.1, 0.5, subtitle, size=17, color=GREY)
    tb(s, 0.6, 7.0, 8, 0.3, "Crop Yield Outlook  |  Predictive Analysis", size=10, color=GREY)
    tb(s, 11.9, 7.0, 0.85, 0.3, str(n_slide[0]), size=10, color=GREY, align=PP_ALIGN.RIGHT)
    if notes: s.notes_slide.notes_text_frame.text = notes
    return s


def picture(slide, path, x, y, w, h, caption=None):
    im = Image.open(path); ar = im.width / im.height
    pw, ph = (w, w / ar) if w / h < ar else (h * ar, h)
    px, py = x + (w - pw) / 2, y + (h - ph) / 2
    slide.shapes.add_picture(str(path), Inches(px), Inches(py), Inches(pw), Inches(ph))
    if caption:
        tb(slide, x, py + ph + 0.03, w, 0.3, caption, size=11, color=GREY, align=PP_ALIGN.CENTER)


def box(slide, x, y, w, h, text, fill=LIGHT, color=INK, size=15, bold=False, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, Inches(x), Inches(y), Inches(w), Inches(h))
    sh.adjustments[0] = 0.08
    sh.fill.solid(); sh.fill.fore_color.rgb = fill
    if line: sh.line.color.rgb = line; sh.line.width = Pt(1)
    else: sh.line.fill.background()
    sh.shadow.inherit = False
    tf = sh.text_frame; tf.word_wrap = True; tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    tf.margin_left = tf.margin_right = Inches(0.1)
    lines = text if isinstance(text, list) else [text]
    for i, t in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph(); p.alignment = PP_ALIGN.CENTER
        r = p.add_run(); r.text = t; r.font.name = FONT; r.font.size = Pt(size if i == 0 else size - 2)
        r.font.bold = bold if i == 0 else False; r.font.color.rgb = color
    return sh


def arrow(slide, x, y, w=0.4, h=0.3):
    a = slide.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(x), Inches(y), Inches(w), Inches(h))
    a.fill.solid(); a.fill.fore_color.rgb = GREY; a.line.fill.background(); a.shadow.inherit = False


def table(slide, x, y, w, rows, col_w, size=13, row_h=0.42, header=True):
    nr, nc = len(rows), len(rows[0])
    gt = slide.shapes.add_table(nr, nc, Inches(x), Inches(y), Inches(w), Inches(row_h * nr))
    t = gt.table
    tblPr = gt._element.graphic.graphicData.tbl.tblPr
    for a in ("bandRow", "firstRow"): tblPr.set(a, "0")
    for j, cw in enumerate(col_w): t.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        t.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            c = t.cell(i, j); c.text = ""; c.vertical_anchor = MSO_ANCHOR.MIDDLE
            c.margin_left = c.margin_right = Inches(0.08)
            c.fill.solid(); c.fill.fore_color.rgb = GREEN if (header and i == 0) else (LIGHT if i % 2 == 0 else WHITE)
            p = c.text_frame.paragraphs[0]; r = p.add_run(); r.text = str(val)
            r.font.name = FONT; r.font.size = Pt(size); r.font.bold = (header and i == 0)
            r.font.color.rgb = WHITE if (header and i == 0) else INK
            if j > 0 and isinstance(val, str) and val.replace(".", "").replace("%", "").replace(",", "").replace("-", "").isdigit():
                p.alignment = PP_ALIGN.RIGHT
    return t


pct = lambda v: f"{v*100:.1f}%"
best = S["best"]; rb = R.set_index("model")
bacc = S["baseline_accuracy"]
best_row = rb.loc[best]
beats = best_row.accuracy > bacc

# ------------------------------------------------------------------ 1 title
s = prs.slides.add_slide(BLANK); n_slide[0] += 1
tb(s, 0.9, 2.2, 11.5, 1.2, "Crop Yield Outlook", size=48, color=GREEN, bold=True)
tb(s, 0.9, 3.35, 11.5, 1.0, "Predicting how likely a good yield is, from the crop, the location and the season", size=24, color=INK)
tb(s, 0.9, 4.9, 11.5, 1.2, [("Shivani Srivastava  |  23BDS0085", dict(size=18, bold=True)),
                            ("Predictive Analysis  |  Supporting component: leaf disease identification", dict(size=16, color=GREY)),
                            ("GitHub profile: github.com/Kiwi-Supreme", dict(size=16, color=MID, link="https://github.com/Kiwi-Supreme")),
                            ("Project code: github.com/Kiwi-Supreme/CropYield", dict(size=16, color=MID, link="https://github.com/Kiwi-Supreme/CropYield"))])
s.notes_slide.notes_text_frame.text = "Introduce the problem: farmers and planners want to know, before or early in a season, whether a crop is likely to do well."

# ------------------------------------------------------------------ 2 objective
s = new_slide("What the system does", "Three inputs from the user, one answer they can act on",
              "Stress that the main deliverable is the yield outlook. Disease detection supports it.")
box(s, 0.6, 1.95, 3.6, 0.75, "1  Crop", fill=LIGHT, bold=True, size=18)
box(s, 0.6, 2.85, 3.6, 0.75, "2  Location (state, district)", fill=LIGHT, bold=True, size=18)
box(s, 0.6, 3.75, 3.6, 0.75, "3  Season (Kharif / Rabi)", fill=LIGHT, bold=True, size=18)
arrow(s, 4.4, 3.0, 0.7, 0.55)
box(s, 5.3, 1.95, 3.0, 2.55, ["Yield model", "trained on 2001-2017", "district records"], fill=GREEN, color=WHITE, size=20, bold=True)
arrow(s, 8.5, 3.0, 0.7, 0.55)
box(s, 9.4, 1.95, 3.3, 0.75, f"P(Good)  P(Average)  P(Poor)", fill=LIGHT, size=16, bold=True)
box(s, 9.4, 2.85, 3.3, 0.75, "Expected yield (kg/ha) with a range", fill=LIGHT, size=15)
box(s, 9.4, 3.75, 3.3, 0.75, "The main reasons behind it", fill=LIGHT, size=15)
bullets(s, 0.6, 4.95, 12.1, 1.9, [
    "Objective:|| detect crop disease from leaf images and predict how good the crop yield will be, using weather and satellite data.",
    "Focus:|| the yield outlook is the main result. Leaf disease identification supports it where leaf photos are available.",
    "Why it helps:|| farmers and planners can see risk early and decide on irrigation, inputs and scouting.",
], size=16)

# ------------------------------------------------------------------ 3 architecture
s = new_slide("How the pieces fit together", "A main yield pipeline and a supporting disease pipeline that feeds back into it",
              "Walk through the top row first, then the bottom row, then the link between them.")
tb(s, 0.6, 1.8, 4, 0.4, "Main: yield outlook", size=16, color=GREEN, bold=True)
xs = [0.6, 3.8, 7.0, 10.2]
labels = [("Crop + location + season", "user input"), ("Seasonal features", "weather, NDVI, climate normals"),
          ("Classifier + regressor", "Random Forest / XGBoost / Logistic"), ("Likelihood of Good, Average, Poor", "plus expected yield")]
for i, (x, (a, b)) in enumerate(zip(xs, labels)):
    box(s, x, 2.25, 2.55, 1.25, [a, b], fill=LIGHT, size=15, bold=True)
    if i < 3: arrow(s, x + 2.62, 2.7, 0.5, 0.35)
tb(s, 0.6, 4.15, 5, 0.4, "Supporting: leaf disease", size=16, color=GREEN, bold=True)
labels2 = [("Leaf photo", "taken while scouting"), ("MobileNetV2 CNN", "transfer learning"), ("Disease class", "share of infected leaves"), ("Adjusted outlook", "assumed yield loss applied")]
for i, (x, (a, b)) in enumerate(zip(xs, labels2)):
    box(s, x, 4.6, 2.55, 1.25, [a, b], fill=WHITE, line=GREY, size=15, bold=True)
    if i < 3: arrow(s, x + 2.62, 5.05, 0.5, 0.35)
tb(s, 0.6, 6.1, 12.1, 0.6, "Only maize has leaf images in PlantVillage, so the combined step is demonstrated for maize.", size=14, color=GREY)

# ------------------------------------------------------------------ 4 data
s = new_slide("Data", "Four public sources, all downloaded and cached in the project",
              "Mention that Sentinel-2 starts in 2015, so MODIS NDVI is used to get a 2001-2017 history.")
table(s, 0.6, 1.8, 12.1, [
    ["Data", "Source", "Content", "Used for"],
    ["Crop yields", "ICRISAT district database", f"13 crops, {S['n_districts']} districts, 2001-2017", "Target: how good a yield was"],
    ["Weather", "NASA POWER (daily)", "Rain, temperature, humidity, solar", "Seasonal weather features"],
    ["Vegetation", "MODIS NDVI, 16-day, 250 m", "Greenness around each district centre", "Seasonal satellite features"],
    ["Leaf images", "PlantVillage, maize classes", "3,852 photos, 4 classes", "Disease CNN"],
], [2.0, 3.3, 3.9, 2.9], size=14, row_h=0.62)
bullets(s, 0.6, 5.2, 12.1, 1.6, [
    "Sentinel-2 vs MODIS:|| the proposal names Sentinel-2, which only starts in 2015. MODIS gives the same NDVI index back to 2000, so it was used for training.",
    f"Final table:|| {S['n_rows']:,} rows, one per crop, district, season and year ({S['n_train']:,} for training, {S['n_test']:,} for testing).",
], size=15)

# ------------------------------------------------------------------ 5 coverage
s = new_slide("Study area and coverage", f"{S['n_districts']} districts in {S['n_states']} states, {S['n_crops']} crops",
              "Point out that coverage differs by crop: wheat is mostly northern, cotton western, and so on.")
picture(s, FIG / "coverage_and_map.png", 0.6, 1.75, 12.1, 4.9, "Left: number of district-years per crop and state. Right: location of the districts.")

# ------------------------------------------------------------------ 6 labels
s = new_slide("Defining a “good” yield", "Good has to mean good for that crop, and yields rise over time",
              "Explain tercile labels: each crop's bottom third is Poor, middle Average, top third Good.")
bullets(s, 0.6, 1.7, 6.0, 2.2, [
    "Remove the trend.|| better seeds and inputs lift yields every year, so each yield is divided by its crop's trend.",
    "Split into thirds.|| for each crop, the bottom third of trend-adjusted yields is Poor, the middle Average, the top third Good.",
], size=15, space=8)
bullets(s, 6.9, 1.7, 5.8, 2.2, [
    "Training years only.|| thresholds come from 2001-2013, so no test year influences the labels.",
    "Location matters.|| thirds are taken across districts, so a naturally productive district is often Good, and weather moves it up or down.",
], size=15, space=8)
picture(s, FIG / "yield_distribution_labels.png", 0.6, 3.7, 12.1, 2.8, "Yield by crop, label balance, and the upward trend that the labels correct for.")

# ------------------------------------------------------------------ 7 features
s = new_slide("What the model sees", f"{S['n_features']} inputs built for each crop, district, season and year",
              "The anomaly features tell the model how unusual this season is for this district.")
table(s, 0.6, 1.8, 12.1, [
    ["Group", "Examples", "Why"],
    ["Season weather", "Total rain, rain by season stage, dry spell, rainy days, temperature, hot days, humidity, sunshine", "Water and heat at the right stage drive yield"],
    ["Satellite (NDVI)", "Average, peak and lowest greenness, timing of the peak, early-season greenness", "Shows how well the crop actually grew"],
    ["Anomalies", "Each of the above minus the district's normal", "Is this season unusual for this place?"],
    ["Location", "Latitude, longitude, state", "Captures climate and soil differences"],
    ["Past yield level", "District's earlier trend-adjusted yield for the crop", "Productive districts stay productive"],
    ["Crop", "One-hot for 13 crops", "Crops respond differently"],
], [2.3, 6.4, 3.4], size=14, row_h=0.64)
tb(s, 0.6, 6.4, 12.1, 0.5, "Past yield uses leave-one-year-out inside training years, so a row never sees its own yield.", size=13, color=GREY)

# ------------------------------------------------------------------ 8 method
s = new_slide("Method and evaluation", "Chronological split: train on the past, test on years the model has not seen",
              "A random split would leak neighbouring years into the test set, so the split is by time.")
bullets(s, 0.6, 1.8, 6.0, 4.9, [
    f"Split.|| train 2001-2013 ({S['n_train']:,} rows), test 2014-2017 ({S['n_test']:,} rows), about an 80/20 split in time.",
    "Baseline.|| predict each district's most common past label for the crop. Any model must beat this.",
    "Models.|| Logistic Regression, Random Forest (tuned), XGBoost.",
    "Tuning.|| grid search validated on 2010-2013, fitting only on earlier years.",
    "Regression.|| a Random Forest regressor gives the expected yield.",
], size=15, space=10)
tb(s, 7.0, 1.8, 5.7, 0.4, "Metrics used", size=18, color=GREEN, bold=True)
table(s, 7.0, 2.3, 5.7, [
    ["Task", "Metrics"],
    ["Poor / Average / Good", "Accuracy, precision, recall, F1, confusion matrix, AUC"],
    ["Probability quality", "Log-loss, calibration of P(Good)"],
    ["Expected yield", "MAE, RMSE, R²"],
], [2.1, 3.6], size=14, row_h=0.8)

# ------------------------------------------------------------------ 9 results
s = new_slide("Results: classification", f"Best model: {best}, accuracy {pct(best_row.accuracy)} against a {pct(bacc)} location baseline" if beats
              else f"The best model ({best}, {pct(best_row.accuracy)}) does not clearly beat the {pct(bacc)} location baseline",
              "Read the table, then explain that AUC measures ranking ability, which is why it matters for a likelihood.")
cd = CategoryChartData()
names = list(R.model); cd.categories = names
cd.add_series("Accuracy", [round(float(v), 3) for v in R.accuracy])
cd.add_series("Macro F1", [round(float(v), 3) for v in R.f1_macro])
gf = s.shapes.add_chart(XL_CHART_TYPE.BAR_CLUSTERED, Inches(0.6), Inches(1.8), Inches(7.0), Inches(4.9), cd)
ch = gf.chart; ch.has_legend = True; ch.legend.position = XL_LEGEND_POSITION.BOTTOM; ch.legend.include_in_layout = False
ch.legend.font.size = Pt(12); ch.legend.font.name = FONT
ch.plots[0].gap_width = 60
for ser, col in zip(ch.plots[0].series, (MID, RGBColor(0xA8, 0xB8, 0xA6))):
    ser.format.fill.solid(); ser.format.fill.fore_color.rgb = col
pl = ch.plots[0]; pl.has_data_labels = True; pl.data_labels.number_format = "0.00"; pl.data_labels.number_format_is_linked = False
pl.data_labels.font.size = Pt(11); pl.data_labels.font.name = FONT; pl.data_labels.position = XL_LABEL_POSITION.OUTSIDE_END
ch.value_axis.maximum_scale = 1.0; ch.value_axis.minimum_scale = 0; ch.value_axis.has_major_gridlines = False
ch.value_axis.tick_labels.font.size = Pt(11); ch.category_axis.tick_labels.font.size = Pt(12)
ch.category_axis.reverse_order = True; ch.value_axis.visible = False
rows = [["Model", "AUC", "Log-loss"]] + [[m, f"{a:.2f}" if a == a else "-", f"{l:.2f}" if l == l else "-"] for m, a, l in zip(R.model, R.auc_ovr, R.logloss)]
table(s, 7.9, 1.9, 4.8, rows, [2.6, 1.0, 1.2], size=12, row_h=0.45)
tb(s, 7.9, 1.9 + 0.45 * len(rows) + 0.15, 4.8, 1.6,
   f"Test set: {S['n_test']:,} crop-district-seasons from 2014-2017. Random guessing would give 33%.", size=13, color=GREY)

# ------------------------------------------------------------------ 10 reliability
s = new_slide("When to trust a prediction", "Accuracy rises with the model's confidence",
              "A farmer cares how far to trust a 70 percent Good. This table answers that directly.")
conf = S["confidence"]
rows = [["Model confidence at least", "Share of cases", "Accuracy"]] + [[f"{c['min_confidence']:.0%}", f"{c['share_of_cases']:.0%}", f"{c['accuracy']:.0%}"] for c in conf]
table(s, 0.6, 1.9, 5.2, rows, [2.4, 1.5, 1.3], size=14, row_h=0.5)
bullets(s, 0.6, 1.9 + 0.5 * len(rows) + 0.3, 5.2, 2.2, [
    "Use confident predictions.|| the more certain the model, the more often it is right.",
    "Be cautious on Average.|| the middle class is the hardest to separate.",
    "History dominates.|| the district's own past yield matters most; weather and NDVI add a modest gain on top.",
], size=14)
picture(s, FIG / "feature_importance_groups.png", 6.3, 1.8, 6.4, 4.6, "Permutation importance on the test years, by group of inputs.")

# ------------------------------------------------------------------ 11 regression
reg = S["regression"]
s = new_slide("Expected yield (kg/ha)", f"Better than the district’s past average on RMSE and R² ({reg['Random Forest']['R²']:.2f} vs {reg[chr(66)+'aseline: district'+chr(39)+'s past average']['R²']:.2f}), about equal on MAE",
              "Be honest: the regressor is not far better than the district's own past average. The value is in the weather response and the range.")
rows = [["Metric", "Random Forest", "District's past average"]]
for k in ["MAE (kg/ha)", "RMSE (kg/ha)", "R²"]:
    rows.append([k, f"{reg['Random Forest'][k]:,.2f}" if k.startswith("R") else f"{reg['Random Forest'][k]:,.0f}",
                 f"{reg[chr(66)+'aseline: district' + chr(39) + 's past average'][k]:,.2f}" if k.startswith("R") else f"{reg[chr(66)+'aseline: district' + chr(39) + 's past average'][k]:,.0f}"])
table(s, 0.6, 1.9, 5.6, rows, [1.9, 1.8, 1.9], size=14, row_h=0.55)
bullets(s, 0.6, 4.4, 5.6, 2.4, [
    "Range, not just a number.|| each estimate comes with an 80% range from the model's spread.",
    "Scenarios.|| dry, normal and wet seasons use real past seasons of that district.",
], size=14)
picture(s, FIG / "yield_scatter.png", 6.4, 1.8, 6.3, 4.8)

# ------------------------------------------------------------------ 12 disease
s = new_slide("Supporting component: leaf disease", f"MobileNetV2 transfer learning reaches {L['test_accuracy']*100:.1f}% on held-out maize leaves",
              "Mention that PlantVillage photos are lab-style and near-duplicates may cross the split, so field accuracy will be lower.")
bullets(s, 0.6, 1.8, 5.5, 4.9, [
    "Data.|| 3,852 PlantVillage maize leaves: gray leaf spot, common rust, northern leaf blight, healthy.",
    "Model.|| ImageNet MobileNetV2, new classification head, then the last block fine-tuned.",
    f"Test.|| {L['n_test']} unseen photos, 80/20 stratified split.",
    "Caution.|| lab-style photos on plain backgrounds. Several photos of one leaf may sit on both sides of the split, so field accuracy will be lower.",
], size=15, space=10)
picture(s, FIG / "disease_confusion_matrix.png", 6.4, 1.8, 6.3, 4.8, "Confusion matrix on the held-out leaves.")

# ------------------------------------------------------------------ 13 integration
nd = pd.DataFrame(I["ndvi_table"])
s = new_slide("Connecting the two components", "The satellite flag is only a weak signal at district level; leaf photos add the missing detail",
              "Be clear which link is measured on data (the satellite check) and which uses assumed loss factors (the disease adjustment).")
tb(s, 0.6, 1.75, 6.0, 0.4, "Link 1: satellite warning, tested on real maize data", size=16, color=GREEN, bold=True)
rows = [["Peak NDVI", "Seasons", "Poor", "Good"]]
for _, r in nd.iterrows():
    rows.append([r["group"], f"{int(r['seasons'])}", f"{r['poor_rate']*100:.0f}%", f"{r['good_rate']*100:.0f}%"])
table(s, 0.6, 2.2, 6.0, rows, [2.9, 1.0, 1.0, 1.1], size=13, row_h=0.5)
tb(s, 0.6, 2.2 + 0.5 * len(rows) + 0.1, 6.0, 0.9, f"Correlation of NDVI anomaly with yield anomaly: r = {I['ndvi_r']:.2f}. Flagged seasons are Poor only slightly more often, so district-level NDVI is a noisy warning.", size=13, color=GREY)
tb(s, 0.6, 4.55, 6.0, 0.4, "Link 2: disease evidence adjusts the outlook", size=16, color=GREEN, bold=True)
bullets(s, 0.6, 5.0, 6.0, 1.9, [
    "Scout and classify leaves, then count the share infected.",
    "Share times an assumed loss per disease gives an expected yield loss.",
    "The loss factors are demo assumptions, not fitted from data.",
], size=14, space=5)
picture(s, FIG / "integrated_outlook.png", 6.8, 1.9, 5.9, 4.5, f"Maize in {I['district']}: outlook before and after disease evidence (assumed loss {I['loss']*100:.0f}%).")

# ------------------------------------------------------------------ 14 demo
s = new_slide("Using it", "Pick a crop, a place and a season in the web page and read the result",
              "Run the page live if possible: python server.py, then open localhost:8000.")
shot = ROOT / "presentation" / "app_screenshot.png"
if shot.exists():
    picture(s, shot, 0.6, 1.75, 6.4, 4.9)
else:
    box(s, 0.6, 1.9, 6.4, 4.5, "App screenshot goes here", fill=LIGHT)
bullets(s, 7.4, 1.9, 5.3, 4.8, [
    "Three inputs.|| crop, state and district, season (set from the crop).",
    "Optional.|| a dry or wet season, a past year to replay, or a leaf disease seen on maize.",
    "Output.|| probability bars, the most likely class, expected yield with a range, and the reasons.",
    "Run it.|| python server.py, then open localhost:8000.",
], size=15, space=10)

# ------------------------------------------------------------------ 15 limitations
s = new_slide("Limitations and next steps", None, "Say these plainly; they are part of an honest result.")
tb(s, 0.6, 1.3, 5.8, 0.4, "Limitations", size=20, color=GREEN, bold=True)
bullets(s, 0.6, 1.85, 5.8, 4.9, [
    "District-level yields,|| not field-level, and one small NDVI box per district.",
    "Season conditions are unknown in advance,|| so typical / dry / wet scenarios stand in for a forecast.",
    "Labels are relative|| to each crop's spread across districts.",
    "Disease link uses assumed losses.|| No public data ties leaf photos to yields.",
    "Lab-style leaf photos|| will score lower in a real field.",
], size=17, space=11)
tb(s, 6.9, 1.3, 5.8, 0.4, "Next steps", size=20, color=GREEN, bold=True)
bullets(s, 6.9, 1.85, 5.8, 4.9, [
    "Add a weather forecast|| so the season is not a scenario.",
    "Use Sentinel-2 or field-level NDVI|| for recent years.",
    "Collect field data|| with disease severity and yield together to fit the loss factors.",
    "Add more districts and crops|| and a mid-season update as NDVI arrives.",
], size=17, space=11)

# ------------------------------------------------------------------ 16 conclusion
s = new_slide("Summary", None, "Close with the main result and the honest caveats.")
bullets(s, 0.6, 1.4, 12.1, 5.2, [
    "The product:|| crop + location + season in, probability of a Good, Average or Poor yield out, with an expected yield and reasons.",
    f"The evidence:|| {best} reaches {pct(best_row.accuracy)} accuracy and {best_row.auc_ovr:.2f} AUC on 2014-2017, against {pct(bacc)} for a location-only baseline." if beats
    else f"The evidence:|| {best} reaches {pct(best_row.accuracy)} accuracy and {best_row.auc_ovr:.2f} AUC on 2014-2017; a location-only baseline scores {pct(bacc)}, so most of the skill still comes from knowing the place.",
    f"The support:|| a MobileNetV2 CNN identifies maize leaf diseases at {L['test_accuracy']*100:.1f}% on held-out photos, and feeds an adjusted outlook.",
    "The caution:|| results are district-level and tested on four years. Confident predictions are the ones to rely on.",
], size=20, space=18)

tb(s, 0.6, 6.3, 12.1, 0.5, [("Code and notebooks: github.com/Kiwi-Supreme/CropYield", dict(size=16, color=MID, link="https://github.com/Kiwi-Supreme/CropYield"))])
prs.save(OUT / "Crop_Yield_Outlook.pptx")
print("saved", OUT / "Crop_Yield_Outlook.pptx", "slides:", len(prs.slides))
