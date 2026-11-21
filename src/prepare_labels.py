"""
Build one tidy table from the three source files:
  - pneumonia-challenge-annotations-adjudicated-kaggle_2018.json  (RSNA adjudicated labels)
  - pneumonia-challenge-dataset-mappings_2018.json                (RSNA id <-> NIH filename)
  - Data_Entry_2017_v2020.csv                                     (NIH age / sex / view)

Outputs (in ./prepared/):
  labels.csv  - one row per image: ids, 3-class label, binary target, age, sex, view, split
  boxes.csv   - one row per Lung Opacity bounding box (for checking Grad-CAM)

Run:  python -I prepare_labels.py
Then upload the 'prepared' folder to Kaggle as a private Dataset.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ANN = os.path.join(HERE, "pneumonia-challenge-annotations-adjudicated-kaggle_2018.json")
MAP = os.path.join(HERE, "pneumonia-challenge-dataset-mappings_2018.json")
NIH = os.path.join(HERE, "Data_Entry_2017_v2020.csv")
OUT = os.path.join(HERE, "prepared")
SEED = 42

os.makedirs(OUT, exist_ok=True)

# ---------- 1. Final labels from the "Calculated" label group ----------
ann = json.load(open(ANN))
calc = next(g for g in ann["labelGroups"] if g["name"] == "Calculated")
calc_ids = {l["id"]: l["name"] for l in calc["labels"]}
print("Calculated labels:", calc_ids)

rows = []
for ds in ann["datasets"]:
    for a in ds["annotations"]:
        if a["labelId"] in calc_ids:
            d = a.get("data") or {}
            rows.append({
                "SOPInstanceUID": a["SOPInstanceUID"],
                "label_name": calc_ids[a["labelId"]],
                "x": d.get("x"), "y": d.get("y"),
                "w": d.get("width"), "h": d.get("height"),
            })
calc_df = pd.DataFrame(rows)

# One class per image (an image with any opacity box is Lung Opacity)
per_img = calc_df.groupby("SOPInstanceUID")["label_name"].agg(lambda s: sorted(set(s)))
conflicts = per_img[per_img.map(len) > 1]
print(f"Images with >1 distinct Calculated class: {len(conflicts)}")
if len(conflicts):
    print(conflicts.head())
label_df = per_img.map(lambda s: "Lung Opacity" if "Lung Opacity" in s else s[0]).rename("label_name").reset_index()

boxes = calc_df[calc_df["label_name"] == "Lung Opacity"].dropna(subset=["x"])
print(f"Bounding boxes: {len(boxes)} on {boxes.SOPInstanceUID.nunique()} images")

# ---------- 2. Mapping RSNA <-> NIH ----------
mp = pd.DataFrame(json.load(open(MAP)))[["img_id", "subset_img_id", "SOPInstanceUID"]]
mp = mp.rename(columns={"img_id": "nih_image", "subset_img_id": "rsna_id"})
assert mp.rsna_id.is_unique and mp.nih_image.is_unique and mp.SOPInstanceUID.is_unique

# ---------- 3. NIH metadata ----------
nih = pd.read_csv(NIH)[["Image Index", "Patient ID", "Patient Age", "Patient Gender", "View Position", "Follow-up #"]]
nih.columns = ["nih_image", "nih_patient_id", "age", "sex", "view", "followup"]

# ---------- 4. Join ----------
df = mp.merge(label_df, on="SOPInstanceUID", how="inner").merge(nih, on="nih_image", how="left")
missing_meta = df.age.isna().sum()
print(f"Labelled images: {len(df)}   missing NIH metadata: {missing_meta}")
if missing_meta:
    sys.exit("Some images have no NIH metadata - check the join before going further.")

df["target"] = (df.label_name == "Lung Opacity").astype(int)
df["sex_m"] = (df.sex == "M").astype(int)
df["view_ap"] = (df.view == "AP").astype(int)

# ---------- 5. Patient-level split 70/15/15, stratified on 'patient has any positive' ----------
rng = np.random.default_rng(SEED)
pat = df.groupby("nih_patient_id").target.max().reset_index()
split_of = {}
for _, grp in pat.groupby("target"):
    ids = grp.nih_patient_id.to_numpy().copy()
    rng.shuffle(ids)
    n = len(ids)
    n_tr, n_va = int(0.70 * n), int(0.15 * n)
    for i, pid in enumerate(ids):
        split_of[pid] = "train" if i < n_tr else ("val" if i < n_tr + n_va else "test")
df["split"] = df.nih_patient_id.map(split_of)

# Sanity: no patient in more than one split
assert df.groupby("nih_patient_id").split.nunique().max() == 1

# ---------- 6. Report + save ----------
print("\nClass counts:\n", df.label_name.value_counts())
print("\nSplit sizes / positive rate:\n", df.groupby("split").target.agg(["size", "mean"]).round(3))
print("\nView position vs target (row %):\n", pd.crosstab(df.label_name, df.view, normalize="index").round(2))
print("\nAge range:", df.age.min(), "-", df.age.max())

cols = ["rsna_id", "nih_image", "SOPInstanceUID", "nih_patient_id", "label_name", "target",
        "age", "sex", "sex_m", "view", "view_ap", "followup", "split"]
df[cols].to_csv(os.path.join(OUT, "labels.csv"), index=False)
boxes.merge(mp[["SOPInstanceUID", "rsna_id"]], on="SOPInstanceUID")[["rsna_id", "x", "y", "w", "h"]] \
    .to_csv(os.path.join(OUT, "boxes.csv"), index=False)
print(f"\nWrote {OUT}\\labels.csv and boxes.csv")
