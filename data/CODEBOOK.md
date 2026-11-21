# Codebook

All data are secondary, public and de-identified. No primary data were collected, so no consent
form or questionnaire applies.

## `prepared/labels.csv` (one row per chest radiograph, n = 29,684)

| Variable | Type | Values / scale | Source | Meaning |
|---|---|---|---|---|
| `rsna_id` | string | UUID | RSNA mapping file | RSNA image ID; equals the DICOM file name in the Kaggle competition |
| `nih_image` | string | e.g. `00012773_000.png` | RSNA mapping file | Original NIH ChestX-ray14 file name |
| `SOPInstanceUID` | string | DICOM UID | RSNA mapping/annotation files | Join key between annotations and mapping |
| `nih_patient_id` | integer | 1-30805 | NIH metadata | Patient ID; used for the patient-level split and bootstrap |
| `label_name` | category | Normal / No Lung Opacity - Not Normal / Lung Opacity | RSNA "Calculated" label group | Final adjudicated class |
| `target` | binary | 1 = Lung Opacity, 0 = otherwise | derived | Outcome variable |
| `age` | integer | years, 0-92 | NIH metadata (2020 revision) | Patient age at the study |
| `sex` | category | M / F | NIH metadata | Patient sex |
| `sex_m` | binary | 1 = male | derived | Model input |
| `view` | category | AP / PA | NIH metadata | Projection: anteroposterior or posteroanterior |
| `view_ap` | binary | 1 = AP | derived | Model input (only in the "+view" model) |
| `followup` | integer | 0-183 | NIH metadata | Follow-up number; **not used as an input** (it encodes how ill the patient is) |
| `split` | category | train / val / test | `prepare_labels.py`, seed 42 | 70/15/15 split by patient, stratified on "patient has any positive image" |

Model inputs derived inside the notebook: `age_n = clip(age, 0, 100) / 100`.

## `prepared/boxes.csv` (one row per bounding box, n = 11,390 on 7,106 images)

| Variable | Type | Meaning |
|---|---|---|
| `rsna_id` | string | Image the box belongs to |
| `x`, `y` | float | Top-left corner in pixels of the 1024 x 1024 image |
| `w`, `h` | float | Box width and height in pixels |

## External set (built inside the notebook, Section 10)

NIH ChestX-ray14 images that are **not** in the RSNA subset and come from patients who are **not** in the
RSNA training or validation splits. Labels are NIH's report-mined labels:
- Definition A: positive = finding list contains "Pneumonia"; negative = "No Finding".
- Definition B: positive = finding list contains Pneumonia, Consolidation or Infiltration; negative = "No Finding".
