# Data

No raw data is stored in this repository.

## Sources

- **RSNA Pneumonia Detection Challenge:** download from Kaggle and accept the competition rules.
- **NIH ChestX-ray14:** download from the NIH Clinical Center release.

## Preparation

1. Place the downloaded data in a local folder (for example `data/raw/`, which is git-ignored if you add it).
2. If you use `src/prepare_labels.py`, run it to generate `labels.csv` and `boxes.csv`:
   ```bash
   python src/prepare_labels.py
   ```
   These files are derived from RSNA annotations, so they are not committed. Regenerate them locally.

See [`CODEBOOK.md`](CODEBOOK.md) for column definitions.
