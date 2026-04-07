# EDA

Exploratory data analysis on the claim dataset (2880 rows, 35 columns)

## Files

- `eda_feature_engineering.ipynb` - the main notebook, does everything: loads data, checks distributions, handles missing values, engineers features, saves cleaned CSVs

## How to run

```
conda activate bolt
jupyter notebook eda_feature_engineering.ipynb
```

Make sure you select the correct kernel.

Outputs go to `../../data/` (claims_cleaned.csv, claims_features.csv, feature_columns.txt)
