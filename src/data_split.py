from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent

# load merged data
merged = pd.read_parquet(project_root / "data" / "processed" / "merged.parquet")

# Sanity checks
print("Merged data rows & columns count: ", merged.shape)
print("Missing values total count: ", merged.isna().sum().sum())
print("Missing values total percentage: ", merged.isna().mean().mean() * 100)
print("Duplicates: ", merged.duplicated().sum())
print("Data types: ", merged.dtypes.value_counts())

# Time-aware split 
threshold = merged['TransactionID'].quantile(0.8)

train = merged[merged['TransactionID'] <= threshold]
test = merged[merged['TransactionID'] > threshold]

train.to_parquet(project_root / "data" / "processed" / "train.parquet")
test.to_parquet(project_root / "data"/ "processed" / "test.parquet")

train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")
test = pd.read_parquet(project_root / "data" / "processed" / "test.parquet")


# Post split checks
print("Train rows and columns: ", train.shape)
print("Train duplicates", train.duplicated().sum())
print("Train missing values percentage: ", train.isna().mean().mean() *100)
print("Train fraud ration", train['isFraud'].mean())

print("Test rows and columns: ", test.shape)
print("Test duplicates: ", test.duplicated().sum())
print("Test missing values percentage: ", test.isna().mean().mean() * 100)
print("Test fraud ratio: ", test['isFraud'].mean())