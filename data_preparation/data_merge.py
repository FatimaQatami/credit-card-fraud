from pathlib import Path
import pandas as pd

project_root = Path(__file__).resolve().parent.parent

# Load raw data
transaction = pd.read_csv(project_root / "data" / "raw" / "train_transaction.csv")
identity = pd.read_csv(project_root / "data" / "raw" / "train_identity.csv")

# Sanity check
files = [("Transaction File",transaction), ("Identity File", identity)]
for name, df in files:
    print(f"\n---{name}---")
    print("Rows and columns: ", df.shape)
    print("Missing values in total: ", df.isna().sum().sum())
    print("Missing values total %: ", df.isna().mean().mean() * 100 )
    print("Duplicates: ", df.duplicated().sum())
    print("Has TransactionID?", "TransactionID" in df.columns)

# Premerge check
for name, df in files:
    print(f"\n TransactionID column in {name}")
    print("Missing values: ", df["TransactionID"].isna().sum())
    print("Data types: ", df["TransactionID"].dtype)
    print("IDs unique? ", df["TransactionID"].is_unique)

# Verify that every TransactionID in identity file exists in transaction file
print("\nDo all IDs overlap in both files? ", identity["TransactionID"].isin(transaction["TransactionID"]).all())

# Left join TransactionID (transaction file is the base)
merged_files = transaction.merge(identity, on="TransactionID", how="left")
merged_files.to_parquet(project_root / "data" / "processed" / "merged.parquet")