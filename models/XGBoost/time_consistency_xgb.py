from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier


# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline_xgboost import apply_feature_engineering_selection


# Load dataset
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")

# Order by time
train = train.sort_values(['TransactionDT']).reset_index(drop=True)

# Drop TransactionID and duplicates
train = train.drop(columns=['TransactionID'])
train = train.drop_duplicates()

# Drop features with over 99% missing values
missing_values = ['id_24', 'id_25', 'id_07', 'id_08', 'id_21', 'id_26', 'id_22', 'id_23', 'id_27']
train = train.drop(columns=missing_values)

# Correct data types 
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']
train[cat_cols] = train[cat_cols].astype('category')

# Feature pipeline 
if use_feature_pipeline:
    train = apply_feature_engineering_selection(train)


# Model training 
first = train['TransactionDT'].min()
last = train['TransactionDT'].max()

df_early = train[train['TransactionDT'] <= first + 30 * 24 * 3600]
df_late = train[train['TransactionDT'] >= last - 30 * 24 * 3600]


X_early = df_early.drop(columns=['isFraud'])
y_early = df_early['isFraud']

X_late = df_late.drop(columns=['isFraud'])
y_late = df_late['isFraud']

results = []

for c in X_early.columns:

    model = XGBClassifier(
        random_state = 42,
        enable_categorical = True,
        )

    model.fit(X_early[[c]], y_early)

    pr_early = average_precision_score(y_early, model.predict_proba(X_early[[c]])[:,1])
    pr_late = average_precision_score(y_late, model.predict_proba(X_late[[c]])[:,1])
    pr_drop = pr_late - pr_early
    results.append([c, pr_drop])

df = pd.DataFrame(results, columns=["feature", "pr_drop"])
df = df[df.pr_drop < 0]
df.to_csv(project_root / "stats" / "time_consistecy_xgb.csv")