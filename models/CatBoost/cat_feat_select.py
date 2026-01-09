from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (average_precision_score, roc_auc_score)
import re

# load dataset
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")

# sort values to keep timely order
train= train.sort_values(['TransactionDT']).reset_index(drop=True)

# drop TransactionID and duplicates
train = train.drop(columns=['TransactionID'])
train = train.drop_duplicates()

# drop features with over 99% missing values
missing_values = ['id_24', 'id_25', 'id_07', 'id_08', 'id_21', 'id_26', 'id_22', 'id_23', 'id_27']
train = train.drop(columns=missing_values)

# Correct data types 
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']
train[cat_cols] = train[cat_cols].astype(str)




# update categorical features 
cat_cols = train.select_dtypes(include=["object","category"]).columns.tolist()
train[cat_cols] = train[cat_cols].astype(str)  

# model training
X = train.drop(columns=['isFraud'])
y = train['isFraud']

train['month'] = train['TransactionDT'] // (30 * 24 * 60 * 60)
groups = train['month']

def run_holdout(X, y, groups, cat_cols):
    val_month = groups.max()
    val_mask = (groups == val_month)

    X_train, X_val = X.loc[~val_mask], X.loc[val_mask]
    y_train, y_val = y.loc[~val_mask], y.loc[val_mask]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)

    model = CatBoostClassifier(
        random_state=42,
        eval_metric="PRAUC",
        )

    model.fit(
        train_pool,
        eval_set=val_pool,
        early_stopping_rounds=100, 
        )

    y_prob = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, y_prob)

baseline_prauc = run_holdout(X, y, groups, cat_cols)

# near_constant
features = [
    'V1','V14','V15','V16','V17','V18','V21','V22','V27','V28','V31','V32','V33','V39',
    'V40','V41','V42','V43','V50','V51','V52','V57','V58','V59','V60','V63','V64',
    'V65','V68','V71','V80','V81','V84','V85','V88','V89','V96','V98','V99','V100',
    'V104','V105','V106','V107','V108','V109','V110','V111','V112','V113','V114',
    'V116','V117','V118','V119','V120','V121','V122','V129','V135','V138','V141',
    'V148','V174','V180','V191','V196','V240','V241','V247','V252','V269','V284',
    'V285','V286','V287','V288','V289','V296','V297','V298','V299','V300','V301',
    'V305','V311','V319', 'C3','C4','C7','C8','C3_UID_mean', 'id_04',
    'card_missing_count','card_missing_ratio']
  

drop_results = []
for f in features:
    X_drop = X.drop(columns=[f])

    prauc_drop = run_holdout(X_drop, y, groups, cat_cols)
    drop_results.append({
        "feature": f,
        "pr_auc": prauc_drop,
        "delta": prauc_drop - baseline_prauc
    })

drop_results = pd.DataFrame(drop_results)
drop_results.to_csv(project_root / "near_const_cat.csv", index=False)


# Permutation: 
# delta < 0 → dropping hurts → keep feature
# delta ≈ 0 or > 0 → dropping is safe