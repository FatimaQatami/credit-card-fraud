from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import GroupKFold
from xgboost.callback import EarlyStopping


# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline_xgboost import apply_feature_engineering_selection


# Load dataset
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")

# Sort values to keep timely order
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



# XGBoost built-in TreeSHAP 
X = train.drop(columns="isFraud")
y = train["isFraud"]

train['month'] = train['TransactionDT'] // (30 * 24 * 60 * 60)
groups = train['month']

fold_imps = [] 
gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, groups=groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = XGBClassifier(
        random_state=42,
        tree_method = "hist",
        eval_metric="aucpr",
        enable_categorical = True,
        callbacks=[EarlyStopping(rounds=100, save_best=True, maximize=True)],        
        )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
        
    dval = xgb.DMatrix(X_val, enable_categorical=True)
    shap_values = model.get_booster().predict(dval, pred_contribs=True)[:, :-1]
    fold_imps.append(pd.Series(np.abs(shap_values).mean(0), index=X_val.columns))



shap_imp = pd.concat(fold_imps, axis=1).mean(axis=1).sort_values(ascending=False)
shap_imp.to_frame("mean_abs_shap").to_csv(project_root / "stats" / "xgb_shap.csv",
    index_label="feature")
