from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from catboost import CatBoostClassifier, Pool
import numpy as np 


# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline import apply_feature_engineering_selection

# Load data
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")


# Sort values to keep timely order
train= train.sort_values(['TransactionDT']).reset_index(drop=True)

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
train[cat_cols] = train[cat_cols].astype(str)


# Feature pipeline 
if use_feature_pipeline:
    train = apply_feature_engineering_selection(train)


# Update categorical features 
cat_cols = train.select_dtypes(include=["object","category"]).columns.tolist()
train[cat_cols] = train[cat_cols].astype(str)  


# CatBoost built-in TreeSHAP 
X = train.drop(columns="isFraud")
y = train["isFraud"]


fold_imps = []
tscv = TimeSeriesSplit(n_splits=5)

for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_train, y_train, cat_features=cat_cols)
    val_pool = Pool(X_val, y_val, cat_features=cat_cols)

    model = CatBoostClassifier(
        random_state = 42,
        thread_count = -1,
        objective = "Logloss",
        iterations = 4000,
        random_strength = 5,
        border_count = 200,
        auto_class_weights='SqrtBalanced',
        depth = 6,
        learning_rate = 0.01, 
        min_child_samples = 30, 
        rsm = 1.0,
        l2_leaf_reg = 10.0,
        bagging_temperature = 1.0,
        subsample = 1.0,
        )

    model.fit(
        train_pool,
        eval_set=val_pool,
        use_best_model=True,
        early_stopping_rounds=100, 
        verbose=False
    )

    shap_vals = model.get_feature_importance(val_pool, type="ShapValues")
    shap_vals = shap_vals[:, :-1] 

    fold_imps.append(pd.Series(np.abs(shap_vals).mean(0), index=X_val.columns))

shap_imp = pd.concat(fold_imps, axis=1).mean(axis=1).sort_values(ascending=False)
shap_imp.to_frame("mean_abs_shap").to_csv(project_root / "cat_shap.csv",
    index_label="feature")
