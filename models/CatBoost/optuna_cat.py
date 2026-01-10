from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import (plot_optimization_history, plot_param_importances)



# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline_catboost import apply_feature_engineering_selection


# Load data
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train_feat_eng.parquet")

# Ensure time-aware order
train = train.sort_values(['TransactionDT']).reset_index(drop=True)

# Correct data types 
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo', 'weekday', 'uid1', 'uid2', 'uid3', 
                    'device_company', 'device_os', 'parent_domain_p', 'parent_domain_r',
                    'suffix_p', 'suffix_r', 'tld_p', 'tld_r']
train[cat_cols] = train[cat_cols].astype(str)


# Feature pipeline 
if use_feature_pipeline:
    train = apply_feature_engineering_selection(train)


# Features to drop based on feature selection tests
feature_list = [
    "id_18","id_05","V259","D14","id_02","V220","V83","id_04","V134","D1","V275","V132","V53","V283",
    "id_30","V52","V127","V221","V3","V309","V311","V72","V105","M3_UID_ct","V273","V128","V266","V119",
    "V278","V300","D6_UID_std","device_os","C2_UID_std","id_missing_count","V99","V5","id_38","V246",
    "V198","V203","ProductCD_is_H","V222","V207","V315","id_17","V87","V80","V30","V7","V126","V277","V157",
    "D7_adj","V205","V228","C5_UID_std","V12","V301","D7","V296","V176","V250","V242","V224","V40","V103","V85",
    "V234","V36","V261","id_32","V320","V265","is_browser","V178","V264","id_34","V101","V120","V190","M1_UID_ct",
    "V170","V66","V124","V240","V202","id_12","V15","V84","V298","V24","V321","V162","V18","V138","V284","V46","V215",
    "V114","id_11","V57","V173","V323","V16","id_10","V65","M8","V10","V71","V192","V168","V263","V286","V160","V249",
    "V22","V161","V74","V145","V167","V262","V180","V310","V147","V253","M1_UID_ctt","V191","V158","V69","V113","V31",
    "V169","V327","V41","id_20_is_507","V333","V334","id_18_is_15","V106","V107","id_19_is_266","D11","V335","V336","V11",
    "id_20_is_325","V110","V111","V337","V112","V338","V339","id_33_is_missing","V118","V117","V42","V64","card2_is_321",
    "id_17_is_225","card5_is_102","V96","V98","ProductCD_is_C","card5_is_138","card5_is_137","id_17_is_166","card3_is_185",
    "card2_is_545","V95","V94","V1","id_14_high_risk","id_13_is_52","V68","V73","id_13_is_33","id_13_high_risk","V8","V100",
    "V88","V89","V90","id_13_is_49","V166","V330","V302","V181","V183","V299","V297","V184","V185","V14","V144","V285","V193",
    "V238","V28","V279","V197","V199","V223","V21","V27","V269","V268","V155","V255","V211","V141","V140","V227","V225","V122",
    "V243","V328","V325","V324","V322","V241","V32","V237","V121","V235","V212","V305","V172"
    ]

if use_feature_pipeline:
    train = train.drop(columns=feature_list, errors='ignore')


# Update categorical features 
cat_cols = train.select_dtypes(include=["object","category"]).columns.tolist()
train[cat_cols] = train[cat_cols].astype(str)  


# Train model
X = train.drop(columns=['isFraud'])
y = train['isFraud']

def objective(trial):
    params = {

        "random_state": 42,
        "loss_function": "Logloss",
        "iterations": 2000,         
        "random_strength": 3,
        "border_count": 254,
        "class_weights": [1, 28],    

        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "depth": trial.suggest_int("depth", 4, 8),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 50),
        "rsm": trial.suggest_float("rsm", 0.7, 1.0),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 5.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 2.0),
        "subsample": trial.suggest_float("trial.suggest_float", 0.8, 1.0),

    }

    tscv = TimeSeriesSplit(n_splits=3)
    results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        pool_train = Pool(X_train, y_train, cat_features=cat_cols)
        pool_val = Pool(X_val, y_val, cat_features=cat_cols)

        model = CatBoostClassifier(
            **params,
            early_stopping_rounds=100,
            grow_policy="Depthwise",
        )

        model.fit(
            pool_train,
            eval_set= pool_val,
        )

        p = model.predict_proba(X_val)[:,1]
        prauc = average_precision_score(y_val, p)
        results.append(prauc)

        trial.report(np.mean(results), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    return float(np.mean(results))

study = optuna.create_study(
    direction = 'maximize',
    sampler = TPESampler(seed=42),
    pruner = MedianPruner(n_min_trials=4)
)

study.optimize(objective, n_trials=40, n_jobs=8, show_progress_bar=True)

print("Best PR-AUC:", study.best_value)
print("Best params:", study.best_params)

plot_optimization_history(study).show()
plot_param_importances(study).show()

