from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import average_precision_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import (plot_optimization_history, plot_param_importances)


# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline_lightgbm import apply_feature_engineering_selection


# Load dataset 
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")

# Ensuring time-aware order
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


# Features to drop based on feature selection tests
low_permutation = [
    "hour", "V74", "V311", "V332", "V124", "V296", "V139", "id_06",
    "id_13_frq", "V243", "V24", "dist2", "V244", "V7", "M7", "V75",
    "V205", "V142", "V293", "D6_UID_std", "V190", "id_17", "V5", "V171"]
low_treeshsp = [
    "V244","V39","V43","V262","V260","V32","V110","V111","V114","V85",
    "V31","V63","V84","V22","V21","V96","V15","V18","V17","V16","V33",
    "id_35","id_34_part2","card4","V4","id_12","id_32","suffix_r",
    "V228","V292","V118","V148","V153","V157","V170","V107","V105",
    "V104","V89","V97","V99","V101","V103","V240","V241","V246","V247",
    "V249","V252","V254","V255","V280","V191","V193","V194","V195",
    "V196","V198","V199","V222","V229","V230","V186"]
time_inconsistent = ['V314']
gain_split = [
    "V244","V39","V43","V260","V262","V15","V16","V17","V18","V21","V22",
    "V31","V32","V33","V63","V84","V85","V96","V110","V111","V114","id_35",
    "card4","V4","V89","V97","V99","V101","V103","V104","V105","V107",
    "V118","V148","V153","V157","V170","V186","V191","V193","V194","V195",
    "V196","V198","V199","V222","V228","V229","V230","V240","V241","V246",
    "V247","V249","V252","V254","V255","V280","V292","id_12","id_32",
    "suffix_r","id_34_part2"]
low_mi = ["V314"]
high_correlation = ['V71', 'V64', 'V63', 'V60', 'V59', 'V58', 'V43', 'V33', 'V32', 'V31', 
                    '180', 'V17', 'V16']
if use_feature_pipeline:
    train = train.drop(columns=(low_permutation + low_treeshsp + time_inconsistent + gain_split + low_mi + high_correlation),
        errors="ignore")


# Model training 
X = train.drop(columns=['isFraud'])
y = train['isFraud']
 

def objective(trial):
    params = {
        "random_state": 42,
        "objective": "binary",
        "boosting_type": "gbdt",
        "metric": "binary_logloss",
        "n_estimators": 3000, 
        "max_depth": -1,
        "feature_fraction": 0.9,
        "bagging_freq": 0,
        "bagging_fraction": 1.0,
        "lambda_l1": 0.3,
        "lambda_l2": 0.3,

        "scale_pos_weight": trial.suggest_int("scale_pos_weight", 2, 5),    
        "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.02, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 50, 300),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 100, 400),
        "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.08),
        }

    gkf = GroupKFold(n_splits=3)
    train['month'] =  train['TransactionDT'] // (30*24*60*60)
    groups = train['month']

    results = []    

    for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=groups), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=100)],
        )

        p = model.predict_proba(X_val)[:, 1]
        prauc = average_precision_score(y_val, p)
        results.append(prauc)

        trial.report(np.mean(results), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return float(np.mean(results))

study = optuna.create_study(
    direction="maximize",
    sampler=TPESampler(seed=42),
    pruner=MedianPruner(n_startup_trials=5),
)
study.optimize(objective, n_trials=50, n_jobs=8, show_progress_bar=True)

print("Best PR-AUC:", study.best_value)
print("Best params:", study.best_params)

plot_optimization_history(study).show()
plot_param_importances(study).show()
