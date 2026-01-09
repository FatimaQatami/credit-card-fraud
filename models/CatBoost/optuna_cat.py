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



# Load data
project_root = Path(__file__).resolve().parent.parent.parent
train_dataset = pd.read_parquet(project_root / "data" / "processed" / "train_feat_eng.parquet")

# Ensure time-aware order
train_dataset = train_dataset.sort_values(['TransactionDT']).reset_index(drop=True)

# Correct data types 
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo', 'weekday', 'uid1', 'uid2', 'uid3', 
                    'device_company', 'device_os', 'parent_domain_p', 'parent_domain_r',
                    'suffix_p', 'suffix_r', 'tld_p', 'tld_r']

train_dataset[cat_cols] = train_dataset[cat_cols].astype(str)

X = train_dataset.drop(columns=['isFraud'])
y = train_dataset['isFraud']

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

