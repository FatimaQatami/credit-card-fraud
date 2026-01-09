from pathlib import Path
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import average_precision_score
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization import (plot_optimization_history, plot_param_importances)



# Load dataset
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

for col in cat_cols:
    train_dataset[col] = LabelEncoder().fit_transform(train_dataset[col].astype('category'))

X = train_dataset.drop(columns=['isFraud'])
y = train_dataset['isFraud']

def objective(trial):
    params = {

        "random_state": 42,
        "objective": "binary:logistic",
        "scale_pos_weight": 0.964865 / 0.035135,    
        "n_estimators": 2000,         
        "subsample": 1.0,
        "tree_method": "hist",

        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 1, 7),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
        "reg_alpha": trial.suggest_float("reg_alpha", 2.0, 5.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 2.0, 5.0),
        "gamma": trial.suggest_float("gamma", 0.0, 2.0),

    }

    tscv = TimeSeriesSplit(n_splits=5)
    results = []


    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


        model = XGBClassifier(
            **params,
            tree_method="hist", 
            callbacks=[EarlyStopping(rounds=100)],
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
        )

        p = model.predict_proba(X_val)[:,1]
        prauc = average_precision_score(y_val, p)
        results.append(prauc)

        
        trial.report(np.mean(results), step=fold)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return float(np.mean(results))


study = optuna.create_study(
    direction="maximize",
    sampler= TPESampler(seed=42),
    pruner= MedianPruner(n_min_trials=10)
)

study.optimize(objective, n_trials=80, n_jobs=8, show_progress_bar=True)

print("Best PR-AUC:", study.best_value)
print("Best params:", study.best_params)

plot_optimization_history(study).show()
plot_param_importances(study).show()