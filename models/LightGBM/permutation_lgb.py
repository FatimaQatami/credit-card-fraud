from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import lightgbm as lgb 
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance

# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline_lightgbm import apply_feature_engineering_selection

# Load dataset
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")

# Sort values to keep timely order
train= train.sort_values(['TransactionDT']).reset_index(drop=True)

# Drop TransactionID and duplicates
train = train.drop(columns=['TransactionID'])
train = train.drop_duplicates()

# drop features with over 99% missing values
missing_values = ['id_24', 'id_25', 'id_07', 'id_08', 'id_21', 'id_26', 'id_22', 'id_23', 'id_27']
train = train.drop(columns=missing_values)


# Categorical and numerical features 
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
X = train.drop(columns=['isFraud'])
y = train['isFraud']

fold_results = []
perm_list = []
feature_names = X.columns

train['month'] = train['TransactionDT'] // (30 * 24 * 60 * 60)
groups = train['month']
gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        random_state=42
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
    )

    y_prob = model.predict_proba(X_val)[:, 1]
    prauc = average_precision_score(y_val, y_prob)
    fold_results.append({'pr_auc': prauc})

    perm = permutation_importance(
        model, X_val, y_val,
        scoring="average_precision",
        n_repeats=5,
        random_state=42,
    )
    perm_list.append(pd.DataFrame({
        "feature": feature_names,
        "importance": perm.importances_mean,
    }))

avg_prauc = np.mean([r['pr_auc'] for r in fold_results])

perm_df = (
    pd.concat(perm_list)
      .groupby("feature", as_index=False)["importance"]
      .mean()
      .sort_values("importance", ascending=False)
)

perm_df.to_csv(project_root / "permutation_lgb.csv", index=False)
