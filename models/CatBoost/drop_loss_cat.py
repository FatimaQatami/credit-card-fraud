from pathlib import Path
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score


# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline import apply_feature_engineering_selection

# Load dataset
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


# Model training
X = train.drop(columns=['isFraud'])
y = train['isFraud']

fold_results = []
perm_list = []
feature_names = X.columns

train['month'] = train['TransactionDT'] // (30 * 24 * 60 * 60)
groups = train['month']

val_month = groups.max()      
val_mask = (groups == val_month)

X_train, y_train = X.loc[~val_mask], y.loc[~val_mask]
X_val,   y_val   = X.loc[val_mask],  y.loc[val_mask]

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
prauc = average_precision_score(y_val, y_prob)
fold_results.append({'pr_auc': prauc})


perm =  model.get_feature_importance(
    data=val_pool,
    type="LossFunctionChange"
)
perm_list.append(pd.DataFrame({
    "feature": feature_names,
    "importance": perm,
}))

perm_df = (
    pd.concat(perm_list)
      .groupby("feature", as_index=False)["importance"]
      .mean()
      .sort_values("importance", ascending=False)
)

perm_df.to_csv(project_root / "drop_loss_cat.csv", index=False)
