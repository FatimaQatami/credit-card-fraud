from pathlib import Path
import pandas as pd
import numpy as np
import re
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline_lightgbm import apply_feature_engineering_selection


# Dataset path
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

train['month'] = train['TransactionDT'] // (30 * 24 * 60 * 60)
groups = train['month']
gkf = GroupKFold(n_splits=5)

def run_cv(X, y, groups):
    fold_results = []
    for train_idx, val_idx in gkf.split(X, groups=groups):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = LGBMClassifier(
            random_state=42,
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric="average_precision",
            callbacks=[lgb.early_stopping(stopping_rounds=100)]
        )

        y_prob = model.predict_proba(X_val)[:, 1]
        fold_results.append(average_precision_score(y_val, y_prob))
    return np.mean(fold_results)

baseline_prauc = run_cv(X, y, groups)

features = [
    "card4","addr2","C3","M1","V1","V2","V4","V8","V9","V10","V11","V14","V15","V16","V17","V18",
    "V21","V22","V26","V27","V28","V31","V32","V33","V39","V41","V43","V47","V56","V63","V65",
    "V68","V84","V85","V88","V89","V92","V93","V96","V97","V98","V99","V100","V101","V102",
    "V103","V104","V105","V106","V107","V110","V111","V113","V114","V115","V117","V118",
    "V119","V121","V122","V128","V135","V136","V137","V146","V148","V153","V157","V159",
    "V164","V170","V172","V173","V174","V175","V177","V179","V180","V181","V182","V183",
    "V184","V185","V186","V191","V193","V194","V195","V196","V198","V199","V204","V207",
    "V211","V212","V213","V214","V215","V216","V222","V224","V225","V226","V228","V229",
    "V230","V231","V235","V236","V237","V238","V239","V240","V241","V244","V246","V247",
    "V249","V252","V254","V255","V259","V260","V262","V264","V265","V268","V269","V274",
    "V275","V276","V277","V278","V280","V284","V286","V287","V290","V292","V297","V298",
    "V299","V301","V302","V303","V304","V305","V322","V323","V324","V325","V328","V329",
    "V330","V331","V335","V336","V337","V338","id_10","id_12","id_16","id_18","id_28","id_32",
    "id_34","id_35","id_36","suffix_r","tld_r","is_browser","amount_log","amount_log10",
    "amount_sqrt","hours_duration","days_duration","id_34_part2","id_33_is_missing",
    "M_missing_ratio","id_28_combo","C3_UID_std","M2_UID_ct","ProductCD_frq",
    "DeviceType_frq","id_15_frq"
]

drop_results = []
for f in features:
    X_drop = X.drop(columns=[f])
    prauc_drop = run_cv(X_drop, y, groups)
    drop_results.append({
        'feature': f,
        'pr_auc': prauc_drop,
        'delta': prauc_drop - baseline_prauc
    })

drop_results = pd.DataFrame(drop_results)
drop_results.to_csv(project_root / "stats" / "GS_lgb.csv")


# Permutation: 
# delta < 0 keep feature
# delta ≈> 0 → safe to drop