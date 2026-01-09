from pathlib import Path
import pandas as pd
import numpy as np
import re
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.metrics import average_precision_score

# dataset path
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


# categorical and numerical features 
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']
train[cat_cols] = train[cat_cols].astype('category')


# Feature Engineering
# Email extracted features
train['P_emaildomain'] = train['P_emaildomain'].str.lower().astype('category')
train['R_emaildomain'] = train['R_emaildomain'].str.lower().astype('category')
train['suffix_r'] = train['R_emaildomain'].str.split(pat='.', n=1).str[1].astype('category')
train['tld_r'] = train['R_emaildomain'].str.split('.').str[-1].astype('category')

# Mobile and browser keywords 
device_mobile = ["sm-", "gt-", "ale-", "cam-", "trt-", "was-", "mya-", "rne-", "cro-", 
                 "bll-", "chc-", "pra", "android", "build/", "huawei", "honor", "hisense", 
                 "zte", "htc", "moto", "xt", "samsung", "mi ", "redmi", "pixel", "nexus", "kf", 
                 "lg", "iphone", "ios"]
device_browser = ["windows", "trident", "rv:", "macos", "mac",
                  "linux"]

# Create mobile and browser binary features 
train['DeviceInfo'] = train['DeviceInfo'].str.lower().astype('category')
train['is_mobile'] = train['DeviceInfo'].str.contains('|'.join(map(re.escape, device_mobile)),
                                                      regex=True, na=None).astype(float)
train['is_browser'] = train['DeviceInfo'].str.contains('|'.join(map(re.escape, device_browser)),
                                                       regex=True, na=None).astype(float)


# TransactionAMT split features
train['TransactionAmt_dec'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[1].astype(float)
#
#  Natural logarithm of transaction amount
train['amount_log'] = np.log1p(train['TransactionAmt'])
train['amount_log10'] = np.log10(train['TransactionAmt'] + 1)
train['amount_sqrt'] = np.sqrt(train['TransactionAmt'])

# Duration features
train['hours_duration'] = train['TransactionDT'] / (60 * 60)
train['days_duration'] = train['TransactionDT'] / (60 * 60 * 24)

# Cyclical calendar features 
# Minute (0–59)
train['minute'] = (train['TransactionDT'] // 60) % 60
# Hour (0–23)
train['hour'] = (train['TransactionDT'] // 3600) % 24
# Weekday (0–6)
train['weekday'] = (train['TransactionDT'] // 86400) %  7


# split features
col_split = ['id_34']
for col in col_split:
    parts = train[col].astype(str).str.split(r'[ /_]', n=1, expand=True)
    parts = parts.reindex(columns=[0, 1]) 
    train[f"{col}_part1"] = parts[0].astype(str).astype('category')
    train[f"{col}_part2"] = parts[1].astype(str).astype('category')


# Missing-indicators features
cols = ['id_33', 'card2', 'card3']
for col in cols:
    train[col + '_is_missing'] = train[col].isna().astype(int)


# Missing-count and missing-ratio features (per group)
group = {
    "M": [f"M{i}" for i in range(1,10)]
}
for name, col in group.items():
    col = [c for c in col if c in train.columns] 
    train[f'{name}_missing_count'] = train[col].isna().sum(axis=1)
    train[f'{name}_missing_ratio'] = train[f'{name}_missing_count'] / len(col)


# Interaction features (try more)
train['id_28_combo'] = (train['id_29'].astype(str) + '_' + train['id_28'].astype(str)).astype('category')

# UID
train['D1n'] = np.floor(train['TransactionDT'] / (24*60*60)) - train['D1']
train['UID'] = (train['card1'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['P_emaildomain'].astype(str)+'_'+train['D1n'].astype(str)).astype('category')

# ProductCD high risk features binry flag
train["ProductCD_is_W"] = (train["ProductCD"] == "W").astype(int)

# Group statistics (use cat feats with count both types and num feats with std/mean)
train['D1_UID_std']  = train.groupby('UID')['D1'].transform('std')
train['D6_UID_std']  = train.groupby('UID')['D6'].transform('std')
train['D11_UID_mean'] = train.groupby('UID')['D11'].transform('mean')
train['D11_UID_std']  = train.groupby('UID')['D11'].transform('std')
train['D12_UID_std']  = train.groupby('UID')['D12'].transform('std')
train['D14_UID_std']  = train.groupby('UID')['D14'].transform('std')
train['D15_UID_std']  = train.groupby('UID')['D15'].transform('std')

train['C1_UID_mean'] = train.groupby('UID')['C1'].transform('mean')
train['C2_UID_std'] = train.groupby('UID')['C2'].transform('std')
train['C3_UID_mean'] = train.groupby('UID')['C3'].transform('mean')
train['C3_UID_std'] = train.groupby('UID')['C3'].transform('std')
train['C6_UID_mean'] = train.groupby('UID')['C6'].transform('mean')
train['C6_UID_std'] = train.groupby('UID')['C6'].transform('std')
train['C7_UID_mean'] = train.groupby('UID')['C7'].transform('mean')
train['C13_UID_mean'] = train.groupby('UID')['C13'].transform('mean')
train['C13_UID_std'] = train.groupby('UID')['C13'].transform('std')

train['M1_UID_ct'] = train.groupby('UID')['M1'].transform('count')
train['M1_UID_ctt'] = train.groupby(['UID', 'M1'])['M1'].transform('count')
train['M2_UID_ct'] = train.groupby('UID')['M2'].transform('count')
train['M4_UID_ct'] = train.groupby(['UID', 'M4'])['M4'].transform('count')
train['M7_UID_ct'] = train.groupby(['UID', 'M7'])['M7'].transform('count')
train['M9_UID_ct'] = train.groupby(['UID', 'M9'])['M9'].transform('count')
train['P_emaildomain_UID_ct'] = train.groupby(['UID', 'P_emaildomain'])['P_emaildomain'].transform('count')

# Frequency encoding 
train['card5_frq'] = train['card5'].map(train['card5'].value_counts())
train['ProductCD_frq'] = train['ProductCD'].map(train['ProductCD'].value_counts())
train['DeviceType_frq'] = train['DeviceType'].map(train['DeviceType'].value_counts())
train['id_13_frq'] = train['id_13'].map(train['id_13'].value_counts())
train['id_15_frq'] = train['id_15'].map(train['id_15'].value_counts())
train['id_19_frq'] = train['id_19'].map(train['id_19'].value_counts())


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
# delta < 0 → dropping hurts → keep feature
# delta ≈ 0 or > 0 → dropping is safe