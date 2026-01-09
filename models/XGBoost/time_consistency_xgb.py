from pathlib import Path
import pandas as pd
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier
import re
import numpy as np


# load dataset
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")

# order by time
train = train.sort_values(['TransactionDT']).reset_index(drop=True)

# drop TransactionID and duplicates
train = train.drop(columns=['TransactionID'])
train = train.drop_duplicates()

# drop features with over 99% missing values
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


# Feature Engineering
# Missing-count and missing-ratio features (per row)
train['missing_count'] = train.isna().sum(axis=1)

# Missing-count and missing-ratio features (per group)
group = {
    "card": ['card1', 'card2', 'card3', 'card4', 'card5', 'card6'],
    "M": [f"M{i}" for i in range(1,10)],
    "V": [f"V{i}" for i in range(1,340)],
    "id": [f"id_{str(i).zfill(2)}" for i in range(1,39)]
}

for name, col in group.items():
    col = [c for c in col if c in train.columns] 
    train[f'{name}_missing_count'] = train[col].isna().sum(axis=1)
    train[f'{name}_missing_ratio'] = train[f'{name}_missing_count'] / len(col)

# Missing-indicators features
cols = ['dist1', 'dist2', 'addr1', 'addr2']

for col in cols:
    train[col + '_is_missing'] = train[col].isna().astype(int)


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
train['is_mobile'] = train['DeviceInfo'].str.contains('|'.join(map(re.escape, device_mobile)),
                                                      regex=True, na=None).astype(float)
train['is_browser'] = train['DeviceInfo'].str.contains('|'.join(map(re.escape, device_browser)),
                                                       regex=True, na=None).astype(float)


# TransactionAMT split features
train['TransactionAmt_dec'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[1].astype(float)


# Natural logarithm of transaction amount
train['amount_log'] = np.log1p(train['TransactionAmt'])
train['amount_log10'] = np.log10(train['TransactionAmt'] + 1)
train['amount_sqrt'] = np.sqrt(train['TransactionAmt'])

# Duration features
train['hours_duration'] = train['TransactionDT'] / (60 * 60)
train['days_duration'] = train['TransactionDT'] / (60 * 60 * 24)


# split features
col_split = ['id_34']
for col in col_split:
    parts = train[col].astype(str).str.split(r'[ /_]', n=1, expand=True)
    parts = parts.reindex(columns=[0, 1]) 
    train[f"{col}_part1"] = parts[0].astype(str).astype('category')
    train[f"{col}_part2"] = parts[1].astype(str).astype('category')


# ProductCD high risk features binry flag
train["ProductCD_is_W"] = (train["ProductCD"] == "W").astype(int)


# UID
train['D1n'] = np.floor(train['TransactionDT'] / (24*60*60)) - train['D1']
train['UID'] = (train['card1'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['P_emaildomain'].astype(str)+'_'+train['D1n'].astype(str)).astype('category')


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




# model training 
first = train['TransactionDT'].min()
last = train['TransactionDT'].max()

df_early = train[train['TransactionDT'] <= first + 30 * 24 * 3600]
df_late = train[train['TransactionDT'] >= last - 30 * 24 * 3600]


X_early = df_early.drop(columns=['isFraud'])
y_early = df_early['isFraud']

X_late = df_late.drop(columns=['isFraud'])
y_late = df_late['isFraud']

results = []

for c in X_early.columns:

    model = XGBClassifier(
        random_state = 42,
        enable_categorical = True,
        )

    model.fit(X_early[[c]], y_early)

    pr_early = average_precision_score(y_early, model.predict_proba(X_early[[c]])[:,1])
    pr_late = average_precision_score(y_late, model.predict_proba(X_late[[c]])[:,1])
    pr_drop = pr_late - pr_early
    results.append([c, pr_drop])

df = pd.DataFrame(results, columns=["feature", "pr_drop"])
df = df[df.pr_drop < 0]
df.to_csv(project_root / "stats" / "time_consistecy_xgb.csv")