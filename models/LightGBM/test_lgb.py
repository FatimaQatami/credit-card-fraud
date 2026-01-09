from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, matthews_corrcoef)
from imblearn.metrics import geometric_mean_score
import re
import json
import joblib


# load dataset
project_root = Path(__file__).resolve().parent.parent.parent
test = pd.read_parquet(project_root / "data" / "processed" / "test.parquet")
feature_maps = joblib.load(project_root / "feature_maps.joblib")

# sort values to keep timely order
test= test.sort_values(['TransactionDT']).reset_index(drop=True)

# drop TransactionID
test = test.drop(columns=['TransactionID'])

# drop features with over 99% missing values
missing_values = ['id_24', 'id_25', 'id_07', 'id_08', 'id_21', 'id_26', 'id_22', 'id_23', 'id_27']
test = test.drop(columns=missing_values)

# Correct data types 
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']
test[cat_cols] = test[cat_cols].astype(str)

# Email extracted features
test['P_emaildomain'] = test['P_emaildomain'].str.lower()
test['R_emaildomain'] = test['R_emaildomain'].str.lower()
test['suffix_r'] = test['R_emaildomain'].str.split(pat='.', n=1).str[1]
test['tld_r'] = test['R_emaildomain'].str.split('.').str[-1]


# Mobile and browser keywords 
device_mobile = ["sm-", "gt-", "ale-", "cam-", "trt-", "was-", "mya-", "rne-", "cro-", 
                 "bll-", "chc-", "pra", "android", "build/", "huawei", "honor", "hisense", 
                 "zte", "htc", "moto", "xt", "samsung", "mi ", "redmi", "pixel", "nexus", "kf", 
                 "lg", "iphone", "ios"]
device_browser = ["windows", "trident", "rv:", "macos", "mac",
                  "linux"]
# Create mobile and browser binary features 
test['DeviceInfo'] = test['DeviceInfo'].str.lower().astype(str)
test['is_mobile'] = test['DeviceInfo'].str.contains('|'.join(map(re.escape, device_mobile)),
                                                      regex=True, na=None).astype(float)
test['is_browser'] = test['DeviceInfo'].str.contains('|'.join(map(re.escape, device_browser)),
                                                       regex=True, na=None).astype(float)

# TransactionAMT split features
train['TransactionAmt_dec'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[1].astype(float)

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
    test[col + '_is_missing'] = test[col].isna().astype(int)

    
# Missing-count and missing-ratio features (per group)
group = {
    "M": [f"M{i}" for i in range(1,10)],
}

for name, col in group.items():
    col = [c for c in col if c in test.columns] 
    test[f'{name}_missing_count'] = test[col].isna().sum(axis=1)
    test[f'{name}_missing_ratio'] = test[f'{name}_missing_count'] / len(col)

# Interaction features (try more)
train['id_28_combo'] = (train['id_29'].astype(str) + '_' + train['id_28'].astype(str)).astype('category')

# ProductCD high risk features binry flag
train["ProductCD_is_W"] = (train["ProductCD"] == "W").astype(int)

# UID
train['D1n'] = np.floor(train['TransactionDT'] / (24*60*60)) - train['D1']
train['UID'] = (train['card1'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['P_emaildomain'].astype(str)+'_'+train['D1n'].astype(str)).astype('category')

# Group statistics
test['D1_UID_std']  = test['UID'].map(feature_maps['D1_UID_std'])
test['D6_UID_std']  = test['UID'].map(feature_maps['D6_UID_std'])
test['D11_UID_mean'] = test['UID'].map(feature_maps['D11_UID_mean'])
test['D11_UID_std']  = test['UID'].map(feature_maps['D11_UID_std'])
test['D12_UID_std']  = test['UID'].map(feature_maps['D12_UID_std'])
test['D14_UID_std']  = test['UID'].map(feature_maps['D14_UID_std'])
test['D15_UID_std']  = test['UID'].map(feature_maps['D15_UID_std'])

test['C1_UID_mean'] = test['UID'].map(feature_maps['C1_UID_mean'])
test['C2_UID_std'] = test['UID'].map(feature_maps['C2_UID_std'])
test['C3_UID_mean'] = test['UID'].map(feature_maps['C3_UID_mean'])
test['C3_UID_std'] = test['UID'].map(feature_maps['C3_UID_std'])
test['C6_UID_mean'] = test['UID'].map(feature_maps['C6_UID_mean'])
test['C6_UID_std'] = test['UID'].map(feature_maps['C6_UID_std'])
test['C7_UID_mean'] = test['UID'].map(feature_maps['C7_UID_mean'])
test['C13_UID_mean'] = test['UID'].map(feature_maps['C13_UID_mean'])
test['C13_UID_std'] = test['UID'].map(feature_maps['C13_UID_std'])
test['M1_UID_ct'] = test['UID'].map(feature_maps['M1_UID_ct'])
test['M1_UID_ctt'] = test.set_index(['UID', 'M1']).index.map(feature_maps['M1_UID_ctt'])
test['M2_UID_ct'] = test['UID'].map(feature_maps['M2_UID_ct'])
test['M4_UID_ct'] = test.set_index(['UID', 'M4']).index.map(feature_maps['M4_UID_ct'])
test['M7_UID_ct'] = test.set_index(['UID', 'M7']).index.map(feature_maps['M7_UID_ct'])
test['M9_UID_ct'] = test.set_index(['UID', 'M9']).index.map(feature_maps['M9_UID_ct'])
test['P_emaildomain_UID_ct'] = test.set_index(['UID', 'P_emaildomain']).index.map(feature_maps['P_emaildomain_UID_ct'])


# Frequency encoding 
test['ProductCD_frq'] = test['ProductCD'].map(feature_maps['ProductCD_frq'])
test['DeviceType_frq'] = test['DeviceType'].map(feature_maps['DeviceType_frq'])
test['id_13_frq'] = test['id_13'].map(feature_maps['id_13_frq'])
test['id_15_frq'] = test['id_15'].map(feature_maps['id_15_frq'])
test['id_19_frq'] = test['id_19'].map(feature_maps['id_19_frq'])
test['card5_frq'] = test['card5'].map(feature_maps['card5_frq'])

# feature selection
high_correlation = ['V71', 'V64', 'V63', 'V60', 'V59', 'V58', 'V43', 'V33', 'V32', 'V31', 
                    '180', 'V17', 'V16']
train = train.drop(columns=high_correlation, errors='ignore')


# load artifacts
with open(project_root / "catboost_artifacts.json") as f:
    artifacts = json.load(f)

feature_order = artifacts["feature_order"]
cat_cols = artifacts["cat_cols"]

# keep labels for scoring (your held-out test split)
y_test = test["isFraud"]

# enforce dtype only for those columns
test[cat_cols] = test[cat_cols].astype(str)

# load model
model = LGBMClassifier()
model.load_model(str(project_root / "catboost_final.cbm"))

# align features
X_test = test[feature_order]

# predict
test_pool = Pool(X_test, cat_features=cat_cols)
y_prob = model.predict_proba(test_pool)[:, 1]

# score
roc = roc_auc_score(y_test, y_prob)
prauc = average_precision_score(y_test, y_prob)
print("ROC-AUC:", roc, "PR-AUC:", prauc)
