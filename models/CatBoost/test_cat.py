from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
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


# Company keywords 
company_keywords = {
"microsoft": ["windows", "trident", "rv:"],
"apple": ["apple", "iphone", "mac", "ios", "macos"],
"samsung": ["samsung", "sm-", "gt-"],
"huawei": ["huawei", "honor", "ale-", "cam-", "pra", "trt-", "was-", "mya-", "rne-", "cro-", 
             "bll-", "chc-"],
"motorola": ["moto"],
"lg": ["lg"],
"zte": ["zte", "blade"],
"xiaomi": ["redmi"],
"htc": ["htc"]
}

# Operating system keywords
operating_systems = {
"os_android": ["samsung", "android", "build/", "sm-", "huawei", "honor", "moto", "xt",
              "lg", "redmi", "zte", "blade", "pixel", "nexus", "kf", "ale-", "cam-","pra", 
              "trt-", "was-", "mya-", "rne-", "cro-", "bll-", "chc-", "gt-", "htc", "hi6", 
              "hisense"],
"os_windows": ["windows", "trident", "rv:"],
"os_ios": ["iphone", "ios"],
"os_macos": ["mac", "macos"],
"os_linux": ["linux"]
}
# Create company name feature
def detect_company(x):
    if pd.isna(x):
        return np.nan
    for company, keywords in company_keywords.items():
        if any(k in x for k in keywords):
            return company
    return "other"

test['device_company'] = test['DeviceInfo'].apply(detect_company)
# Create operating system feature
def detect_os(x):
    if pd.isna(x):
        return np.nan
    for os, keywords in operating_systems.items():
        if any(k in x for k in keywords):
            return os
    return "other"

test['device_os'] = test['DeviceInfo'].apply(detect_os)


# Missing-indicators features
cols = ['id_33', 'card2', 'card3']
for col in cols:
    test[col + '_is_missing'] = test[col].isna().astype(int)

    
# Missing-count and missing-ratio features (per group)
group = {
    "card": ['card1', 'card2', 'card3', 'card4', 'card5', 'card6'],
    "M": [f"M{i}" for i in range(1,10)],
    "V": [f"V{i}" for i in range(1,340)],
    "id": [f"id_{str(i).zfill(2)}" for i in range(1,39)]
}

for name, col in group.items():
    col = [c for c in col if c in test.columns] 
    test[f'{name}_missing_count'] = test[col].isna().sum(axis=1)
    test[f'{name}_missing_ratio'] = test[f'{name}_missing_count'] / len(col)


high_risk_vals = [52.0, 49.0, 33.0]
test["id_13_high_risk"] = test["id_13"].isin(high_risk_vals).astype(int)
test["id_13_is_52"] = (test["id_13"] == 52.0).astype(int)
test["id_13_is_49"] = (test["id_13"] == 49.0).astype(int)
test["id_13_is_33"] = (test["id_13"] == 33.0).astype(int)
test["id_14_high_risk"] = (test["id_14"] == -300.0).astype(int)
test["id_17_is_225"] = (test["id_17"] == 225.0).astype(int)
test["id_17_is_166"] = (test["id_17"] == 166.0).astype(int)
test["id_18_is_15"] = (test["id_18"] == 15.0).astype(int)
test["id_19_is_266"] = (test["id_19"] == 266.0).astype(int)
test["id_20_is_507"] = (test["id_20"] == 507.0).astype(int)
test["id_20_is_325"] = (test["id_20"] == 325.0).astype(int)
test["card2_is_545"] = (test["card2"] == 545.0).astype(int)
test["card2_is_321"] = (test["card2"] == 321.0).astype(int)
test["card3_is_185"] = (test["card3"] == 185.0).astype(int)
test["card4_is_mastercard"] = (test["card4"] == "mastercard").astype(int)
test["card4_is_visa"] = (test["card4"] == "visa").astype(int)
test["card5_is_102"] = (test["card5"] == 102.0).astype(int)
test["card5_is_137"] = (test["card5"] == 137.0).astype(int)
test["card5_is_138"] = (test["card5"] == 138.0).astype(int)
test["card6_is_credit"] = (test["card6"] == "credit").astype(int)
test["card6_is_debit"] = (test["card6"] == "debit").astype(int)
# ProductCD high risk features binry flag
test["ProductCD_is_C"] = (test["ProductCD"] == "C").astype(int)
test["ProductCD_is_W"] = (test["ProductCD"] == "W").astype(int)
test["ProductCD_is_H"] = (test["ProductCD"] == "H").astype(int)
test["ProductCD_is_R"] = (test["ProductCD"] == "R").astype(int)

# UID features 
test['D1n'] = np.floor(test['TransactionDT'] / (24*60*60)) - test['D1']
test['UID'] = test['card1'].astype(str)+'_'+test['addr1'].astype(str)+'_'+test['P_emaildomain'].astype(str)+'_'+test['D1n'].astype(str)

# Group statistics
test['D1_UID_std']  = test['UID'].map(feature_maps['D1_UID_std'])
test['D6_UID_std']  = test['UID'].map(feature_maps['D6_UID_std'])
test['D11_UID_mean'] = test['UID'].map(feature_maps['D11_UID_mean'])
test['D11_UID_std']  = test['UID'].map(feature_maps['D11_UID_std'])
test['D12_UID_std']  = test['UID'].map(feature_maps['D12_UID_std'])
test['D14_UID_std']  = test['UID'].map(feature_maps['D14_UID_std'])
test['D15_UID_std']  = test['UID'].map(feature_maps['D15_UID_std'])
test['D2_UID_std']  = test['UID'].map(feature_maps['D2_UID_std'])
test['D2_UID_mean']  = test['UID'].map(feature_maps['D2_UID_mean'])
test['D5_UID_std']  = test['UID'].map(feature_maps['D5_UID_std'])
test['D8_UID_std']  = test['UID'].map(feature_maps['D8_UID_std'])
test['D9_UID_std']  = test['UID'].map(feature_maps['D9_UID_std'])
test['D10_UID_std']  = test['UID'].map(feature_maps['D10_UID_std'])
test['C1_UID_mean'] = test['UID'].map(feature_maps['C1_UID_mean'])
test['C2_UID_std'] = test['UID'].map(feature_maps['C2_UID_std'])
test['C3_UID_mean'] = test['UID'].map(feature_maps['C3_UID_mean'])
test['C3_UID_std'] = test['UID'].map(feature_maps['C3_UID_std'])
test['C6_UID_mean'] = test['UID'].map(feature_maps['C6_UID_mean'])
test['C6_UID_std'] = test['UID'].map(feature_maps['C6_UID_std'])
test['C7_UID_mean'] = test['UID'].map(feature_maps['C7_UID_mean'])
test['C13_UID_mean'] = test['UID'].map(feature_maps['C13_UID_mean'])
test['C13_UID_std'] = test['UID'].map(feature_maps['C13_UID_std'])
test['C5_UID_std'] = test['UID'].map(feature_maps['C5_UID_std'])
test['C11_UID_std'] = test['UID'].map(feature_maps['C11_UID_std'])
test['C14_UID_std'] = test['UID'].map(feature_maps['C14_UID_std'])
test['M1_UID_ct'] = test['UID'].map(feature_maps['M1_UID_ct'])
test['M1_UID_ctt'] = test.set_index(['UID', 'M1']).index.map(feature_maps['M1_UID_ctt'])
test['M2_UID_ct'] = test['UID'].map(feature_maps['M2_UID_ct'])
test['M3_UID_ct'] = test['UID'].map(feature_maps['M3_UID_ct'])
test['M4_UID_ct'] = test.set_index(['UID', 'M4']).index.map(feature_maps['M4_UID_ct'])
test['M7_UID_ct'] = test.set_index(['UID', 'M7']).index.map(feature_maps['M7_UID_ct'])
test['M9_UID_ct'] = test.set_index(['UID', 'M9']).index.map(feature_maps['M9_UID_ct'])
test['P_emaildomain_UID_ct'] = test.set_index(['UID', 'P_emaildomain']).index.map(feature_maps['P_emaildomain_UID_ct'])
test['M5_UID_ct'] = test.set_index(['UID', 'M5']).index.map(feature_maps['M5_UID_ct'])
test['M2_UID_ctt'] = test.set_index(['UID', 'M2']).index.map(feature_maps['M2_UID_ctt'])
test['M6_UID_ctt'] = test.set_index(['UID', 'M6']).index.map(feature_maps['M6_UID_ctt'])


# feature selection
feature_list = [
    "id_18","id_20_frq","id_05","amt_card5_mean","V259","amt_card1_addr1_mean","D14","DeviceInfo_frq","id_02",
    "card2_frq","id_19_frq","V220","card1_addr1_frq","V83","id_04","V134","D1","V275","V132","V53","V283",
    "card1_addr1_pemail_frq","M5_frq","id_30","id_34_frq","D14_UID_std","V52","V127",
    "V221","V3","V309","V311","V72","V105","M3_UID_ct","V273","V128","V266","id_18_frq","V119",
    "V278","V300","D6_UID_std","device_os","C2_UID_std","id_missing_count","V99","V5","id_38","V246","id_17_frq",
    "V198","id_13_frq","V203","card5_frq","ProductCD_is_H","V222","V207","V315","id_17","V87","V80","V30","V7",
    "V126","D8_UID_std","V277","V157","D7_adj","V205","V228","C5_UID_std","V12","V301","D7","id_31_frq",
    "V296","V176","V250","V242","V224","V40","V103","amt_card1_std","V85","V234","V36","V261","id_32",
    "id_38_frq","addr2_frq","V320","V265","id_15_frq","is_browser","V178","V264","id_14_frq","amt_addr1_mean",
    "id_34","V101","V120","V190","M1_UID_ct","V170","V66","V124","V240","V202","id_12","V15","amt_card4_std",
    "V84","V298","V24","V321","V162","V18","V138","V284","V46","V215","V114","id_11","V57","V173","V323",
    "V16","id_10","V65","M8","V10","V71","V192","V168","V263","amt_card3_mean","V286","V160","V249","V22",
    "V161","V74","V145","id_16_frq","V167","V262","V180","V310","V147","V253","M1_UID_ctt","V191","V158",
    "V69","card4_frq","V113","V31","amt_card6_std","V169","V327",
    
    
    "V41","id_20_is_507","V333","V334","id_18_is_15","V106","V107","id_19_is_266","D11","V335","V336","V11",
    "id_20_is_325","V110","V111","V337","V112","V338","V339","id_33_is_missing","V118","V117","V42","V64",
    "card2_is_321","id_17_is_225","card5_is_102","V96","V98","ProductCD_is_C","card5_is_138","card5_is_137",
    "id_17_is_166","card3_is_185","card2_is_545","V95","V94","V1","id_14_high_risk","id_13_is_52","V68","V73",
    "id_13_is_33","id_13_high_risk","V8","V100","V88","V89","V90","id_13_is_49","V166","V330","V302","V181","V183",
    "V299","V297","V184","V185","V14","V144","V285","V193","V238","V28","V279","V197","V199","V223","V21","V27","V269",
    "V268","V155","V255","V211","V141","V140","V227","V225","V122","V243","V328","V325","V324","V322","V241","V32","V237",
    "V121","V235","V212","V305","V172"
    ]

test = test.drop(columns=feature_list, errors="ignore")

# update categorical features 
cat_cols = test.select_dtypes(include=["object","category"]).columns.tolist()
test[cat_cols] = test[cat_cols].astype(str)  


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
model = CatBoostClassifier()
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
