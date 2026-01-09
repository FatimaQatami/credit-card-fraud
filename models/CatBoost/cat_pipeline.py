from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, matthews_corrcoef)
from imblearn.metrics import geometric_mean_score
import re
import joblib
import json
 

# load dataset
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

# Correct data types 
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']
train[cat_cols] = train[cat_cols].astype(str)


# Email extracted features
train['P_emaildomain'] = train['P_emaildomain'].str.lower()
train['R_emaildomain'] = train['R_emaildomain'].str.lower()
train['suffix_r'] = train['R_emaildomain'].str.split(pat='.', n=1).str[1]
train['tld_r'] = train['R_emaildomain'].str.split('.').str[-1]


# Mobile and browser keywords 
device_mobile = ["sm-", "gt-", "ale-", "cam-", "trt-", "was-", "mya-", "rne-", "cro-", 
                 "bll-", "chc-", "pra", "android", "build/", "huawei", "honor", "hisense", 
                 "zte", "htc", "moto", "xt", "samsung", "mi ", "redmi", "pixel", "nexus", "kf", 
                 "lg", "iphone", "ios"]
device_browser = ["windows", "trident", "rv:", "macos", "mac",
                  "linux"]

# Create mobile and browser binary features 
train['DeviceInfo'] = train['DeviceInfo'].str.lower().astype(str)
train['is_mobile'] = train['DeviceInfo'].str.contains('|'.join(map(re.escape, device_mobile)),
                                                      regex=True, na=None).astype(float)
train['is_browser'] = train['DeviceInfo'].str.contains('|'.join(map(re.escape, device_browser)),
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

train['device_company'] = train['DeviceInfo'].apply(detect_company)

# Create operating system feature
def detect_os(x):
    if pd.isna(x):
        return np.nan
    for os, keywords in operating_systems.items():
        if any(k in x for k in keywords):
            return os
    return "other"

train['device_os'] = train['DeviceInfo'].apply(detect_os)



# Missing-indicators features
cols = ['id_33', 'card2', 'card3']
for col in cols:
    train[col + '_is_missing'] = train[col].isna().astype(int)

    
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


high_risk_vals = [52.0, 49.0, 33.0]
train["id_13_high_risk"] = train["id_13"].isin(high_risk_vals).astype(int)
train["id_13_is_52"] = (train["id_13"] == 52.0).astype(int)
train["id_13_is_49"] = (train["id_13"] == 49.0).astype(int)
train["id_13_is_33"] = (train["id_13"] == 33.0).astype(int)
train["id_14_high_risk"] = (train["id_14"] == -300.0).astype(int)
train["id_17_is_225"] = (train["id_17"] == 225.0).astype(int)
train["id_17_is_166"] = (train["id_17"] == 166.0).astype(int)
train["id_18_is_15"] = (train["id_18"] == 15.0).astype(int)
train["id_19_is_266"] = (train["id_19"] == 266.0).astype(int)
train["id_20_is_507"] = (train["id_20"] == 507.0).astype(int)
train["id_20_is_325"] = (train["id_20"] == 325.0).astype(int)
train["card2_is_545"] = (train["card2"] == 545.0).astype(int)
train["card2_is_321"] = (train["card2"] == 321.0).astype(int)
train["card3_is_185"] = (train["card3"] == 185.0).astype(int)
train["card4_is_mastercard"] = (train["card4"] == "mastercard").astype(int)
train["card4_is_visa"] = (train["card4"] == "visa").astype(int)
train["card5_is_102"] = (train["card5"] == 102.0).astype(int)
train["card5_is_137"] = (train["card5"] == 137.0).astype(int)
train["card5_is_138"] = (train["card5"] == 138.0).astype(int)
train["card6_is_credit"] = (train["card6"] == "credit").astype(int)
train["card6_is_debit"] = (train["card6"] == "debit").astype(int)
# ProductCD high risk features binry flag
train["ProductCD_is_C"] = (train["ProductCD"] == "C").astype(int)
train["ProductCD_is_W"] = (train["ProductCD"] == "W").astype(int)
train["ProductCD_is_H"] = (train["ProductCD"] == "H").astype(int)
train["ProductCD_is_R"] = (train["ProductCD"] == "R").astype(int)


# UID
train['D1n'] = np.floor(train['TransactionDT'] / (24*60*60)) - train['D1']
train['UID'] = train['card1'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['P_emaildomain'].astype(str)+'_'+train['D1n'].astype(str)


# Group statistics (use cat feats with count both types and num feats with std/mean)
train['D1_UID_std']  = train.groupby('UID')['D1'].transform('std')
train['D6_UID_std']  = train.groupby('UID')['D6'].transform('std')
train['D11_UID_mean'] = train.groupby('UID')['D11'].transform('mean')
train['D11_UID_std']  = train.groupby('UID')['D11'].transform('std')
train['D12_UID_std']  = train.groupby('UID')['D12'].transform('std')
train['D14_UID_std']  = train.groupby('UID')['D14'].transform('std')
train['D15_UID_std']  = train.groupby('UID')['D15'].transform('std')
train['D2_UID_std']  = train.groupby('UID')['D2'].transform('std')
train['D2_UID_mean']  = train.groupby('UID')['D2'].transform('mean')
train['D5_UID_std']  = train.groupby('UID')['D5'].transform('std')
train['D8_UID_std']  = train.groupby('UID')['D8'].transform('std')
train['D9_UID_std']  = train.groupby('UID')['D9'].transform('std')
train['D10_UID_std']  = train.groupby('UID')['D10'].transform('std')
train['C1_UID_mean'] = train.groupby('UID')['C1'].transform('mean')
train['C2_UID_std'] = train.groupby('UID')['C2'].transform('std')
train['C3_UID_mean'] = train.groupby('UID')['C3'].transform('mean')
train['C3_UID_std'] = train.groupby('UID')['C3'].transform('std')
train['C6_UID_mean'] = train.groupby('UID')['C6'].transform('mean')
train['C6_UID_std'] = train.groupby('UID')['C6'].transform('std')
train['C7_UID_mean'] = train.groupby('UID')['C7'].transform('mean')
train['C13_UID_mean'] = train.groupby('UID')['C13'].transform('mean')
train['C13_UID_std'] = train.groupby('UID')['C13'].transform('std')
train['C5_UID_std'] = train.groupby('UID')['C5'].transform('std')
train['C11_UID_std'] = train.groupby('UID')['C11'].transform('std')
train['C14_UID_std'] = train.groupby('UID')['C14'].transform('std')
train['M1_UID_ct'] = train.groupby('UID')['M1'].transform('count')
train['M1_UID_ctt'] = train.groupby(['UID', 'M1'])['M1'].transform('count')
train['M2_UID_ct'] = train.groupby('UID')['M2'].transform('count')
train['M3_UID_ct'] = train.groupby('UID')['M3'].transform('count')
train['M4_UID_ct'] = train.groupby(['UID', 'M4'])['M4'].transform('count')
train['M7_UID_ct'] = train.groupby(['UID', 'M7'])['M7'].transform('count')
train['M9_UID_ct'] = train.groupby(['UID', 'M9'])['M9'].transform('count')
train['P_emaildomain_UID_ct'] = train.groupby(['UID', 'P_emaildomain'])['P_emaildomain'].transform('count')
train['M5_UID_ct'] = train.groupby(['UID', 'M5'])['M5'].transform('count')
train['M2_UID_ctt'] = train.groupby(['UID', 'M2'])['M2'].transform('count')
train['M6_UID_ctt'] = train.groupby(['UID', 'M6'])['M6'].transform('count')



feature_maps = {
    # Group statistics (use cat feats with count both types and num feats with std/mean)
    'D1_UID_std' : train.groupby('UID')['D1'].std(),
    'D6_UID_std' : train.groupby('UID')['D6'].std(),
    'D11_UID_mean' : train.groupby('UID')['D11'].mean(),
    'D11_UID_std'  : train.groupby('UID')['D11'].std(),
    'D12_UID_std'  : train.groupby('UID')['D12'].std(),
    'D14_UID_std'  : train.groupby('UID')['D14'].std(),
    'D15_UID_std'  : train.groupby('UID')['D15'].std(),
    'D2_UID_std'  : train.groupby('UID')['D2'].std(),
    'D2_UID_mean' : train.groupby('UID')['D2'].mean(),
    'D5_UID_std'  : train.groupby('UID')['D5'].std(),
    'D8_UID_std'  : train.groupby('UID')['D8'].std(),
    'D9_UID_std'  : train.groupby('UID')['D9'].std(),
    'D10_UID_std' : train.groupby('UID')['D10'].std(),
    'C1_UID_mean' : train.groupby('UID')['C1'].mean(),
    'C2_UID_std' : train.groupby('UID')['C2'].std(),
    'C3_UID_mean' : train.groupby('UID')['C3'].mean(),
    'C3_UID_std' : train.groupby('UID')['C3'].std(),
    'C6_UID_mean' : train.groupby('UID')['C6'].mean(),
    'C6_UID_std' : train.groupby('UID')['C6'].std(),
    'C7_UID_mean' : train.groupby('UID')['C7'].mean(),
    'C13_UID_mean' : train.groupby('UID')['C13'].mean(),
    'C13_UID_std' : train.groupby('UID')['C13'].std(),
    'C5_UID_std' : train.groupby('UID')['C5'].std(),
    'C11_UID_std' : train.groupby('UID')['C11'].std(),
    'C14_UID_std' : train.groupby('UID')['C14'].std(),
    'M1_UID_ct' : train.groupby('UID')['M1'].count(),
    'M1_UID_ctt' : train.groupby(['UID', 'M1'])['M1'].count(),
    'M2_UID_ct' : train.groupby('UID')['M2'].count(),
    'M3_UID_ct' : train.groupby('UID')['M3'].count(),
    'M4_UID_ct' : train.groupby(['UID', 'M4'])['M4'].count(),
    'M7_UID_ct' : train.groupby(['UID', 'M7'])['M7'].count(),
    'M9_UID_ct' : train.groupby(['UID', 'M9'])['M9'].count(),
    'P_emaildomain_UID_ct' : train.groupby(['UID', 'P_emaildomain'])['P_emaildomain'].count(),
    'M5_UID_ct' : train.groupby(['UID', 'M5'])['M5'].count(),
    'M2_UID_ctt' : train.groupby(['UID', 'M2'])['M2'].count(),
    'M6_UID_ctt' : train.groupby(['UID', 'M6'])['M6'].count(),
}
joblib.dump(feature_maps, project_root / "feature_maps.joblib")


# feature selection
feature_list = [
    "id_18","id_05","V259","D14","id_02",
    "V220","V83","id_04","V134","D1","V275","V132","V53","V283",
    "id_30","V52","V127",
    "V221","V3","V309","V311","V72","V105","V273","V128","V266","V119",
    "V278","V300","device_os","id_missing_count","V99","V5","id_38","V246",
    "V198","V203","ProductCD_is_H","V222","V207","V315","id_17","V87","V80","V30","V7",
    "V126","V277","V157","V205","V228","V12","V301","D7",
    "V296","V176","V250","V242","V224","V40","V103","V85","V234","V36","V261","id_32",
    "V320","V265","is_browser","V178","V264",
    "id_34","V101","V120","V190","V170","V66","V124","V240","V202","id_12","V15",
    "V84","V298","V24","V321","V162","V18","V138","V284","V46","V215","V114","id_11","V57","V173","V323",
    "V16","id_10","V65","M8","V10","V71","V192","V168","V263","V286","V160","V249","V22",
    "V161","V74","V145","V167","V262","V180","V310","V147","V253","V191","V158",
    "V69","V113","V31","V169","V327",
    
    "amt_card6_std","card4_frq","id_16_frq","M1_UID_ctt","amt_card3_mean","amt_card4_std","M1_UID_ct","id_38_frq","addr2_frq",
    "id_14_frq","amt_addr1_mean","amt_card1_std","id_15_frq","id_31_frq","C5_UID_std","D8_UID_std","D7_adj","id_13_frq",
    "card5_frq","id_17_frq","C2_UID_std","D6_UID_std","M3_UID_ct","id_18_frq","card1_addr1_pemail_frq","M5_frq","id_34_frq","D14_UID_std",
    "card2_frq","id_19_frq","card1_addr1_frq","id_20_frq","amt_card5_mean","amt_card1_addr1_mean","DeviceInfo_frq",

    "V41","id_20_is_507","V333","V334","id_18_is_15","V106","V107","id_19_is_266","D11","V335","V336","V11",
    "id_20_is_325","V110","V111","V337","V112","V338","V339","id_33_is_missing","V118","V117","V42","V64",
    "card2_is_321","id_17_is_225","card5_is_102","V96","V98","ProductCD_is_C","card5_is_138","card5_is_137",
    "id_17_is_166","card3_is_185","card2_is_545","V95","V94","V1","id_14_high_risk","id_13_is_52","V68","V73",
    "id_13_is_33","id_13_high_risk","V8","V100","V88","V89","V90","id_13_is_49","V166","V330","V302","V181","V183",
    "V299","V297","V184","V185","V14","V144","V285","V193","V238","V28","V279","V197","V199","V223","V21","V27","V269",
    "V268","V155","V255","V211","V141","V140","V227","V225","V122","V243","V328","V325","V324","V322","V241","V32","V237",
    "V121","V235","V212","V305","V172"
    ]

train = train.drop(columns=feature_list, errors='ignore')


# update categorical features 
cat_cols = train.select_dtypes(include=["object","category"]).columns.tolist()
train[cat_cols] = train[cat_cols].astype(str)  


# model training
X = train.drop(columns=['isFraud'])
y = train['isFraud']

train['month'] = train['TransactionDT'] // (30 * 24 * 60 * 60)
groups = train['month']

val_month = groups.max()
val_mask = (groups == val_month)

X_train, y_train = X.loc[~val_mask], y.loc[~val_mask]
X_val,   y_val   = X.loc[val_mask],  y.loc[val_mask]

train_pool = Pool(X_train, y_train, cat_features=cat_cols)
val_pool   = Pool(X_val,   y_val,   cat_features=cat_cols)

model = CatBoostClassifier(
    random_state=42,
    eval_metric="PRAUC",
    objective="Logloss",
    thread_count=-1,
    class_weights=[1, 3],
)

model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=100, verbose=False)

# metrics on holdout
y_prob = model.predict_proba(val_pool)[:, 1]
roc = roc_auc_score(y_val, y_prob)
prauc = average_precision_score(y_val, y_prob)

y_pred = (y_prob >= 0.5).astype(int)
acc = accuracy_score(y_val, y_pred)
prec = precision_score(y_val, y_pred)
rec = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
f2 = fbeta_score(y_val, y_pred, beta=2)
gmean = geometric_mean_score(y_val, y_pred)
mcc = matthews_corrcoef(y_val, y_pred)

cm = confusion_matrix(y_val, y_pred)
tn, fp, fn, tp = cm.ravel()
fnr = fn / (fn + tp)
fpr = fp / (fp + tn)

results = pd.DataFrame([{
    'roc_auc': roc, 'pr_auc': prauc, 'accuracy': acc, 'precision': prec,
    'recall': rec, 'f1': f1, 'f2': f2, 'gmean': gmean, 'mcc': mcc,
    'fnr': fnr, 'fpr': fpr
}])
results.to_csv(project_root / "cat_feat.csv", index=False)




# train final on ALL data using best iteration from holdout training
final_iters = model.get_best_iteration()

full_pool = Pool(X, y, cat_features=cat_cols)
final_model = CatBoostClassifier(
    random_state=42,
    eval_metric="PRAUC",
    objective="Logloss",
    thread_count=-1,
    iterations=final_iters,
)

final_model.fit(full_pool, verbose=False)

final_model.save_model(str(project_root / "catboost_final.cbm"))

artifact = {
    "feature_order": X.columns.tolist(),
    "cat_cols": cat_cols,
    "final_iters": int(final_iters)
}
with open(project_root / "catboost_artifacts.json", "w") as f:
    json.dump(artifact, f, indent=2)
