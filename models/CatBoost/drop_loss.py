from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import average_precision_score
import re


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



# UID
train['D1n'] = np.floor(train['TransactionDT'] / (24*60*60)) - train['D1']
train['UID'] = train['card1'].astype(str)+'_'+train['addr1'].astype(str)+'_'+train['P_emaildomain'].astype(str)+'_'+train['D1n'].astype(str)


# ProductCD high risk features binry flag
train["ProductCD_is_C"] = (train["ProductCD"] == "C").astype(int)
train["ProductCD_is_W"] = (train["ProductCD"] == "W").astype(int)
train["ProductCD_is_H"] = (train["ProductCD"] == "H").astype(int)
train["ProductCD_is_R"] = (train["ProductCD"] == "R").astype(int)


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


# Group statistics 
train['amt_card1_mean'] = train.groupby('card1')['TransactionAmt'].transform('mean')
train['amt_card1_std']  = train.groupby('card1')['TransactionAmt'].transform('std')
train['amt_card2_mean'] = train.groupby('card2')['TransactionAmt'].transform('mean')
train['amt_card2_std']  = train.groupby('card2')['TransactionAmt'].transform('std')
train['amt_card3_mean'] = train.groupby('card3')['TransactionAmt'].transform('mean')
train['amt_card3_std']  = train.groupby('card3')['TransactionAmt'].transform('std')
train['amt_card5_mean'] = train.groupby('card5')['TransactionAmt'].transform('mean')
train['amt_card5_std']  = train.groupby('card5')['TransactionAmt'].transform('std')
train['amt_card6_mean'] = train.groupby('card6')['TransactionAmt'].transform('mean')
train['amt_card6_std']  = train.groupby('card6')['TransactionAmt'].transform('std')
train['amt_addr1_mean'] = train.groupby('addr1')['TransactionAmt'].transform('mean')
train['amt_addr1_std']  = train.groupby('addr1')['TransactionAmt'].transform('std')
train['amt_addr2_mean'] = train.groupby('addr2')['TransactionAmt'].transform('mean')
train['amt_addr2_std']  = train.groupby('addr2')['TransactionAmt'].transform('std')
train['amt_card1_addr1_mean'] = (train.groupby(['card1','addr1'])['TransactionAmt']
           .transform('mean'))
train['amt_card1_addr1_std'] = (train.groupby(['card1','addr1'])['TransactionAmt']
           .transform('std'))
train['amt_card1_addr1_mean'] = (train.groupby(['card1','addr1', 'P_emaildomain'])['TransactionAmt'].transform('mean'))
train['amt_card1_addr1_std'] = (train.groupby(['card1','addr1', 'P_emaildomain'])['TransactionAmt'].transform('std'))
train['D11_card1_mean'] = (train.groupby(['card1', 'D11'])['TransactionAmt']
           .transform('mean'))
train['D11_card1_std'] = (train.groupby(['card1', 'D11'])['TransactionAmt']
           .transform('std'))
train['amt_card4_mean'] = train.groupby('card4')['TransactionAmt'].transform('mean')
train['amt_card4_std']  = train.groupby('card4')['TransactionAmt'].transform('std')



# Frequency encoding 
train['card5_frq'] = train['card5'].map(train['card5'].value_counts())
train['ProductCD_frq'] = train['ProductCD'].map(train['ProductCD'].value_counts())
train['DeviceType_frq'] = train['DeviceType'].map(train['DeviceType'].value_counts())
train['id_13_frq'] = train['id_13'].map(train['id_13'].value_counts())
train['id_15_frq'] = train['id_15'].map(train['id_15'].value_counts())
train['id_19_frq'] = train['id_19'].map(train['id_19'].value_counts())
train['card2_frq'] = train['card2'].map(train['card2'].value_counts())
train['email_frq'] = train['P_emaildomain'].map(train['P_emaildomain'].value_counts())
train['card1_addr1_comb'] = train['card1'].astype(str) + '_' + train['addr1'].astype(str)
train['card1_addr1_frq'] = train['card1_addr1_comb'].map(train['card1_addr1_comb'].value_counts())
train['card1_addr1_pemail_comb'] = train['card1'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train['P_emaildomain'].astype(str)
train['card1_addr1_pemail_frq'] = train['card1_addr1_pemail_comb'].map(train['card1_addr1_pemail_comb'].value_counts())
train['id_20_frq'] = train['id_20'].map(train['id_20'].value_counts())
train['D7_adj'] = train['D7'] - (train['TransactionDT'] / (24*60*60))
train['D7_norm'] = (train['D7_adj'] - train['D7_adj'].mean()) / train['D7_adj'].std()


high_risk_vals = [52.0, 49.0, 33.0]
train["id_13_high_risk"] = train["id_13"].isin(high_risk_vals).astype(int)

train["id_13_is_52"] = (train["id_13"] == 52.0).astype(int)
train["id_13_is_49"] = (train["id_13"] == 49.0).astype(int)
train["id_13_is_33"] = (train["id_13"] == 33.0).astype(int)

train["id_14_high_risk"] = (train["id_14"] == -300.0).astype(int)

train["id_17_is_225"] = (train["id_17"] == 225.0).astype(int)
train["id_17_is_166"] = (train["id_17"] == 466.0).astype(int)

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


columns = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']
freq_maps = {col: train[col].value_counts() for col in columns}
for col in columns:
    train[f"{col}_frq"] = train[col].map(freq_maps[col])



# update categorical features 
cat_cols = train.select_dtypes(include=["object","category"]).columns.tolist()
train[cat_cols] = train[cat_cols].astype(str)  



# model training
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
