import pandas as pd
from pathlib import Path
import numpy as np
import re

# Load files
project_root = Path(__file__).resolve().parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train_cleaned.parquet")

# Sanity checks
print("\nTrain Split (clean)")
print("Rows and columns: ", train.shape)
print("Dulicates: ", train.duplicated().sum())
print("Missing Values %: ", train.isna().mean().mean() *100)
print("Data types: ", train.dtypes.value_counts())

print(train['TransactionDT'].min()) 
print(train['TransactionDT'].max())
 
# Correct data types 
columns_category = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 
                    'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']

train[columns_category] = train[columns_category].astype('category')


# sort values to keep timely order
train = train.sort_values(['TransactionDT']).reset_index(drop=True)

# Feature engineering
# Missing-count and missing-ratio features (per row)
train['missing_count'] = train.isna().sum(axis=1)
train['missing_ratio'] = train['missing_count'] / train.shape[1]

# Missing-count and missing-ratio features (per group)
group = {
    "card": ['card1', 'card2', 'card3', 'card4', 'card5', 'card6'],
    "addr": ['addr1', 'addr2'],
    "dist": ['dist1', 'dist2'],
    "email": ['P_emaildomain','R_emaildomain'],
    "device": ['DeviceType','DeviceInfo'],
    "C": [f"C{i}" for i in range(1,15)],
    "D": [f"D{i}" for i in range(1,16)],
    "M": [f"M{i}" for i in range(1,10)],
    "V": [f"V{i}" for i in range(1,340)],
    "id": [f"id_{str(i).zfill(2)}" for i in range(1,39)]
}

for name, col in group.items():
    col = [c for c in col if c in train.columns] 
    train[f'{name}_missing_count'] = train[col].isna().sum(axis=1)
    train[f'{name}_missing_ratio'] = train[f'{name}_missing_count'] / len(col)


# Missing-indicators features
cols = ['card2', 'card3', 'card4', 'card5', 'card6',
        'dist1', 'dist2', 'addr1', 'addr2',
        'P_emaildomain', 'R_emaildomain',
        'DeviceInfo', 'DeviceType']

for col in cols:
    train[col + '_is_missing'] = train[col].isna().astype(int)



# Time-based features
# Duration features
train['hours_duration'] = train['TransactionDT'] / (60 * 60)
train['days_duration'] = train['TransactionDT'] / (60 * 60 * 24)

# Cyclical calendar features 
# Minute (0–59)
train['minute'] = (train['TransactionDT'] // 60) % 60
train['minute_sin'] = np.sin(2 * np.pi * train['minute'] / 60)
train['minute_cos'] = np.cos(2 * np.pi * train['minute'] / 60)

# Hour (0–23)
train['hour'] = (train['TransactionDT'] // 3600) % 24
train['hour_sin'] = np.sin(2 * np.pi * train['hour'] / 24)
train['hour_cos'] = np.cos(2 * np.pi * train['hour'] / 24)

# Weekday (0–6)
train['weekday'] = (train['TransactionDT'] // 86400) %  7
train['weekday_sin'] = np.sin(2 * np.pi * train['weekday'] / 7)
train['weekday_cos'] = np.cos(2 * np.pi * train['weekday'] / 7)


# Elapsed features 
train['elapsed_days'] = (train['TransactionDT'] - train['TransactionDT'].min()) // 86400


# Natural logarithm of transaction amount
train['amount_log'] = np.log1p(train['TransactionAmt'])

# Transaction amount features
train['dollars'] = train['TransactionAmt'].astype(int)
train['cents'] = (np.round(train['TransactionAmt'] * 100) % 100).astype(int)


# Email extracted features
train['P_emaildomain'] = train['P_emaildomain'].str.lower()
train['R_emaildomain'] = train['R_emaildomain'].str.lower()

train['parent_domain_p'] = train['P_emaildomain'].str.split('.').str[0]
train['parent_domain_r'] = train['R_emaildomain'].str.split('.').str[0]

train['suffix_p'] = train['P_emaildomain'].str.split(pat='.', n=1).str[1]
train['suffix_r'] = train['R_emaildomain'].str.split(pat='.', n=1).str[1]

train['tld_p'] = train['P_emaildomain'].str.split('.').str[-1]
train['tld_r'] = train['R_emaildomain'].str.split('.').str[-1]


# Device Info extracted features 
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

# Mobile and browser keywords 
device_mobile = ["sm-", "gt-", "ale-", "cam-", "trt-", "was-", "mya-", "rne-", "cro-", 
                 "bll-", "chc-", "pra", "android", "build/", "huawei", "honor", "hisense", 
                 "zte", "htc", "moto", "xt", "samsung", "mi ", "redmi", "pixel", "nexus", "kf", 
                 "lg", "iphone", "ios"]


device_browser = ["windows", "trident", "rv:", "macos", "mac",
                  "linux"]

# Create mobile and browser binary features 
train['DeviceInfo'] = train['DeviceInfo'].str.lower()
train['is_mobile'] = train['DeviceInfo'].str.contains('|'.join(map(re.escape, device_mobile)),
                                                      regex=True, na=None).astype(float)
train['is_browser'] = train['DeviceInfo'].str.contains('|'.join(map(re.escape, device_browser)),
                                                       regex=True, na=None).astype(float)

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




# Frequency encoding features 
train['card1_frq'] = train['card1'].map(train['card1'].value_counts())
train['card3_frq'] = train['card3'].map(train['card3'].value_counts())
train['addr1_frq'] = train['addr1'].map(train['addr1'].value_counts())
train['addr2_frq'] = train['addr2'].map(train['addr2'].value_counts())
train['id_30_frq'] = train['id_30'].map(train['id_30'].value_counts())
train['P_emaildomain_frq'] = train['P_emaildomain'].map(train['P_emaildomain'].value_counts())


# Create unique user identifiers (UIDs)
train['uid1'] = train['card1'].astype(str)
train['uid2'] = train['card2'].astype(str) + '_' + train['card3'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train['P_emaildomain'].astype(str) 
train['uid3'] = train['card2'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train['P_emaildomain'].astype(str) 

# Create velocity features
# Recency feature per uid (time since last transaction)
train = train.sort_values(['uid1', 'TransactionDT'])
train['recency_uid1'] = train.groupby('uid1')['TransactionDT'].diff()

train = train.sort_values(['uid2', 'TransactionDT'])
train['recency_uid2'] = train.groupby('uid2')['TransactionDT'].diff()

train = train.sort_values(['uid3', 'TransactionDT'])
train['recency_uid3'] = train.groupby('uid3')['TransactionDT'].diff()


# Transaction frequency and spending velocity features 
train['ts'] = pd.to_datetime('2017-12-01') + pd.to_timedelta(train['TransactionDT'], unit='s')
uids  = ['uid1', 'uid2', 'uid3']
wins  = ['1h', '12h', '1d', '7d']

for u in uids:
    train = train.sort_values([u, 'ts'])
    for w in wins:
        ccol = f'txn_count_{w}_{u}'
        scol = f'txn_sum_{w}_{u}'
        train[ccol] = np.nan
        train[scol] = np.nan

        for _, g in train.groupby(u, sort=False):
            s = g.set_index('ts')['TransactionAmt'].shift(1)
            train.loc[g.index, ccol] = s.rolling(w).count().values
            train.loc[g.index, scol] = s.rolling(w).sum().values



# Missing-indicators features
cols = ['M1', 'M2', 'M3', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
        'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 'id_31', 'id_32', 'id_33',
        'id_34', 'id_35', 'id_36', 'id_37', 'id_38']
for col in cols:
    train[col + '_is_missing'] = train[col].isna().astype(int)

# ProductCD high risk features binry flags
train["ProductCD_is_C"] = (train["ProductCD"] == "C").astype(int)
train["ProductCD_is_W"] = (train["ProductCD"] == "W").astype(int)
train["ProductCD_is_H"] = (train["ProductCD"] == "H").astype(int)
train["ProductCD_is_R"] = (train["ProductCD"] == "R").astype(int)


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


# TransactionAMT split features
train['TransactionAmt_int'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[0].astype(float)
train['TransactionAmt_dec'] = train['TransactionAmt'].astype(str).str.split('.', expand=True)[1].astype(float)


# split features
col_split = ['id_30', 'id_31', 'id_33', 'id_34']
for col in col_split:
    parts = train[col].astype(str).str.split(r'[ /_]', n=1, expand=True)
    parts = parts.reindex(columns=[0, 1]) 
    train[f"{col}_part1"] = parts[0].astype(str).astype('category')
    train[f"{col}_part2"] = parts[1].astype(str).astype('category')



# UID
train['card1_addr1'] = train['card1'].astype(str) + '_' + train['addr1'].astype(str)
train['UID'] = (
    train['card1_addr1'].astype(str) + '_' +
    np.floor(train['TransactionDT'] - train['D1']).astype(str)
)


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
train['amt_comp_mean'] = train.groupby('device_company')['TransactionAmt'].transform('mean')
train['amt_comp_std']  = train.groupby('device_company')['TransactionAmt'].transform('std')
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


# Frequency encoding 
train['card2_frq'] = train['card2'].map(train['card2'].value_counts())
train['email_frq'] = train['P_emaildomain'].map(train['P_emaildomain'].value_counts())
train['card1_addr1_comb'] = train['card1'].astype(str) + '_' + train['addr1'].astype(str)
train['card1_addr1_frq'] = train['card1_addr1_comb'].map(train['card1_addr1_comb'].value_counts())
train['card1_addr1_pemail_comb'] = train['card1'].astype(str) + '_' + train['addr1'].astype(str) + '_' + train['P_emaildomain'].astype(str)
train['card1_addr1_pemail_frq'] = train['card1_addr1_pemail_comb'].map(train['card1_addr1_pemail_comb'].value_counts())
train['id_20_frq'] = train['id_20'].map(train['id_20'].value_counts())
train['D7_adj'] = train['D7'] - (train['TransactionDT'] / (24*60*60))
train['D7_norm'] = (train['D7_adj'] - train['D7_adj'].mean()) / train['D7_adj'].std()
train['id_24_frq'] = train['id_24'].map(train['id_24'].value_counts())
train['id_25_frq'] = train['id_25'].map(train['id_25'].value_counts())
train['id_26_frq'] = train['id_26'].map(train['id_26'].value_counts())





# drop ts column
train = train.drop(columns=['ts'])

