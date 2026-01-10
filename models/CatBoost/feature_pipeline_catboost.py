import pandas as pd
import re
import numpy as np

def apply_feature_engineering_selection(df):
        
    # Email extracted features
    df['P_emaildomain'] = df['P_emaildomain'].str.lower()
    df['R_emaildomain'] = df['R_emaildomain'].str.lower()
    df['suffix_r'] = df['R_emaildomain'].str.split(pat='.', n=1).str[1]
    df['tld_r'] = df['R_emaildomain'].str.split('.').str[-1]


    # Mobile and browser keywords 
    device_mobile = ["sm-", "gt-", "ale-", "cam-", "trt-", "was-", "mya-", "rne-", "cro-", 
                    "bll-", "chc-", "pra", "android", "build/", "huawei", "honor", "hisense", 
                    "zte", "htc", "moto", "xt", "samsung", "mi ", "redmi", "pixel", "nexus", "kf", 
                    "lg", "iphone", "ios"]
    device_browser = ["windows", "trident", "rv:", "macos", "mac",
                    "linux"]

    # Create mobile and browser binary features 
    df['DeviceInfo'] = df['DeviceInfo'].str.lower().astype(str)
    df['is_mobile'] = df['DeviceInfo'].str.contains('|'.join(map(re.escape, device_mobile)),
                                                        regex=True, na=None).astype(float)
    df['is_browser'] = df['DeviceInfo'].str.contains('|'.join(map(re.escape, device_browser)),
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

    df['device_company'] = df['DeviceInfo'].apply(detect_company)

    # Create operating system feature
    def detect_os(x):
        if pd.isna(x):
            return np.nan
        for os, keywords in operating_systems.items():
            if any(k in x for k in keywords):
                return os
        return "other"

    df['device_os'] = df['DeviceInfo'].apply(detect_os)



    # Missing-indicators features
    cols = ['id_33', 'card2', 'card3']
    for col in cols:
        df[col + '_is_missing'] = df[col].isna().astype(int)

        
    # Missing-count and missing-ratio features (per group)
    group = {
        "card": ['card1', 'card2', 'card3', 'card4', 'card5', 'card6'],
        "M": [f"M{i}" for i in range(1,10)],
        "V": [f"V{i}" for i in range(1,340)],
        "id": [f"id_{str(i).zfill(2)}" for i in range(1,39)]
    }

    for name, col in group.items():
        col = [c for c in col if c in df.columns] 
        df[f'{name}_missing_count'] = df[col].isna().sum(axis=1)
        df[f'{name}_missing_ratio'] = df[f'{name}_missing_count'] / len(col)


    high_risk_vals = [52.0, 49.0, 33.0]
    df["id_13_high_risk"] = df["id_13"].isin(high_risk_vals).astype(int)
    df["id_13_is_52"] = (df["id_13"] == 52.0).astype(int)
    df["id_13_is_49"] = (df["id_13"] == 49.0).astype(int)
    df["id_13_is_33"] = (df["id_13"] == 33.0).astype(int)
    df["id_14_high_risk"] = (df["id_14"] == -300.0).astype(int)
    df["id_17_is_225"] = (df["id_17"] == 225.0).astype(int)
    df["id_17_is_166"] = (df["id_17"] == 166.0).astype(int)
    df["id_18_is_15"] = (df["id_18"] == 15.0).astype(int)
    df["id_19_is_266"] = (df["id_19"] == 266.0).astype(int)
    df["id_20_is_507"] = (df["id_20"] == 507.0).astype(int)
    df["id_20_is_325"] = (df["id_20"] == 325.0).astype(int)
    df["card2_is_545"] = (df["card2"] == 545.0).astype(int)
    df["card2_is_321"] = (df["card2"] == 321.0).astype(int)
    df["card3_is_185"] = (df["card3"] == 185.0).astype(int)
    df["card4_is_mastercard"] = (df["card4"] == "mastercard").astype(int)
    df["card4_is_visa"] = (df["card4"] == "visa").astype(int)
    df["card5_is_102"] = (df["card5"] == 102.0).astype(int)
    df["card5_is_137"] = (df["card5"] == 137.0).astype(int)
    df["card5_is_138"] = (df["card5"] == 138.0).astype(int)
    df["card6_is_credit"] = (df["card6"] == "credit").astype(int)
    df["card6_is_debit"] = (df["card6"] == "debit").astype(int)
    # ProductCD high risk features binry flag
    df["ProductCD_is_C"] = (df["ProductCD"] == "C").astype(int)
    df["ProductCD_is_W"] = (df["ProductCD"] == "W").astype(int)
    df["ProductCD_is_H"] = (df["ProductCD"] == "H").astype(int)
    df["ProductCD_is_R"] = (df["ProductCD"] == "R").astype(int)

    # UID
    df['D1n'] = np.floor(df['TransactionDT'] / (24*60*60)) - df['D1']
    df['UID'] = df['card1'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['P_emaildomain'].astype(str)+'_'+df['D1n'].astype(str)

    # Group statistics
    df['D1_UID_std']  = df.groupby('UID')['D1'].transform('std')
    df['D6_UID_std']  = df.groupby('UID')['D6'].transform('std')
    df['D11_UID_mean'] = df.groupby('UID')['D11'].transform('mean')
    df['D11_UID_std']  = df.groupby('UID')['D11'].transform('std')
    df['D12_UID_std']  = df.groupby('UID')['D12'].transform('std')
    df['D14_UID_std']  = df.groupby('UID')['D14'].transform('std')
    df['D15_UID_std']  = df.groupby('UID')['D15'].transform('std')
    df['D2_UID_std']  = df.groupby('UID')['D2'].transform('std')
    df['D2_UID_mean']  = df.groupby('UID')['D2'].transform('mean')
    df['D5_UID_std']  = df.groupby('UID')['D5'].transform('std')
    df['D8_UID_std']  = df.groupby('UID')['D8'].transform('std')
    df['D9_UID_std']  = df.groupby('UID')['D9'].transform('std')
    df['D10_UID_std']  = df.groupby('UID')['D10'].transform('std')

    df['C1_UID_mean'] = df.groupby('UID')['C1'].transform('mean')
    df['C2_UID_std'] = df.groupby('UID')['C2'].transform('std')
    df['C3_UID_mean'] = df.groupby('UID')['C3'].transform('mean')
    df['C3_UID_std'] = df.groupby('UID')['C3'].transform('std')
    df['C6_UID_mean'] = df.groupby('UID')['C6'].transform('mean')
    df['C6_UID_std'] = df.groupby('UID')['C6'].transform('std')
    df['C7_UID_mean'] = df.groupby('UID')['C7'].transform('mean')
    df['C13_UID_mean'] = df.groupby('UID')['C13'].transform('mean')
    df['C13_UID_std'] = df.groupby('UID')['C13'].transform('std')
    df['C5_UID_std'] = df.groupby('UID')['C5'].transform('std')
    df['C11_UID_std'] = df.groupby('UID')['C11'].transform('std')
    df['C14_UID_std'] = df.groupby('UID')['C14'].transform('std')
    df['M1_UID_ct'] = df.groupby('UID')['M1'].transform('count')
    df['M1_UID_ctt'] = df.groupby(['UID', 'M1'])['M1'].transform('count')
    df['M2_UID_ct'] = df.groupby('UID')['M2'].transform('count')
    df['M3_UID_ct'] = df.groupby('UID')['M3'].transform('count')
    df['M4_UID_ct'] = df.groupby(['UID', 'M4'])['M4'].transform('count')
    df['M7_UID_ct'] = df.groupby(['UID', 'M7'])['M7'].transform('count')
    df['M9_UID_ct'] = df.groupby(['UID', 'M9'])['M9'].transform('count')
    df['P_emaildomain_UID_ct'] = df.groupby(['UID', 'P_emaildomain'])['P_emaildomain'].transform('count')
    df['M5_UID_ct'] = df.groupby(['UID', 'M5'])['M5'].transform('count')
    df['M2_UID_ctt'] = df.groupby(['UID', 'M2'])['M2'].transform('count')
    df['M6_UID_ctt'] = df.groupby(['UID', 'M6'])['M6'].transform('count')
    return df

