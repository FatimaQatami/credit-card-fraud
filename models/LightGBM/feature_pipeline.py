import re
import numpy as np

def apply_feature_engineering_selection(df):
        
    # Email extracted features
    df['P_emaildomain'] = df['P_emaildomain'].str.lower().astype('category')
    df['R_emaildomain'] = df['R_emaildomain'].str.lower().astype('category')
    df['suffix_r'] = df['R_emaildomain'].str.split(pat='.', n=1).str[1].astype('category')
    df['tld_r'] = df['R_emaildomain'].str.split('.').str[-1].astype('category')

    # Mobile and browser keywords 
    device_mobile = ["sm-", "gt-", "ale-", "cam-", "trt-", "was-", "mya-", "rne-", "cro-", 
                    "bll-", "chc-", "pra", "android", "build/", "huawei", "honor", "hisense", 
                    "zte", "htc", "moto", "xt", "samsung", "mi ", "redmi", "pixel", "nexus", "kf", 
                    "lg", "iphone", "ios"]
    device_browser = ["windows", "trident", "rv:", "macos", "mac",
                    "linux"]

    # Create mobile and browser binary features 
    df['DeviceInfo'] = df['DeviceInfo'].str.lower().astype('category')
    df['is_mobile'] = df['DeviceInfo'].str.contains('|'.join(map(re.escape, device_mobile)),
                                                        regex=True, na=None).astype(float)
    df['is_browser'] = df['DeviceInfo'].str.contains('|'.join(map(re.escape, device_browser)),
                                                        regex=True, na=None).astype(float)


    # TransactionAMT split features
    df['TransactionAmt_dec'] = df['TransactionAmt'].astype(str).str.split('.', expand=True)[1].astype(float)

    #  Natural logarithm of transaction amount
    df['amount_log'] = np.log1p(df['TransactionAmt'])
    df['amount_log10'] = np.log10(df['TransactionAmt'] + 1)
    df['amount_sqrt'] = np.sqrt(df['TransactionAmt'])

    # Duration features
    df['hours_duration'] = df['TransactionDT'] / (60 * 60)
    df['days_duration'] = df['TransactionDT'] / (60 * 60 * 24)

    # Cyclical calendar features 
    # Minute (0–59)
    df['minute'] = (df['TransactionDT'] // 60) % 60
    # Hour (0–23)
    df['hour'] = (df['TransactionDT'] // 3600) % 24
    # Weekday (0–6)
    df['weekday'] = (df['TransactionDT'] // 86400) %  7


    # Split features
    col_split = ['id_34']
    for col in col_split:
        parts = df[col].astype(str).str.split(r'[ /_]', n=1, expand=True)
        parts = parts.reindex(columns=[0, 1]) 
        df[f"{col}_part1"] = parts[0].astype(str).astype('category')
        df[f"{col}_part2"] = parts[1].astype(str).astype('category')


    # Missing-indicators features
    cols = ['id_33', 'card2', 'card3']
    for col in cols:
        df[col + '_is_missing'] = df[col].isna().astype(int)


    # Missing-count and missing-ratio features (per group)
    group = {
        "M": [f"M{i}" for i in range(1,10)]
    }
    for name, col in group.items():
        col = [c for c in col if c in df.columns] 
        df[f'{name}_missing_count'] = df[col].isna().sum(axis=1)
        df[f'{name}_missing_ratio'] = df[f'{name}_missing_count'] / len(col)


    # Interaction features (try more)
    df['id_28_combo'] = (df['id_29'].astype(str) + '_' + df['id_28'].astype(str)).astype('category')

    # ProductCD high risk features binry flag
    df["ProductCD_is_W"] = (df["ProductCD"] == "W").astype(int)

    # UID
    df['D1n'] = np.floor(df['TransactionDT'] / (24*60*60)) - df['D1']
    df['UID'] = (df['card1'].astype(str)+'_'+df['addr1'].astype(str)+'_'+df['P_emaildomain'].astype(str)+'_'+df['D1n'].astype(str)).astype('category')


    # Group statistics (use cat feats with count both types and num feats with std/mean)
    df['D1_UID_std']  = df.groupby('UID')['D1'].transform('std')
    df['D6_UID_std']  = df.groupby('UID')['D6'].transform('std')
    df['D11_UID_mean'] = df.groupby('UID')['D11'].transform('mean')
    df['D11_UID_std']  = df.groupby('UID')['D11'].transform('std')
    df['D12_UID_std']  = df.groupby('UID')['D12'].transform('std')
    df['D14_UID_std']  = df.groupby('UID')['D14'].transform('std')
    df['D15_UID_std']  = df.groupby('UID')['D15'].transform('std')

    df['C1_UID_mean'] = df.groupby('UID')['C1'].transform('mean')
    df['C2_UID_std'] = df.groupby('UID')['C2'].transform('std')
    df['C3_UID_mean'] = df.groupby('UID')['C3'].transform('mean')
    df['C3_UID_std'] = df.groupby('UID')['C3'].transform('std')
    df['C6_UID_mean'] = df.groupby('UID')['C6'].transform('mean')
    df['C6_UID_std'] = df.groupby('UID')['C6'].transform('std')
    df['C7_UID_mean'] = df.groupby('UID')['C7'].transform('mean')
    df['C13_UID_mean'] = df.groupby('UID')['C13'].transform('mean')
    df['C13_UID_std'] = df.groupby('UID')['C13'].transform('std')

    df['M1_UID_ct'] = df.groupby('UID')['M1'].transform('count')
    df['M1_UID_ctt'] = df.groupby(['UID', 'M1'])['M1'].transform('count')
    df['M2_UID_ct'] = df.groupby('UID')['M2'].transform('count')
    df['M4_UID_ct'] = df.groupby(['UID', 'M4'])['M4'].transform('count')
    df['M7_UID_ct'] = df.groupby(['UID', 'M7'])['M7'].transform('count')
    df['M9_UID_ct'] = df.groupby(['UID', 'M9'])['M9'].transform('count')
    df['P_emaildomain_UID_ct'] = df.groupby(['UID', 'P_emaildomain'])['P_emaildomain'].transform('count')

    # Frequency encoding 
    df['card5_frq'] = df['card5'].map(df['card5'].value_counts())
    df['ProductCD_frq'] = df['ProductCD'].map(df['ProductCD'].value_counts())
    df['DeviceType_frq'] = df['DeviceType'].map(df['DeviceType'].value_counts())
    df['id_13_frq'] = df['id_13'].map(df['id_13'].value_counts())
    df['id_15_frq'] = df['id_15'].map(df['id_15'].value_counts())
    df['id_19_frq'] = df['id_19'].map(df['id_19'].value_counts())
    return df

