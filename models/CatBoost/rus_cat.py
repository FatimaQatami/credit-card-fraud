from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, matthews_corrcoef)
from imblearn.metrics import geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler



# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline_catboost import apply_feature_engineering_selection


# Load dataset
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

# Correct data types 
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']
train[cat_cols] = train[cat_cols].astype(str)


# Feature pipeline 
if use_feature_pipeline:
    train = apply_feature_engineering_selection(train)


# Features to drop based on feature selection tests
feature_list = [
    "id_18","id_05","V259","D14","id_02","V220","V83","id_04","V134","D1","V275","V132","V53","V283",
    "id_30","V52","V127","V221","V3","V309","V311","V72","V105","M3_UID_ct","V273","V128","V266","V119",
    "V278","V300","D6_UID_std","device_os","C2_UID_std","id_missing_count","V99","V5","id_38","V246",
    "V198","V203","ProductCD_is_H","V222","V207","V315","id_17","V87","V80","V30","V7","V126","V277","V157",
    "D7_adj","V205","V228","C5_UID_std","V12","V301","D7","V296","V176","V250","V242","V224","V40","V103","V85",
    "V234","V36","V261","id_32","V320","V265","is_browser","V178","V264","id_34","V101","V120","V190","M1_UID_ct",
    "V170","V66","V124","V240","V202","id_12","V15","V84","V298","V24","V321","V162","V18","V138","V284","V46","V215",
    "V114","id_11","V57","V173","V323","V16","id_10","V65","M8","V10","V71","V192","V168","V263","V286","V160","V249",
    "V22","V161","V74","V145","V167","V262","V180","V310","V147","V253","M1_UID_ctt","V191","V158","V69","V113","V31",
    "V169","V327","V41","id_20_is_507","V333","V334","id_18_is_15","V106","V107","id_19_is_266","D11","V335","V336","V11",
    "id_20_is_325","V110","V111","V337","V112","V338","V339","id_33_is_missing","V118","V117","V42","V64","card2_is_321",
    "id_17_is_225","card5_is_102","V96","V98","ProductCD_is_C","card5_is_138","card5_is_137","id_17_is_166","card3_is_185",
    "card2_is_545","V95","V94","V1","id_14_high_risk","id_13_is_52","V68","V73","id_13_is_33","id_13_high_risk","V8","V100",
    "V88","V89","V90","id_13_is_49","V166","V330","V302","V181","V183","V299","V297","V184","V185","V14","V144","V285","V193",
    "V238","V28","V279","V197","V199","V223","V21","V27","V269","V268","V155","V255","V211","V141","V140","V227","V225","V122",
    "V243","V328","V325","V324","V322","V241","V32","V237","V121","V235","V212","V305","V172"
    ]

if use_feature_pipeline:
    train = train.drop(columns=feature_list, errors='ignore')

# Update categorical features 
cat_cols = train.select_dtypes(include=["object","category"]).columns.tolist()
train[cat_cols] = train[cat_cols].astype(str)  


# Model training 
X = train.drop(columns=['isFraud'])
y = train['isFraud']

fold_results = []

train['month'] = train['TransactionDT'] // (30 * 24 * 60 * 60)
groups = train['month']

gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    rus = RandomUnderSampler(random_state=42)
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    train_pool = Pool(X_train_res, y_train_res, cat_features=cat_cols)
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
    roc = roc_auc_score(y_val, y_prob)
    prauc = average_precision_score(y_val, y_prob)

    y_pred = (y_prob >= 0.5).astype(int)  # default threshold
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


    fold_results.append({'roc_auc': roc, 'pr_auc': prauc, 'accuracy':acc, 'precision': prec,
                         'recall': rec, 'f1': f1, 'f2': f2, 'gmean': gmean, 'mcc': mcc,
                         'fnr': fnr, 'fpr': fpr, 'confusion matrix': cm})


results = pd.DataFrame(fold_results)
results.to_csv(project_root / "cat_rus.csv")