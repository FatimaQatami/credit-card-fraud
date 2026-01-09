from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, matthews_corrcoef)
from imblearn.metrics import geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler

# Including feature pipeline on/off
use_feature_pipeline = True  # False = baseline
if use_feature_pipeline:
    from feature_pipeline import apply_feature_engineering_selection


# Load train file
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

# Correct data types and fill NaNs
cat_cols = ['ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 
                    'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain', 'M1', 'M2', 'M3', 
                    'M4', 'M5', 'M6', 'M7', 'M8', 'M9', 'id_12', 'id_13', 'id_14', 'id_15', 
                    'id_16', 'id_17', 'id_18', 'id_19', 'id_20', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_32', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 
                    'DeviceType', 'DeviceInfo']
train[cat_cols] = train[cat_cols].astype('category')



# Feature pipeline 
if use_feature_pipeline:
    train = apply_feature_engineering_selection(train)


# Features to drop (based on feature selection)
permutation = [
    "V139","id_37","V3","V137","V206","V93","V234","V42","V212",
    "suffix_r","V37","V164","V273","V51","V63","V223","is_mobile"]
train = train.drop(columns=permutation)

tree_shap = ['V339', 'V172']
train = train.drop(columns=tree_shap)

mi = ['V287', 'V99', 'V284']
train = train.drop(columns=mi)


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


    model = XGBClassifier(
        enable_categorical=True,
        tree_method='hist',
        random_state=42,
        eval_metric="aucpr",
        callbacks=[EarlyStopping(rounds=100, save_best=True, maximize=True)],        
        )
    
    model.fit(
        X_train_res, y_train_res,
        eval_set=[(X_val, y_val)],
        )
    
    y_prob = model.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, y_prob)
    prauc = average_precision_score(y_val, y_prob)

    y_pred = (y_prob >= 0.5).astype(int) # default threshold
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

avg_roc = np.mean([r['roc_auc'] for r in fold_results])
avg_prauc = np.mean([r['pr_auc'] for r in fold_results])
avg_acc = np.mean([r['accuracy'] for r in fold_results])
avg_prec = np.mean([r['precision'] for r in fold_results])
avg_rec = np.mean([r['recall']for r in fold_results])
avg_f1 = np.mean([r['f1'] for r in fold_results])
avg_f2 = np.mean([r['f2'] for r in fold_results])
avg_gmean = np.mean([r['gmean'] for r in fold_results])
avg_mcc = np.mean([r['mcc'] for r in fold_results])
avg_fnr = np.mean([r['fnr'] for r in fold_results])
avg_fpr = np.mean([r['fpr'] for r in fold_results])


fold_results.append({'roc_auc': avg_roc, 'pr_auc': avg_prauc,  'accuracy': avg_acc,
                     'precision': avg_prec, 'recall': avg_rec, 'f1': avg_f1, 'f2': avg_f2,
                     'gmean': avg_gmean, 'mcc': avg_mcc, 'fnr': avg_fnr, 'fpr': avg_fpr})


results = pd.DataFrame(fold_results)
results.to_csv(project_root / "ieee-cis" / "baseline" / "xgb_rus.csv")
