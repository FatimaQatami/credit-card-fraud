from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, matthews_corrcoef)
from imblearn.metrics import geometric_mean_score
from xgboost import XGBClassifier
from xgboost.callback import EarlyStopping
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import time


# Load file
project_root = Path(__file__).resolve().parent.parent
train = pd.read_parquet(project_root / "data" / "raw" / "paysim.parquet")


# sort values to keep timely order 
train = train.sort_values(['step']).reset_index(drop=True)

# Correct data types 
cat_cols = ['type','nameOrig','nameDest']
train[cat_cols] = train[cat_cols].astype('category')




# model training 
X = train.drop(columns=['isFraud'])
y = train['isFraud']

fold_results = []

train['time_group'] = train['step'] // 50 
groups = train['time_group']
gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]


    model = XGBClassifier(
        random_state=42,
        eval_metric="aucpr",
        tree_method = 'hist',
        enable_categorical = True,
        #scale_pos_weight=12,
        callbacks=[EarlyStopping(rounds=100, save_best=True, maximize=True)],        
        )
    
    start_time = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        )
    end_time = time.time()
    train_time = end_time - start_time
    
    y_prob = model.predict_proba(X_val)[:, 1]
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


    fold_results.append({'roc_auc': roc, 'pr_auc': prauc, 'accuracy':acc, 'precision': prec,
                         'recall': rec, 'f1': f1, 'f2': f2, 'gmean': gmean, 'mcc': mcc,
                         'fnr': fnr, 'fpr': fpr, 'confusion matrix': cm, 'train_time': train_time})

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
avg_time = np.mean([r['train_time'] for r in fold_results])


fold_results.append({'roc_auc': avg_roc, 'pr_auc': avg_prauc,  'accuracy': avg_acc,
                     'precision': avg_prec, 'recall': avg_rec, 'f1': avg_f1, 'f2': avg_f2,
                     'gmean': avg_gmean, 'mcc': avg_mcc, 'fnr': avg_fnr, 'fpr': avg_fpr, 'train_time': avg_time})


results = pd.DataFrame(fold_results)
results.to_csv(project_root / "paysim" / "xgb_baseline_paysim.csv")
