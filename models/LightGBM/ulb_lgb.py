from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
import lightgbm as lgb
from sklearn.metrics import (average_precision_score, roc_auc_score, accuracy_score, 
    precision_score, recall_score, f1_score, fbeta_score, confusion_matrix, matthews_corrcoef)
from imblearn.metrics import geometric_mean_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE
import time

 # Load file
project_root = Path(__file__).resolve().parent.parent.parent
train = pd.read_csv(project_root / "data" / "raw" / "ulb.csv")

# Sort values to keep timely order 
train = train.sort_values(['Time']).reset_index(drop=True)

# Model training 
X = train.drop(columns=['Class'])
y = train['Class']

fold_results = []

train['time_group'] = train['Time'] // 3600  
groups = train['time_group']
gkf = GroupKFold(n_splits=5)

for train_idx, val_idx in gkf.split(X, y, groups=groups):
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = LGBMClassifier(
        random_state=42,
        )
    
    
    start_time = time.time()

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="average_precision",
        callbacks=[lgb.early_stopping(stopping_rounds=100)]
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
results.to_csv(project_root / "ulb" / "lgb_baseline_ulb.csv")
