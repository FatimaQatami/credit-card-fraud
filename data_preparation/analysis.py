import pandas as pd
from pathlib import Path 
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
import shap
from sklearn.metrics import roc_curve, average_precision_score, precision_recall_curve


project_root = Path(__file__).resolve().parent.parent
train = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")


# Export missing value percentage ordered descending 
missing = train.isna().mean().sort_values(ascending=False) * 100
missing.to_csv(project_root / "graphs" / "missing_values_per.csv")


# Export near-constant features
thresh = 0.98
results = []
for col in train.columns:
    # missing values percentage
    miss_prc = train[col].isna().mean()
    # near-constant features ignoring NaNs
    counts = train[col].dropna().value_counts(normalize=True).iloc[0]
    # Store results 
    results.append([col, counts, miss_prc])
# Save results 
df_results = pd.DataFrame(results, columns=['feature', 'dominance_percent', 'missing_percent'])
df_results.to_csv(project_root / "graphs" / "near_constant.csv")



# Correlations: Pearson and Spearman (numerical features)
num = train.select_dtypes(include=['number'])
pearson = num.corr(method='pearson')
pearson_no_dup = pearson.where(np.triu(np.ones(pearson.shape), k=1).astype(bool)).stack().reset_index()
pearson_no_dup.columns = ['feature1', 'feature2', 'pearson']
spearman = num.corr(method='spearman')
spearman_no_dup = spearman.where(np.triu(np.ones(spearman.shape), k=1).astype(bool)).stack().reset_index()
spearman_no_dup.columns = ['feature1', 'feature2', 'spearman']
correlations = pearson_no_dup.merge(spearman_no_dup, on=['feature1', 'feature2'])
high_corr = correlations[correlations['spearman'].abs() > 0.95]
high_corr.to_csv(project_root / "stats" / " corr_fe.csv", index=False)



# Mutual Info for (numerical features)
num_cols = train.select_dtypes(include='number').columns 
train[num_cols] = train[num_cols].fillna(-9999)
X = train[num_cols].drop(columns=['isFraud'])
y = train['isFraud']
mi = mutual_info_classif(X, y)
mi_df = pd.DataFrame({'feature': X.columns, 'MI': mi})
mi_df.to_csv(project_root / "stats" / "mi_fe.csv")



# model comparison bar plots
models = ["CatBoost", "XGBoost", "LightGBM"]
methods = ["CSL", "SMOTE", "ROS", "RUS", "None"]

prauc = {
    "CatBoost":  [0.7564, 0.6829, 0.6638, 0.6714, 0.7427],
    "XGBoost":   [0.6701, 0.5898, 0.6631, 0.6009, 0.6227],
    "LightGBM":  [0.6773, 0.6791, 0.6630, 0.6263, 0.6798],
}

colors = [
    "#E8A6B8",  # pastel pink
    "#A7C7E7",  # pastel blue
    "#B7E4C7",  # pastel green
    "#CDB4DB",  # pastel purple
    "#FFD6A5",  # pastel orange
]

x = np.arange(len(models))
w = 0.16

plt.figure(figsize=(9, 4.8))
for i, method in enumerate(methods):
    plt.bar(
        x + (i - 2) * w,
        [prauc[m][i] for m in models],
        width=w,
        label=method,
        color=colors[i]
    )

plt.xticks(x, models)
plt.ylabel("PR-AUC")
plt.legend(ncols=3, frameon=False)
plt.tight_layout()
plt.show()




# SHAP impportance and beeswarm for best model
X_explain = train.sample(n=min(5000, len(train)), random_state=42)

explainer = shap.TreeExplainer()
shap_values = explainer.shap_values(X_explain)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Beeswarm
shap.summary_plot(shap_values, X_explain, show=False)
plt.savefig(project_root / "shap_beeswarm.png", dpi=300, bbox_inches="tight")
plt.close()

# Global importance
shap.summary_plot(shap_values, X_explain, plot_type="bar", max_display=20, show=False)
plt.savefig(project_root / "shap_importance.png", dpi=300, bbox_inches="tight")
plt.close()



# ROC curve
y_true = np.concatenate(all_y_true)
y_score = np.concatenate(all_y_score)

fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], "--")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.show()

# PR curve
precision, recall, _ = precision_recall_curve(y_true, y_score)
pr_auc = average_precision_score(y_true, y_score)

plt.figure()
plt.plot(recall, precision, label=f"PR (AP = {pr_auc:.4f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision–Recall Curve")
plt.legend()
plt.tight_layout()
plt.show()

