import numpy as np
from scipy.stats import friedmanchisquare
from scipy.stats import wilcoxon
import pandas as pd
from pathlib import Path 


"""
xgb_none = np.array([0.4982, 0.6117, 0.6724, 0.6638, 0.6672])
cat_none= np.array([0.6151, 0.7648, 0.7953, 0.7846, 0.7978])
lgb_none = np.array([0.5387, 0.6902, 0.7263, 0.7172, 0.7267])

xgb_csl = np.array([0.5391, 0.6921, 0.6914, 0.7202, 0.7076])
cat_csl = np.array([0.6399, 0.7702, 0.7917, 0.7869, 0.7931])
lgb_csl = np.array([0.5353, 0.6923, 0.7222, 0.7123, 0.7244])

xgb_smote = np.array([0.4372, 0.5973, 0.6411, 0.6470, 0.6266])
cat_smote = np.array([0.5818, 0.6596, 0.7476, 0.7266, 0.7248])
lgb_smote = np.array([0.5141, 0.6824, 0.7428, 0.7319, 0.7242])

xgb_ros = np.array([0.5371, 0.6727, 0.7040, 0.7009, 0.7006])
cat_ros = np.array([0.5310, 0.6767, 0.6982, 0.7012, 0.7025])
lgb_ros = np.array([0.5346, 0.6683, 0.7061, 0.6995, 0.7066])

xgb_rus = np.array([0.4693, 0.6181, 0.6498, 0.6597, 0.6077])
cat_rus = np.array([0.5212, 0.6936, 0.7058, 0.7239, 0.7109])
lgb_rus = np.array([0.4816, 0.6331, 0.6820, 0.6888, 0.6457])

stat, p = friedmanchisquare(
    xgb_none, cat_none, lgb_none,
    xgb_csl,  cat_csl,  lgb_csl,
    xgb_smote,cat_smote,lgb_smote,
    xgb_ros,  cat_ros,  lgb_ros,
    xgb_rus,  cat_rus,  lgb_rus
)

print("Friedman chi2:", stat)
print("p-value:", p)


# Friedman chi2: 60.01999999999998
# p-value: 1.1637574055375151e-07


best =  np.array([0.6399, 0.7702, 0.7917, 0.7869, 0.7931])
other = np.array([0.4816, 0.6331, 0.6820, 0.6888, 0.6457])

stat, p = wilcoxon(best, other)

# effect size r
z = (stat - (len(best)*(len(best)+1)/4)) / np.sqrt(len(best)*(len(best)+1)*(2*len(best)+1)/24)
r = abs(z) / np.sqrt(len(best))

print("p-value:", p)
print("effect size r:", r)


# cat_csl: xgb_rus, lgb_rus, cat_rus, lgb_ros, cat_ros, xgb_ros, lgb_smote, cat_smote, xgb_smote, lgb_csl, xgb_csl, lgb_none, cat_none, xgb_none
# p-value: 0.0625
# effect size r: 0.9045340337332908

"""

import pingouin as pg


project_root = Path(__file__).resolve().parent.parent
X = pd.read_parquet(project_root / "data" / "processed" / "train.parquet")


import scipy
print(scipy.__version__)


import pingouin as pg
pg.mcar(X)   # X = pandas DataFrame with NaNs
