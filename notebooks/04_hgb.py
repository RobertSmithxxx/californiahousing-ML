import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor

# 1) Load data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Logaithm-transform target
y_log = np.log1p(y)

# CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# HGB
hgb = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    random_state=42
)

scores = cross_val_score(hgb, X, y_log, cv=cv, scoring="r2", n_jobs=-1)
print("HGB :", scores.mean(), "±", scores.std())
