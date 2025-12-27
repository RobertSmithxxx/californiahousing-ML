import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

# Load data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# Reproducible CV (within-distribution)
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Linear Regression (scaling helps linear models)
lin = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
scores = cross_val_score(lin, X, y, cv=cv, scoring="r2", n_jobs=-1)
print("LinearRegression:", scores.mean(), "±", scores.std())

# L2 = linear + regularization (helps if linear is overfitting)
ridge = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge(alpha=1.0))
])
scores = cross_val_score(ridge, X, y, cv=cv, scoring="r2", n_jobs=-1)
print("Ridge(alpha=1.0):", scores.mean(), "±", scores.std())
