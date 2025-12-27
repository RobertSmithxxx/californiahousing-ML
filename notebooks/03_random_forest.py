from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import RandomForestRegressor

# Load data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Random Forest baseline
rf = RandomForestRegressor(
    n_estimators=500,
    random_state=42,
    n_jobs=-1
)

scores = cross_val_score(rf, X, y, cv=cv, scoring="r2", n_jobs=-1)
print("RandomForest:", scores.mean(), "±", scores.std())
