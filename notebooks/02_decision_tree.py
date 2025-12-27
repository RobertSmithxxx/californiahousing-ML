from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import KFold, cross_val_score
from sklearn.tree import DecisionTreeRegressor

# 1) Load data
data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target

# 2) CV
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# 3) Try multiple depths to see among underfit -> best -> overfit
depths = [3, 5, 9, 12, 15, 20, None]

for i in depths:
    tree = DecisionTreeRegressor(max_depth=i, random_state=42)
    scores = cross_val_score(tree, X, y, cv=cv, scoring="r2", n_jobs=-1)
    print(f"DecisionTree(depth={i}): {scores.mean():.4f} ± {scores.std():.4f}")
