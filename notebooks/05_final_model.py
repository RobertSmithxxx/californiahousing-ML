from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.datasets import fetch_california_housing
import numpy as np

data = fetch_california_housing(as_frame=True)
X = data.data
y = data.target
y_log = np.log1p(y)


model = HistGradientBoostingRegressor(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    random_state=42
)


cv = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y_log, cv=cv, scoring="r2", n_jobs=-1)

print("Final CV R2:", scores.mean(), "±", scores.std())


X_train, X_test, y_train, y_test = train_test_split(
    X, y_log, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Final Test R2:", r2_score(y_test, y_pred))
