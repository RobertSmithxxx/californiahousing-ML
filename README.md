My project explores multiple regression models for predicting house prices on the California Housing dataset, with a focus on proper evaluation and model comparison

Dataset
Source : sklearn.datasets.fetch_california_housing
Samples : 20,000
Features: 8 numerical features (income, house age, rooms, population, latitude, longitude)
Target: Median house value

Models Evaluated
Linear Regression
Ridge Regression
Decision Tree (various depths)
Random Forest
HistGradientBoostingRegressor

Linear models show limited performance (R² around 0.6), indicating underfitting due to the non-linear nature of the data
Decision Trees capture non-linearity but overfit when too deep( the best r2 : 0.6954 ± 0.0136)
Random Forest provides a strong non-linear baseline(r2 : 0.8121112666848171 +- 0.007939648511709044)

Gradient Boosting with log-transformed target performed best(r2 : 0.8539249009870822 +- 0.005519072838775356)


Some Notes
Results reflect within-distribution performance, which is the standard assumption in most real-world ML applications.
Due to strong spatial patterns (latitude and longitude), random splits can yield higher.
The final reported score is stable and reproducible.
