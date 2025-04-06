import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error

train_df = pd.read_csv("home_data/train.csv")
test_df = pd.read_csv("home_data/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

y = np.log1p(train_df['SalePrice'])
train_df.drop(['SalePrice'], axis=1, inplace=True)

combined = pd.concat([train_df, test_df], axis=0, sort=False)

combined = pd.get_dummies(combined)

X_train = combined.iloc[:len(y), :]
X_test = combined.iloc[len(y):, :]

model = RandomForestRegressor(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

grid_search = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=5,
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X_train, y)
print("Best Parameters:", grid_search.best_params_)
print("Best CV RMSE:", -grid_search.best_score_)

best_model = grid_search.best_estimator_
preds = np.expm1(best_model.predict(X_test))

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": preds
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")

