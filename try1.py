import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer

train_df = pd.read_csv("home_data/train.csv")
test_df = pd.read_csv("home_data/test.csv")

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

sns.histplot(train_df['SalePrice'], kde=True)
plt.title("Distribution of Sale Prices")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
#plt.show()

y = train_df['SalePrice']
train_df.drop(['SalePrice'], axis=1, inplace=True)

combined = pd.concat([train_df, test_df], axis=0, sort=False)

# We use one-hot encoding so that the model can understand the text features
combined = pd.get_dummies(combined)

X_train = combined.iloc[:len(y), :]
X_test = combined.iloc[len(y):, :]

model = RandomForestRegressor(n_estimators=100, random_state=42)

scores = -cross_val_score(model, X_train, y, scoring="neg_root_mean_squared_error", cv=5)
print("Cross-validated RMSE:", scores.mean())

model.fit(X_train, y)
preds = model.predict(X_test)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": preds
})

submission.to_csv("try1.csv", index=False)
print("Submission file created: submission.csv")
