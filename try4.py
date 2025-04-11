import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

# Load data
train_df = pd.read_csv("home_data/train.csv")
test_df = pd.read_csv("home_data/test.csv")
print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)

# Save IDs for submission
test_ids = test_df['Id']

# Prepare target
y = np.log1p(train_df['SalePrice'])
train_df.drop(['SalePrice', 'Id'], axis=1, inplace=True)
test_df.drop(['Id'], axis=1, inplace=True)

# Combine train and test for consistent preprocessing
combined = pd.concat([train_df, test_df], axis=0, sort=False)

# ----------------------
# Feature Engineering
# ----------------------

# Create new features
combined['TotalSF'] = combined['TotalBsmtSF'] + combined['1stFlrSF'] + combined['2ndFlrSF']
combined['TotalBath'] = (combined['FullBath'] + 
                         (0.5 * combined['HalfBath']) + 
                         combined['BsmtFullBath'] + 
                         (0.5 * combined['BsmtHalfBath']))
combined['TotalPorch'] = (combined['OpenPorchSF'] + combined['EnclosedPorch'] + 
                         combined['3SsnPorch'] + combined['ScreenPorch'])
combined['Age'] = combined['YrSold'] - combined['YearBuilt']
combined['RemodAge'] = combined['YrSold'] - combined['YearRemodAdd']
combined['IsRemod'] = (combined['YearBuilt'] != combined['YearRemodAdd']).astype(int)
combined['IsNew'] = (combined['YrSold'] == combined['YearBuilt']).astype(int)

# Drop columns that are redundant or not useful
cols_to_drop = ['Utilities', 'Street', 'PoolQC', 'MiscFeature', 'Alley']
combined.drop(cols_to_drop, axis=1, inplace=True, errors='ignore')

# ----------------------
# Handle Missing Values
# ----------------------

# Fill numerical missing values
num_cols = combined.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    if combined[col].isnull().sum() > 0:
        if 'SF' in col or 'Area' in col or 'Porch' in col:
            combined[col].fillna(0, inplace=True)  # Assume no area if missing
        elif 'Yr' in col or 'Year' in col:
            combined[col].fillna(combined[col].median(), inplace=True)
        else:
            combined[col].fillna(combined[col].median(), inplace=True)

# Fill categorical missing values
cat_cols = combined.select_dtypes(include=['object']).columns
for col in cat_cols:
    combined[col].fillna('None', inplace=True)

# ----------------------
# Encode Categorical Variables
# ----------------------

# Label encode ordinal variables (where order matters)
ordinal_mapping = {
    'ExterQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'ExterCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'BsmtCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'HeatingQC': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'KitchenQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'FireplaceQu': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageQual': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'GarageCond': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'PoolQC': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
    'LotShape': {'None': 0, 'IR3': 1, 'IR2': 2, 'IR1': 3, 'Reg': 4},
    'LandSlope': {'None': 0, 'Sev': 1, 'Mod': 2, 'Gtl': 3},
    'BsmtExposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
    'BsmtFinType1': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'BsmtFinType2': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
    'Functional': {'None': 0, 'Sal': 1, 'Sev': 2, 'Maj2': 3, 'Maj1': 4, 'Mod': 5, 'Min2': 6, 'Min1': 7, 'Typ': 8},
    'GarageFinish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
    'PavedDrive': {'None': 0, 'N': 1, 'P': 2, 'Y': 3},
    'Fence': {'None': 0, 'MnWw': 1, 'GdWo': 2, 'MnPrv': 3, 'GdPrv': 4}
}

for col, mapping in ordinal_mapping.items():
    if col in combined.columns:
        combined[col] = combined[col].map(mapping).fillna(0).astype(int)

# One-hot encode remaining categorical variables
combined = pd.get_dummies(combined)

# ----------------------
# Split back into train and test
# ----------------------
X_train = combined.iloc[:len(y), :]
X_test = combined.iloc[len(y):, :]

# ----------------------
# Feature Selection
# ----------------------
selector = SelectFromModel(RandomForestRegressor(n_estimators=100, random_state=42), threshold="1.25*median")
selector.fit(X_train, y)
selected_features = X_train.columns[selector.get_support()]
X_train = X_train[selected_features]
X_test = X_test[selected_features]

# ----------------------
# Model Training with Cross-Validation
# ----------------------
def rmse_cv(model, X, y, cv=5):
    kf = KFold(cv, shuffle=True, random_state=42)
    rmse = np.sqrt(-cross_val_score(model, X, y, scoring="neg_mean_squared_error", cv=kf))
    return rmse.mean()

# Try multiple models
models = {
    'RandomForest': RandomForestRegressor(random_state=42, n_estimators=500, max_depth=15),
    'XGBoost': xgb.XGBRegressor(random_state=42, n_estimators=1000, learning_rate=0.01),
    'LightGBM': lgb.LGBMRegressor(random_state=42, n_estimators=1000, learning_rate=0.01)
}

for name, model in models.items():
    score = rmse_cv(model, X_train, y)
    print(f"{name} CV RMSE: {score:.5f}")

# Ensemble predictions
final_predictions = []
for model in models.values():
    model.fit(X_train, y)
    preds = model.predict(X_test)
    final_predictions.append(preds)

# Average predictions
final_pred = np.mean(final_predictions, axis=0)
final_pred = np.expm1(final_pred)

# ----------------------
# Create Submission
# ----------------------
submission = pd.DataFrame({
    "Id": test_ids,
    "SalePrice": final_pred
})

submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv")