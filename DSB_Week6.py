import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", 101)

data = pd.read_csv('train.csv')

#Question 1 and 2

X_test[num_cols] = scaler.transform(X_test[num_cols])
y_pred = reg.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)
X_test.describe()

#Question 3

ridge = Ridge(alpha=1)
ridge.fit(X_train,y_train)
y_pred = ridge.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test), label='true')
plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred), label = 'pred')
plt.legend()

ridge.coef_

lasso = Lasso(alpha=1)
lasso.fit(X_train,y_train)
y_pred = lasso.predict(X_test)
print(mean_absolute_error(y_test,y_pred), mean_squared_error(y_test,y_pred)**0.5)

plt.scatter(np.arange(len(np.sort(y_test))),np.sort(y_test))
plt.scatter(np.arange(len(np.sort(y_pred))),np.sort(y_pred))

lasso.coef_

#Quesion 4

data.fillna(data.median(), inplace=True)

for column in data.select_dtypes(include=['object']).columns:
    data[column] = LabelEncoder().fit_transform(data[column])

X = data.drop('salary', axis=1)
y = data['salary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_model = DecisionTreeRegressor(random_state=42)

decision_tree_model.fit(X_train, y_train)

y_pred = decision_tree_model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")

#Quesion 5

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    errors = {
        'train_mae': mean_absolute_error(y_train, y_train_pred),
        'train_mse': mean_squared_error(y_train, y_train_pred),
        'train_rmse': mean_squared_error(y_train, y_train_pred, squared=False),
        'test_mae': mean_absolute_error(y_test, y_test_pred),
        'test_mse': mean_squared_error(y_test, y_test_pred),
        'test_rmse': mean_squared_error(y_test, y_test_pred, squared=False),
    }
    return errors

decision_tree_default = DecisionTreeRegressor(random_state=42)
dt_default_errors = train_and_evaluate_model(decision_tree_default, X_train, X_test, y_train, y_test)

random_forest_default = RandomForestRegressor(random_state=42)
rf_default_errors = train_and_evaluate_model(random_forest_default, X_train, X_test, y_train, y_test)

decision_tree_modified = DecisionTreeRegressor(max_depth=5, random_state=42)
dt_modified_errors = train_and_evaluate_model(decision_tree_modified, X_train, X_test, y_train, y_test)

random_forest_modified = RandomForestRegressor(max_depth=5, random_state=42)
rf_modified_errors = train_and_evaluate_model(random_forest_modified, X_train, X_test, y_train, y_test)

print("Default Decision Tree Errors:", dt_default_errors)
print("Default Random Forest Errors:", rf_default_errors)
print("Modified Decision Tree Errors:", dt_modified_errors)
print("Modified Random Forest Errors:", rf_modified_errors)