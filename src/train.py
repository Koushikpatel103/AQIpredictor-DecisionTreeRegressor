import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from preprocess import (
    load_data,
    preprocess_data
)

# Load

df = load_data()

# Preprocess

X_train, X_test, y_train, y_test, feature_names = preprocess_data(df)

# GridSearch

param_grid = {

    'n_estimators':[100,200],

    'max_depth':[10,20,None],

    'min_samples_split':[2,5],

    'min_samples_leaf':[1,2]
}

grid = GridSearchCV(

    RandomForestRegressor(
        random_state=42
    ),

    param_grid,

    cv=3,

    scoring='r2',

    n_jobs=-1
)

grid.fit(
    X_train,
    y_train
)

model = grid.best_estimator_

print(
    grid.best_params_
)

# Prediction

y_pred = model.predict(
    X_test
)

# Metrics

r2 = r2_score(
    y_test,
    y_pred
)

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = mean_squared_error(
    y_test,
    y_pred
) ** 0.5

print("R2:", r2)
print("MAE:", mae)
print("RMSE:", rmse)

# Feature Importance

importance = model.feature_importances_

feat_df = pd.DataFrame({

    'Feature':feature_names,
    'Importance':importance

}).sort_values(
    by='Importance',
    ascending=False
)

plt.figure(figsize=(8,5))

sns.barplot(
    data=feat_df,
    x='Importance',
    y='Feature'
)

plt.title(
    "Feature Importance"
)

plt.show()

# Save

joblib.dump(
    model,
    "Models/model.pkl"
)

print(
    "Model Saved"
)