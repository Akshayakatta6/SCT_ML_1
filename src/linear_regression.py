# Task 01: House Price Prediction using Linear Regression
# Dataset: Kaggle - House Prices Advanced Regression Techniques

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --------------------------------------------------
# 1. Load Dataset (Robust Path Handling)
# --------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "train.csv")

data = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# 2. Select Features and Target
# --------------------------------------------------
X = data[['GrLivArea', 'BedroomAbvGr', 'FullBath']]
y = data['SalePrice']

# --------------------------------------------------
# 3. Handle Missing Values
# --------------------------------------------------
X = X.fillna(X.mean())

# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------------------------
# 5. Train Linear Regression Model
# --------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# --------------------------------------------------
# 6. Predictions
# --------------------------------------------------
y_pred = model.predict(X_test)

# --------------------------------------------------
# 7. Model Evaluation
# --------------------------------------------------
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("Model Evaluation Results")
print("------------------------")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R² Score: {r2:.4f}")

# --------------------------------------------------
# 8. Model Coefficients
# --------------------------------------------------
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

print("\nModel Coefficients")
print("------------------")
print(coefficients)
print(f"Intercept: {model.intercept_:.2f}")

# ==================================================
# 9. VISUALIZATION: ALL GRAPHS IN ONE CLEAN DISPLAY
# ==================================================

fig, axes = plt.subplots(2, 2, figsize=(14, 9))  # reduced size

# --------------------------------------------------
# Graph 1: Actual vs Predicted Prices
# --------------------------------------------------
axes[0, 0].scatter(y_test, y_pred)
axes[0, 0].plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()]
)
axes[0, 0].set_title("Actual vs Predicted Prices", fontsize=11)
axes[0, 0].set_xlabel("Actual Price", fontsize=10)
axes[0, 0].set_ylabel("Predicted Price", fontsize=10)

# --------------------------------------------------
# Feature graphs settings
# --------------------------------------------------
features = [
    ('GrLivArea', 'Square Footage'),
    ('BedroomAbvGr', 'Bedrooms'),
    ('FullBath', 'Bathrooms')
]

positions = [(0, 1), (1, 0), (1, 1)]

# --------------------------------------------------
# Graphs 2–4: Single-Feature Linear Regression
# --------------------------------------------------
for (feature, label), pos in zip(features, positions):
    X_single = data[[feature]]
    y_single = data['SalePrice']

    model_single = LinearRegression()
    model_single.fit(X_single, y_single)

    sorted_idx = np.argsort(X_single.values.flatten())
    X_sorted = X_single.values[sorted_idx]
    X_sorted_df = pd.DataFrame(X_sorted, columns=[feature])
    y_line = model_single.predict(X_sorted_df)

    axes[pos].scatter(X_single, y_single)
    axes[pos].plot(X_sorted, y_line)
    axes[pos].set_title(f"{label} vs Price", fontsize=11)
    axes[pos].set_xlabel(label, fontsize=10)
    axes[pos].set_ylabel("House Price", fontsize=10)

# --------------------------------------------------
# Layout, Save & Show
# --------------------------------------------------
plt.tight_layout(pad=2.0)
plt.savefig(os.path.join(BASE_DIR, "..", "outputs", "all_regression_graphs.png"))
plt.show()

# --------------------------------------------------
# 10. Sample Prediction
# --------------------------------------------------
sample_house = pd.DataFrame({
    'GrLivArea': [2000],
    'BedroomAbvGr': [3],
    'FullBath': [2]
})

predicted_price = model.predict(sample_house)

print("\nSample Prediction")
print("-----------------")
print(f"Predicted House Price: ₹ {predicted_price[0]:,.2f}")
