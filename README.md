
# House Price Prediction using Linear Regression

This project implements a **Linear Regression model** to predict house prices using selected features from the **Kaggle House Prices: Advanced Regression Techniques** dataset.  
It is developed as part of a **Machine Learning internship task**.

---

## 📌 Project Overview

The goal of this project is to:
- Build a Linear Regression model
- Predict house prices based on key features
- Evaluate model performance
- Visualize relationships between features and house prices

---

## 📊 Dataset

- **Source:** Kaggle – House Prices: Advanced Regression Techniques
- **File used:** `train.csv`
- **Target variable:** `SalePrice`

### Selected Features
- `GrLivArea` – Above ground living area (square feet)
- `BedroomAbvGr` – Number of bedrooms
- `FullBath` – Number of full bathrooms

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Git & GitHub

---

## 🧠 Model Details

- **Algorithm:** Linear Regression
- **Train-Test Split:** 80% training, 20% testing
- **Missing Values:** Handled using mean imputation

### Evaluation Metrics
- **RMSE (Root Mean Squared Error)**
- **R² Score**

---

## 📈 Visualizations

The project generates and saves the following graphs:
1. Actual vs Predicted House Prices
2. Square Footage vs House Price
3. Bedrooms vs House Price
4. Bathrooms vs House Price

All graphs are saved automatically in the `outputs/` folder.

---

## 📂 Project Structure

```

House-Price-Prediction-Linear-Regression/
│
├── data/
│   └── train.csv
│
├── src/
│   └── linear_regression.py
│
├── outputs/
│   └── all_regression_graphs.png
│
└── README.md

````

---

## ▶️ How to Run the Project

1. Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
````

2. Navigate to the project folder:

```bash
cd House-Price-Prediction-Linear-Regression
```

3. Install required libraries:

```bash
pip install pandas numpy scikit-learn matplotlib
```

4. Run the script:

```bash
python src/linear_regression.py
```

---

## ✅ Sample Output (Terminal)

```
Model Evaluation Results
------------------------
Root Mean Squared Error (RMSE): 52975.72
R² Score: 0.6341

Sample Prediction
-----------------
Predicted House Price: ₹ 240,377.51
```

---

## 🧾 Conclusion

* Square footage and bathrooms show a strong positive influence on house prices.
* Bedrooms have a weaker relationship with price.
* The model demonstrates reasonable performance for a basic Linear Regression approach.

---
