# Logistics Late Orders Predictor

This Machine Learning project aims to predict the probability of a logistics order being delayed (`late_order`). It uses a supervised classification pipeline based on **XGBoost**, optimizing hyperparameters through `GridSearchCV` to ensure high accuracy.



## Project Overview

In the supply chain and logistics sector, anticipating delays is crucial for maintaining customer satisfaction and optimizing warehouse resources. This project processes multiple relational datasets—including city coordinates, distances, and product attributes—to build a robust predictive model.

### Key Workflow Steps:
1. **Data Cleaning:** Standardization of geographic names (e.g., standardizing "ATHENAS" to "Athens") and handling of null/missing values.
2. **Feature Engineering:** * Generation of bidirectional route logic.
   * Calculation of distances broken down into two main legs: *Origin Port -> Logistic Hub* and *Logistic Hub -> Final Customer*.
3. **Categorical Encoding:** Transformation of categorical string variables (such as customs procedures and 3PL providers) into numerical weights for the model.
4. **Model Training & Optimization:** Deployment of an `XGBClassifier` iterating over tree depth (`max_depth`) and the number of estimators (`n_estimators`) via a grid search.
5. **Prediction Output:** Generation of probability scores for the test set, formatted automatically for Kaggle competitions.

## Data Structure Requirements

To run this notebook successfully, ensure the following datasets (CSV format, separated by `;`) are located in the root directory:

| File Name | Description |
| :--- | :--- |
| `orders.csv` | Historical training data containing the target variable (`late_order`). |
| `test.csv` | The unseen data we need to run our predictions on. |
| `cities_data.csv` | Core mapping data connecting cities and coordinates. |
| `cities_data_costs.csv` | Cost information associated with specific logistic nodes. |
| `product_attributes.csv` | Specific characteristics of the items being shipped. |
| `product_weight_class.csv` | Categorization of products based on their physical weight. |

## Tech Stack

* **Programming Language:** Python 3.x
* **Data Manipulation:** `pandas`, `numpy`
* **Machine Learning:** `xgboost`, `scikit-learn` (`GridSearchCV`, `train_test_split`, `accuracy_score`)
