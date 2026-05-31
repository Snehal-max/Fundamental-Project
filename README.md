# Employee Attrition Prediction – ML Project

A machine learning project to predict whether an employee will stay or leave a company, based on demographic, professional, and workplace attributes.

---

## Project Structure
---

## Dataset

- **Records:** 59,598 training / 14,900 test
- **Features:** 24 columns (demographic, professional, workplace)
- **Target:** `Attrition` — `Stayed (0)` or `Left (1)`
- **Class Balance:** 31,260 Stayed | 28,338 Left (roughly balanced)
- **Missing Values:** None

Key features include Age, Gender, Monthly Income, Job Role, Job Level, Work-Life Balance, Job Satisfaction, Performance Rating, Overtime, Remote Work, Marital Status, and more.

---

## What's in the Notebook

### Step 1 – Data Understanding & Preprocessing
- Loaded and explored train/test datasets
- Checked for missing values (none found)
- Dropped `Employee ID` (non-predictive)
- Label-encoded target: `Stayed → 0`, `Left → 1`
- One-hot encoded all categorical columns using `get_dummies(drop_first=True)`
- Result: 41 input features after encoding

### Step 2 – Exploratory Data Analysis (EDA)
- Histograms and boxplots for numeric features
- Correlation heatmap
- Attrition distribution plot
- Pairplot for key numeric columns

### Step 3 – Model Implementation (5 Models)
All models trained on an 80/20 stratified train-validation split.

| Model | Notes |
|---|---|
| Logistic Regression | Scaled features, max_iter=1000 |
| Decision Tree | Default params, random_state=42 |
| Random Forest | 200 estimators, random_state=42 |
| KNN | k=5, scaled features |
| XGBoost | 200 estimators, max_depth=6, lr=0.1 |

### Step 4 – Model Evaluation
Each model evaluated using Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| XGBoost | 75.05% | 73.81% | 73.68% | 73.74% |
| Logistic Regression | 74.78% | 73.73% | 72.95% | 73.34% |
| Random Forest | 74.60% | 73.79% | 72.23% | 73.00% |
| ANN | 74.85% | 72.64% | 75.56% | 74.07% |
| KNN | 66.95% | 64.69% | 67.13% | 65.89% |
| Decision Tree | 66.17% | 64.35% | 64.68% | 64.51% |

### Step 5 – From-Scratch Implementation
Two algorithms implemented from scratch using only NumPy:

- **KNN Scratch** – Euclidean distance computed manually, majority vote among k=5 neighbors → matched library accuracy exactly (66.95%)
- **Logistic Regression Scratch** – Gradient descent with sigmoid activation, 1000 epochs, lr=0.01 → 74.50% vs library's 74.78% (gap due to optimizer differences)

### Step 6 – ANN (Optional Task)
A Feedforward MLP built with TensorFlow/Keras:
- Architecture: `128 → Dropout(0.3) → 64 → Dropout(0.3) → 32 → 1`
- Activation: ReLU (hidden), Sigmoid (output)
- Optimizer: Adam | Loss: Binary Cross-Entropy | Epochs: 20
- **ANN achieved the highest Recall (75.56%)** among all models — best at catching employees who are about to leave

---

## How to Run

1. Clone or download the project folder
2. Install dependencies:
```bash
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
```
3. Place `train.csv` and `test.csv` in the same directory as the notebook
4. Open `Untitled.ipynb` in Jupyter and run all cells top to bottom

---

## Key Findings

- **Best accuracy:** XGBoost (75.05%)
- **Best recall (catching leavers):** ANN (75.56%)
- **Top predictors of attrition:** Marital Status (Single), Job Level (Senior), Remote Work, Work-Life Balance (Poor)
- Ensemble methods (XGBoost, Random Forest) significantly outperformed single models (Decision Tree, KNN)
- ~75% accuracy across models reflects the inherent difficulty of predicting human behavior from structured data

---

## Dependencies
pandas,
numpy,
matplotlib,
seaborn,
scikit-learn,
xgboost,
tensorflow
