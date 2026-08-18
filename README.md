# 📈 Yes Bank Stock Price Prediction

## 📌 Overview

A Machine Learning project to predict the **next month's closing price of Yes Bank** using historical monthly stock-price data.

The project includes data analysis, time-series feature engineering, model comparison, hyperparameter tuning, and Streamlit deployment.
## 🚀 Live Demo

🔗 **Streamlit App:** [Open Live Application](yes-bank-future-price-prediction-8xbp3xbgc9rtbhgguxwkh4.streamlit.app)

The application is deployed using Streamlit Community Cloud.

## 🎯 Objective

To forecast the next month's `Close` price using historical stock-price patterns while avoiding data leakage through a chronological time-series split.

## 📊 Dataset

The dataset contains monthly Yes Bank stock prices from **July 2005 to November 2020**.

**Columns:**

* `Date`
* `Open`
* `High`
* `Low`
* `Close`

**Target:** Next month's `Close` price.

## 🔧 Approach

```text
Data Loading
     ↓
Data Cleaning
     ↓
EDA
     ↓
Feature Engineering
     ↓
Time-Series Train/Test Split
     ↓
Model Training
     ↓
Model Comparison
     ↓
Hyperparameter Tuning
     ↓
Final Prediction
     ↓
Streamlit Deployment
```

### Feature Engineering

The model uses:

* Lag features
* Moving averages
* Rolling standard deviation
* Monthly returns
* Month and Year

## 🤖 Models Used

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor

Random Forest is further optimized using `GridSearchCV` with `TimeSeriesSplit`.

## 📏 Evaluation Metrics

Models are evaluated using:

* **MAE** — Mean Absolute Error
* **RMSE** — Root Mean Squared Error
* **R² Score**

| Model               | MAE | RMSE | R² |
| ------------------- | --: | ---: | -: |
| Linear Regression   |   — |    — |  — |
| Random Forest       |   — |    — |  — |
| Gradient Boosting   |   — |    — |  — |
| Tuned Random Forest |   — |    — |  — |

> Add the actual scores after running the final model.

## 🛠️ Tech Stack

* Python 3.11
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn
* Joblib
* Streamlit

## 📁 Project Structure

```text
YesBank_Stock_Prediction/
│
├── data/
│   └── data_YesBank_StockPrices.csv
│
├── models/
│   └── yesbank_model.pkl
│
├── notebooks/
│   └── YesBank_Stock_Prediction.ipynb
│
├── src/
│   └── train_model.py
│
├── app.py
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 How to Run

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Environment

**Windows:**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Train the Model

```bash
python src/train_model.py
```

### 5. Run Streamlit

```bash
streamlit run app.py
```

## 📌 Key Highlights

* Time-series based forecasting
* No random train-test splitting
* Historical lag and rolling features
* Multiple ML models compared
* Hyperparameter tuning
* Model evaluation and feature importance
* Interactive Streamlit deployment

## ⚠️ Disclaimer

This project is for **educational and portfolio purposes only**. Stock-price predictions are uncertain and should not be considered financial or investment advice.
