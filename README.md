# Credit Risk Evaluator

Predicts whether a credit-card customer will default on next month's payment, and explains each prediction with SHAP.

It trains three classifiers (Logistic Regression, Random Forest, XGBoost) on the UCI "Default of Credit Card Clients" dataset (30,000 customers, 23 features), saves the XGBoost model, and serves it in a Streamlit app where you enter a customer's details and get a default probability plus a SHAP force plot showing which features pushed the score up or down.

## Project layout

```
explore_data.py     # quick look at the dataset: shape, missing values, class balance
prepare_data.py     # cleans categories, 80/20 split, StandardScaler -> data/processed_data.pkl, model/scaler.pkl
train_model.py      # trains LR, RF and XGBoost, prints reports + ROC-AUC, saves model/credit_risk_model.pkl
shap_explainer.py   # global SHAP summary plot for the saved model
app.py              # Streamlit app: enter one customer, get prediction + SHAP explanation
```

## Setup

Requires Python 3.10+.

```bash
git clone https://github.com/yuktaablitz/credit-risk-evaluator.git
cd credit-risk-evaluator
python3 -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Get the data

The dataset is not committed (`data/` is gitignored).

1. Download "Default of Credit Card Clients" from the UCI Machine Learning Repository:
   https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients
2. Unzip it and put the Excel file at `data/default of credit card clients.xls` (keep the file name exactly).

## Run

```bash
python explore_data.py      # optional: inspect the data
python prepare_data.py      # writes data/processed_data.pkl and model/scaler.pkl
python train_model.py       # prints metrics for all 3 models, saves model/credit_risk_model.pkl
python shap_explainer.py    # optional: writes shap_summary_plot.png
streamlit run app.py        # opens http://localhost:8501
```

`app.py` needs `model/credit_risk_model.pkl` and `model/scaler.pkl`, so run `prepare_data.py` and `train_model.py` first.

## Input fields in the app

- `LIMIT_BAL`: credit limit
- `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`: demographics (codes as in the UCI dataset)
- `PAY_0`, `PAY_2` ... `PAY_6`: repayment status for the last 6 months (-2 = no use, -1 = paid on time, 1-8 = months of delay)
- `BILL_AMT1-6`: bill statement amounts for the last 6 months
- `PAY_AMT1-6`: amounts paid in the last 6 months

## Tech stack

Python, pandas, scikit-learn, XGBoost, SHAP, Streamlit, matplotlib, joblib.
