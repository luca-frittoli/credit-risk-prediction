# Credit Risk Prediction

A small machine learning project that predicts the probability of default on a *Pay Later* purchase and serves the model through a REST API.

## The case

A Pay Later loan lets consumers pay for a purchase 14 days after buying it. Customers who have not paid by then get a reminder and 7 more days. A loan still unpaid after 21 days goes to debt collection, and this counts as a **default**.

The goal is a model that estimates the probability of default for each new loan. The Underwriting team uses these probabilities to accept or decline transactions and to choose how much credit risk to take. The full case description is in [`docs/case_description.md`](docs/case_description.md).

## Proposed solution

The target is `default_flag`, which is true when `amount_outstanding_21d > 0`. The default rate in the data is about 5.4%.

Instead of interpreting a classifier's score as a probability, the problem is framed as a regression on `default_flag`. A **Decision Tree** (`max_depth=4`, `min_samples_leaf=300`) splits the feature space into clusters and predicts, for each new loan, the observed default rate of the cluster it falls into. That rate is a direct estimate of the default probability.

Models were compared with AUC and R-squared alongside MSE and MAE, because with such a strong class imbalance a model that always predicts 0 can look good on MSE/MAE alone. The Decision Tree performed best on AUC, R-squared and MSE among Logistic Regression, Linear Regression, Random Forest and XGBoost.

The model uses four features:

| Feature | Type | Description |
|---|---|---|
| `merchant_group` | string | Group of the merchant where the purchase is made |
| `days_since_first_loan` | int | Days since the consumer's first loan |
| `num_confirmed_payments_6m` | int | Repayments made in the 6 months before the loan |
| `num_failed_payments_1y` | int | Repayments missed in the year before the loan |

The full reasoning, results table and next steps are in the one-page report [`docs/case_study_report.md`](docs/case_study_report.md). My data exploration and model selection is in  [`notebooks/analysis.ipynb`](notebooks/analysis.ipynb)

## Project structure

```
.
├── app.py              # FastAPI app serving the model
├── model.pkl           # Trained Decision Tree
├── train_model.py      # Training script producing model.pkl
├── try_api.py          # Example of calling the API from Python
├── README.md
├── requirements.txt
└── data/
    └── data_sample.csv               # Sample dataset (just 500 rows)
└── docs/
    ├── case_description.md           # Case description
    └── case_study_report.md          # One-page report
└── notebooks/
    └── analysis.ipynb                # Jupyter notebook for data exploration
```

## Running the API locally

**1. Create a virtual environment and install the dependencies**

```bash
python -m venv venv
source venv/bin/activate        # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Use the scikit-learn version the model was trained with to avoid errors when loading `model.pkl`.

**2. Start the server** from the project root, where `app.py` and `model.pkl` are:

```bash
uvicorn app:app --reload
```

The API is now available at http://127.0.0.1:8000.

**3. Explore the interactive docs** at http://127.0.0.1:8000/docs, where you can send test requests directly from the browser.

## Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Health check with a short description of the model |
| `POST` | `/predict` | Returns the predicted default probability for one loan |

## Example API call

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
        "merchant_group": "Entertainment",
        "days_since_first_loan": 120,
        "num_confirmed_payments_6m": 5,
        "num_failed_payments_1y": 1
      }'
```

Response:

```json
{"prediction": 0.0412}
```

`prediction` is the estimated probability of default, between 0 and 1. The value above is only illustrative. `merchant_group` should be one of the groups seen in the training data.

### Calling the API from Python

The script `try_api.py` shows how to call the API with the `requests` library. With the server running, open a second terminal and run:

```bash
python try_api.py
```
