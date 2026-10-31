from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, mean_absolute_error
import pandas as pd
import joblib

if __name__ == '__main__':
    data_path = "data/data_sample.csv"
    df = pd.read_csv(data_path)

    # Create target variable "default" = "has not repaid after 21 days"
    df['default_flag'] = df['amount_outstanding_21d'] > 0

    cat_features = ['merchant_group']
    num_features = [
        'days_since_first_loan',
        'num_confirmed_payments_6m',
        'num_failed_payments_1y',
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
            ("num", "passthrough", num_features)
        ]
    )

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", DecisionTreeRegressor(max_depth=4, min_samples_leaf=300))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        df[cat_features+num_features], df['default_flag'], test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    joblib.dump(model, "model.pkl")

    pred = model.predict(X_test)
    print(
        f'AUC: {roc_auc_score(y_test, pred):.4f}\nR^2: {r2_score(y_test, pred):.4f}\nMSE: {mean_squared_error(y_test, pred):.4f}\nMAE: {mean_absolute_error(y_test, pred):.4f}'
    )