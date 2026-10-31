import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "merchant_group": "Clothing & Shoes",
    "days_since_first_loan": 717,
    "num_confirmed_payments_6m": 36,
    "num_failed_payments_1y": 0
}

response = requests.post(url, json=payload)

if response.status_code == 200:
    print("Prediction:", response.json()["prediction"])
else:
    print("Error:", response.text)
