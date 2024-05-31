# api_integration.py

import requests

def fetch_user_profile(user_id):
    response = requests.get(f"https://api.ziropay.com/user/{user_id}")
    return response.json()

def update_transaction_limits(user_id, new_limit):
    payload = {'new_limit': new_limit}
    response = requests.put(f"https://api.ziropay.com/user/{user_id}/limits", json=payload)
    return response.status_code == 200
