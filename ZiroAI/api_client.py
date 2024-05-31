# api_client.py

import requests

class ZiroPayAPI:
    def __init__(self, base_url):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json',
        })

    def get(self, path, params=None):
        try:
            response = self.session.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e)}

    def post(self, path, data=None):
        try:
            response = self.session.post(f"{self.base_url}{path}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e)}

    def put(self, path, data=None):
        try:
            response = self.session.put(f"{self.base_url}{path}", json=data)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            return {'error': str(e)}

    def authenticate(self, username, password):
        auth_data = {
            'username': username,
            'password': password
        }
        response = self.post("/auth/login", data=auth_data)
        if 'token' in response:
            self.session.headers.update({'Authorization': f"Bearer {response['token']}"})
            return True
        return False

    def fetch_user_profile(self, user_id):
        return self.get(f"/user/{user_id}")

    def update_transaction_limits(self, user_id, new_limit):
        return self.put(f"/user/{user_id}/limits", data={'new_limit': new_limit})
