import requests
import json

url = "http://localhost:3058/api/describe"
payload = {
    "asset_path": "images/truck.jpg",
    "prompt": "Identify the main segmentable objects and their visual attributes.",
    "want_breakdown": True
}

try:
    response = requests.post(url, json=payload, timeout=60)
    print("Status Code:", response.status_code)
    try:
        print("JSON Response:", json.dumps(response.json(), indent=2))
    except:
        print("Raw Response Content:", response.text)
except Exception as e:
    print("Request failed:", e)
