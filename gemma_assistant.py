import requests
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables

API_KEY = os.getenv("GEMMA3N_API_KEY")

def gemma3n_reply(prompt):
    url = "https://api.gemma3n.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    data = {"prompt": prompt}

    response = requests.post(url, headers=headers, json=data)
    response_json = response.json()

    print(response_json)  # Debugging

    if "reply" in response_json:
        return response_json["reply"]
    else:
        return f"Unexpected Response: {response_json}"
