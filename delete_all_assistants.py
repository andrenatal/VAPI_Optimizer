import os, requests
from dotenv import load_dotenv
load_dotenv()
import time

API_KEY = os.environ["VAPI_API_KEY"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

# List all assistants
resp = requests.get("https://api.vapi.ai/assistant", headers=HEADERS)
assistants = resp.json()

print(f"Found {len(assistants)} assistants:")
for a in assistants:
    print(f"  {a['id']} — {a.get('name', 'unnamed')}")

# Delete all
for a in assistants:
    r = requests.delete(f"https://api.vapi.ai/assistant/{a['id']}", headers=HEADERS)
    print(f"  Deleted {a['id']}: {r.status_code}")
    time.sleep(5)  # be nice to the API