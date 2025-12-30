import requests
import sys

print("=" * 50)
print("StockVision Status Check")
print("=" * 50)

# Check Backend
try:
    response = requests.get('http://localhost:5000/health', timeout=2)
    if response.status_code == 200:
        data = response.json()
        print(f"[OK] Backend API: RUNNING - {data['message']}")
        print(f"     URL: http://localhost:5000")
    else:
        print(f"[ERROR] Backend API: ERROR - Status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Backend API: NOT RUNNING - {e}")
    sys.exit(1)

# Check Frontend
try:
    response = requests.get('http://localhost:8501', timeout=2)
    if response.status_code == 200:
        print(f"[OK] Frontend UI: RUNNING - Status {response.status_code}")
        print(f"     URL: http://localhost:8501")
    else:
        print(f"[ERROR] Frontend UI: ERROR - Status {response.status_code}")
        sys.exit(1)
except Exception as e:
    print(f"[ERROR] Frontend UI: NOT RUNNING - {e}")
    sys.exit(1)

print("\n" + "=" * 50)
print("[SUCCESS] Both services are running successfully!")
print("=" * 50)
print("\nOpen your browser to: http://localhost:8501")
print("\nIf the browser didn't open automatically, click the link above.")

