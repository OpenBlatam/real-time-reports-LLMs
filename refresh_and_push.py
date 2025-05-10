import os
import requests
from dotenv import load_dotenv
import datetime
import clickhouse_connect

load_dotenv()

CLIENT_KEY = os.getenv("TIKTOK_CLIENT_KEY")
CLIENT_SECRET = os.getenv("TIKTOK_CLIENT_SECRET")
AUTH_CODE = os.getenv("TIKTOK_AUTH_CODE")
CLICKHOUSE_HOST = os.getenv("CLICKHOUSE_HOST")

def refresh_access_token():
    url = "https://business-api.tiktok.com/open_api/v1.2/oauth2/access_token/"
    data = {
        "app_id": CLIENT_KEY,
        "secret": CLIENT_SECRET,
        "auth_code": AUTH_CODE,
        "grant_type": "authorized_code"
    }
    response = requests.post(url, data=data)
    result = response.json()
    return result.get("data", {}).get("access_token"), result.get("data", {}).get("advertiser_id")

def fetch_ads_data(token, advertiser_id):
    url = "https://business-api.tiktok.com/open_api/v1.2/report/ad/get/"
    headers = {"Access-Token": token}
    payload = {
        "advertiser_id": advertiser_id,
        "report_type": "BASIC",
        "report_name": "example_report",
        "dimensions": ["campaign_name"],
        "metrics": ["spend", "impressions", "clicks"],
        "start_date": (datetime.date.today() - datetime.timedelta(days=1)).isoformat(),
        "end_date": datetime.date.today().isoformat()
    }
    response = requests.post(url, json=payload, headers=headers)
    return response.json().get("data", {}).get("list", [])

def push_to_clickhouse(data):
    client = clickhouse_connect.get_client(host=CLICKHOUSE_HOST.replace("http://", ""))
    client.command("""
    CREATE TABLE IF NOT EXISTS tiktok_ads (
        date Date,
        campaign_name String,
        impressions UInt32,
        clicks UInt32,
        spend Float64
    ) ENGINE = MergeTree() ORDER BY date
    """)
    rows = []
    for item in data:
        rows.append([
            datetime.date.today(),
            item["dimensions"]["campaign_name"],
            int(item["metrics"]["impressions"]),
            int(item["metrics"]["clicks"]),
            float(item["metrics"]["spend"])
        ])
    client.insert("tiktok_ads", rows, column_names=["date", "campaign_name", "impressions", "clicks", "spend"])

if __name__ == "__main__":
    access_token, advertiser_id = refresh_access_token()
    if not access_token:
        print("❌ Token refresh failed.")
        exit(1)
    print("✅ Access token refreshed.")
    ads_data = fetch_ads_data(access_token, advertiser_id)
    print(f"📦 Retrieved {len(ads_data)} ad records.")
    push_to_clickhouse(ads_data)
    print("📊 Data pushed to ClickHouse.")

