import requests
import pandas as pd
from clickhouse_driver import Client
from msal import ConfidentialClientApplication

# Fetch data from ClickHouse
client = Client('clickhouse-server')
query = """
SELECT
  toStartOfMinute(event_time) AS time,
  sum(spend) AS total_spend,
  sum(revenue) AS total_revenue,
  platform
FROM ads_metrics
WHERE event_time >= now() - INTERVAL 1 HOUR
GROUP BY time, platform
ORDER BY time
"""
data = client.execute(query)
df = pd.DataFrame(data, columns=["time", "total_spend", "total_revenue", "platform"])

# Convert the DataFrame to JSON format for Power BI
json_data = df.to_dict(orient='records')

# Power BI Authentication details (client id, secret, tenant id, etc.)
client_id = 'your-client-id'
client_secret = 'your-client-secret'
tenant_id = 'your-tenant-id'
resource_url = 'https://analysis.windows.net/powerbi/api'

# Authenticate with Power BI API
authority = f"https://login.microsoftonline.com/{tenant_id}"
app = ConfidentialClientApplication(client_id, authority=authority, client_credential=client_secret)
token_response = app.acquire_token_for_client(scopes=[resource_url + "/.default"])

access_token = token_response['access_token']

# Power BI REST API endpoint to push data
powerbi_api_url = "https://api.powerbi.com/v1.0/myorg/groups/{group_id}/datasets/{dataset_id}/tables/{table_name}/rows"

headers = {
    'Content-Type': 'application/json',
    'Authorization': f'Bearer {access_token}'
}

# Data to send to Power BI
body = {
    "rows": json_data
}

# Send data to Power BI dataset
response = requests.post(powerbi_api_url, headers=headers, json=body)

if response.status_code == 200:
    print("Data successfully pushed to Power BI!")
else:
    print(f"Error: {response.status_code}, {response.text}")

