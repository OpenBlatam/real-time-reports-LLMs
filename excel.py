import pandas as pd
from clickhouse_driver import Client

# Connect to ClickHouse server
client = Client('clickhouse-server')  # Ensure this points to your ClickHouse server

# Define the SQL query to fetch your data
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

# Execute the query and fetch data
data = client.execute(query)

# Convert the fetched data to a pandas DataFrame
df = pd.DataFrame(data, columns=["time", "total_spend", "total_revenue", "platform"])

# Define the path where the Excel file will be saved
excel_file_path = "ads_performance_report.xlsx"

# Export the DataFrame to an Excel file
df.to_excel(excel_file_path, index=False, engine='openpyxl')

print(f"Data successfully exported to {excel_file_path}")

