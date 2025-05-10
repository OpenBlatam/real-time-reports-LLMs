import pandas as pd
from clickhouse_driver import Client
from tableau_api_lib import TableauServerConnection
from tableau_api_lib.utils.querying import get_projects, get_workbooks
from tableau_api_lib.utils.data import create_data_frame

# Define connection details for ClickHouse
clickhouse_client = Client('clickhouse-server')  # Hostname
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
data = clickhouse_client.execute(query)

# Convert to DataFrame for manipulation
df = pd.DataFrame(data, columns=["time", "total_spend", "total_revenue", "platform"])

# Connect to Tableau Server
connection = TableauServerConnection(
    server='http://your-tableau-server',
    username='your-username',
    password='your-password',
    site='your-site'
)

# Sign in to Tableau
connection.sign_in()

# Find the target project on Tableau Server
project_id = get_projects(connection).json()['projects']['project'][0]['id']

# Create the Tableau data source (create a CSV or use the DataFrame directly)
csv_file_path = '/path/to/your/csv_file.csv'
df.to_csv(csv_file_path, index=False)

# Upload the CSV to Tableau
response = connection.publish_data_source(
    datasource_file_path=csv_file_path,
    datasource_name='ads_performance_report',
    project_id=project_id
)

# If successful, you will get a response with the data source details
print(response.status_code)
print(response.text)

# Sign out from Tableau
connection.sign_out()

