import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import clickhouse_connect

# Connect to ClickHouse
client = clickhouse_connect.get_client(host='localhost', port=8123)

# Query the ads metrics
query = """
    SELECT
        platform,
        toDate(event_time) AS day,
        sum(spend) AS total_spend,
        sum(clicks) AS total_clicks,
        sum(impressions) AS total_impressions
    FROM ads_metrics
    GROUP BY platform, day
    ORDER BY day
"""

result = client.query_df(query)

# Preview data
print(result.head())

# Plot: Daily Spend by Platform
plt.figure(figsize=(12, 6))
sns.lineplot(data=result, x='day', y='total_spend', hue='platform')
plt.title('Daily Ad Spend by Platform')
plt.xlabel('Date')
plt.ylabel('Spend ($)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("spend_by_platform.png")
plt.show()

