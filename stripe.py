import stripe
import pandas as pd
from clickhouse_connect import get_client
from datetime import datetime

# Stripe setup
stripe.api_key = 'sk_live_XXXX'  # Your Stripe secret key

# ClickHouse setup
client = get_client(host='localhost', port=8123)

# Fetch recent charges
charges = stripe.Charge.list(limit=100)
rows = []

for charge in charges.auto_paging_iter():
    rows.append({
        'id': charge['id'],
        'amount': charge['amount'],
        'fee': charge['balance_transaction'] and stripe.BalanceTransaction.retrieve(charge['balance_transaction'])['fee'] or 0,
        'created': datetime.utcfromtimestamp(charge['created']),
    })

df = pd.DataFrame(rows)

# Create table if not exists
client.command("""
CREATE TABLE IF NOT EXISTS stripe_transactions (
    id String,
    amount Float64,
    fee Float64,
    created DateTime
) ENGINE = MergeTree()
ORDER BY created
""")

# Insert into ClickHouse
client.insert_df('stripe_transactions', df)

