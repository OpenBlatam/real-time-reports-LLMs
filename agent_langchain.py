from langchain.agents import create_sql_agent
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from langchain.sql_database import SQLDatabase
from langchain.llms import OpenAI  # Replace with DeepSeek wrapper if needed

from clickhouse_connect import get_client

# Setup ClickHouse connection
CLICKHOUSE_HOST = "localhost"
CLICKHOUSE_PORT = 8123
CLICKHOUSE_DB = "default"

client = get_client(host=CLICKHOUSE_HOST, port=CLICKHOUSE_PORT, username="default", password="")
db = SQLDatabase.from_uri(f"clickhouse+http://{CLICKHOUSE_HOST}:{CLICKHOUSE_PORT}/{CLICKHOUSE_DB}")

# LLM (Swap in DeepSeek or other model as needed)
llm = OpenAI(temperature=0)

# LangChain Agent
toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent = create_sql_agent(llm=llm, toolkit=toolkit, verbose=True)

# Ask anything
response = agent.run("What was the total ad spend per platform last week?")
print(response)

