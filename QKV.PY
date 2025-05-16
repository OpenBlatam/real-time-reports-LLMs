import streamlit as st
from langchain import OpenAI
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
import pandas as pd
import quantlib as ql
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
import networkx as nx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from prometheus_client import start_http_server, Counter, Histogram, CollectorRegistry, push_to_gateway
from influxdb import InfluxDBClient
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt
import plotly.express as px
from fpdf import FPDF
import io
import joblib
from numba import njit
from functools import lru_cache
from kafka import KafkaConsumer
from streamz import Stream
from pyspark.sql import SparkSession
from supersetapiclient.client import SupersetClient
from fbprophet import Prophet
from lifetimes.utils import summary_data_from_transaction_data
from lifetimes import BetaGeoFitter, GammaGammaFitter
from textblob import TextBlob
from sklearn.linear_model import LinearRegression
from grafanalib.core import Dashboard, TimeSeries, Target, Row

# -- CONFIGURATION -----------------------------------------------------------
registry = CollectorRegistry()
dcf_counter = Counter('dcf_requests_total', 'Total DCF requests', registry=registry)
mc_counter = Counter('monte_carlo_requests_total', 'Total Monte Carlo requests', registry=registry)
request_hist = Histogram('request_latency_seconds', 'Latency of requests', registry=registry)

# InfluxDB client
def get_influx():
    return InfluxDBClient(host='influxdb_host', port=8086, database='financial_metrics')

# Prometheus Pushgateway
gateway = 'http://pushgateway:9091'

# SQLAlchemy engine
sql_engine = create_engine('postgresql://user:pass@db:5432/financial')

# Spark session
spark = SparkSession.builder.appName('FinancialAnalytics').getOrCreate()

# Superset client
superset = SupersetClient('http://superset:8088', 'admin', 'password')

# Kafka & Streamz
def start_stream():
    consumer = KafkaConsumer('financial_events', bootstrap_servers=['kafka:9092'], value_deserializer=lambda m: m.decode())
    source = Stream()
    source.map(lambda msg: pd.read_json(msg)).sink(lambda df: get_influx().write_points(df.to_dict('records')))
    for msg in consumer:
        source.emit(msg.value)

# -- UTILITIES ---------------------------------------------------------------
@st.cache_data
def load_csv(path='data.csv'):
    return pd.read_csv(path, parse_dates=['date'])

@lru_cache(maxsize=4)
def fast_mc(n):
    return monte_carlo_sim(n)

@njit
def monte_carlo_sim(n):
    arr = np.random.randn(n)
    return arr.cumsum()

# -- FINANCIAL TOOLS ---------------------------------------------------------
@request_hist.time()
def run_dcf(cash_flows: str) -> str:
    dcf_counter.inc()
    flows = np.fromstring(cash_flows, sep=',')
    pv = (flows / (1 + 0.1) ** np.arange(1, len(flows) + 1)).sum()
    influx = get_influx()
    influx.write_points([{'measurement':'dcf','fields':{'value':float(pv)}}])
    push_to_gateway(gateway, job='financial_suite', registry=registry)
    return f"VAN: {pv:.2f}"

@request_hist.time()
def run_monte_carlo(params: str) -> str:
    mc_counter.inc()
    sims = int(params)
    last = float(fast_mc(sims)[-1])
    influx = get_influx()
    influx.write_points([{'measurement':'mc','fields':{'last':last}}])
    push_to_gateway(gateway, job='financial_suite', registry=registry)
    return f"MC Last: {last:.2f}"

# -- MARKETING TOOLS ---------------------------------------------------------
def run_prophet_forecast(csv_data: str) -> str:
    df = pd.read_csv(io.StringIO(csv_data), parse_dates=['ds'])
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)
    buf = io.StringIO()
    forecast[['ds','yhat','yhat_lower','yhat_upper']].to_csv(buf, index=False)
    return buf.getvalue()


def run_clv_prediction(transactions_csv: str) -> str:
    txns = pd.read_csv(io.StringIO(transactions_csv), parse_dates=['date'])
    summary = summary_data_from_transaction_data(txns, 'customer_id', 'date', monetary_value_col='amount', observation_period_end=txns['date'].max())
    bgf = BetaGeoFitter()
    bgf.fit(summary['frequency'], summary['recency'], summary['T'])
    ggf = GammaGammaFitter()
    ggf.fit(summary['frequency'], summary['monetary_value'])
    clv = ggf.customer_lifetime_value(bgf, summary['frequency'], summary['recency'], summary['T'], time=6, discount_rate=0.01)
    return clv.to_csv()


def run_cohort_analysis(transactions_csv: str) -> str:
    df = pd.read_csv(io.StringIO(transactions_csv), parse_dates=['date'])
    df['cohort_month'] = df.groupby('customer_id')['date'].transform('min').dt.to_period('M')
    df['order_month'] = df['date'].dt.to_period('M')
    cohorts = df.groupby(['cohort_month','order_month']).agg({'customer_id':'nunique'}).reset_index()
    buf = io.StringIO()
    cohorts.to_csv(buf, index=False)
    return buf.getvalue()


def run_marketing_mix_model(data_csv: str) -> str:
    df = pd.read_csv(io.StringIO(data_csv), parse_dates=['ds'])
    X = df[['tv','social','digital']].values
    y = df['sales'].values
    reg = LinearRegression().fit(X, y)
    coef = dict(zip(['tv','social','digital'], reg.coef_))
    return str(coef)


def run_sentiment_analysis(text: str) -> str:
    blob = TextBlob(text)
    return f"Polarity: {blob.polarity:.2f}, Subjectivity: {blob.subjectivity:.2f}"

# -- AGENT SETUP -------------------------------------------------------------
tools = [
    Tool(name="DCF", func=run_dcf, description="Calcula VAN."),
    Tool(name="Monte Carlo", func=run_monte_carlo, description="Simulación MC."),
    Tool(name="Prophet Forecast", func=run_prophet_forecast, description="Forecast con Prophet."),
    Tool(name="CLV Prediction", func=run_clv_prediction, description="Predice CLV."),
    Tool(name="Cohort Analysis", func=run_cohort_analysis, description="Análisis de cohortes."),
    Tool(name="Marketing Mix", func=run_marketing_mix_model, description="MMM."),
    Tool(name="Sentiment Analysis", func=run_sentiment_analysis, description="Sentiment Analysis."),
]
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, memory=memory)

# -- STREAMLIT UI ------------------------------------------------------------
st.set_page_config(page_title="Real-Time Financial & Marketing Suite", layout="wide")
st.title("🚀 Real-Time Financial & Marketing Analytics")

# Start background streaming
t.start_stream()  # starts Kafka->Influx pipeline

# Sidebar Data Sources
st.sidebar.header("Data Sources & BI")
st.sidebar.markdown(
    "- CSV Files\n- Kafka Stream\n- SQL Database\n- Spark Tables\n- InfluxDB Metrics\n- Superset\n- Prometheus Metrics"
)

# Query Interface
query = st.text_input("Consulta IA o Marketing:")
if query:
    with st.spinner("Procesando..."):
        st.write(agent.run(query))

# Real-time VAN Chart\if st.checkbox("Show real-time VAN chart"):
    df = get_influx().query('SELECT mean("value") as v FROM "dcf" WHERE time > now() - 5m GROUP BY time(30s)').get('financial_metrics')
    df['time'] = pd.to_datetime(df['time'])
    fig = px.line(df, x='time', y='v', title='VAN Real-Time')
    st.plotly_chart(fig, use_container_width=True)

# Marketing Reports Section
st.header("Marketing Reports")
if st.sidebar.button("Prophet Forecast"):
    csv_file = st.file_uploader("Upload CSV with ds,y columns:")
    if csv_file:
        st.download_button("Download Forecast", run_prophet_forecast(csv_file.getvalue().decode()), "forecast.csv", "text/csv")

# Add similar sidebar entries for other marketing tools...

# Grafana Dashboard as Code
st.header("Grafana Dashboard Definition")
st.code(
    Dashboard(
        title="AI Financial Suite",
        rows=[Row(panels=[TimeSeries(title="DCF Rate", dataSource="Prometheus", targets=[Target(expr="rate(dcf_requests_total[1m])")])])]
    ).to_json_data(), language='json'
)

# End of app
