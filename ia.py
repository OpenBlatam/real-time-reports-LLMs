import streamlit as st
from langchain import OpenAI, LLMChain
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.memory import ConversationBufferMemory
import pandas as pd
import quantlib as ql  # QuantLib Python bindings
import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text
import networkx as nx
import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# -- Data Loader -------------------------------------------------------------
def load_financial_data():
    """Carga datos financieros en tiempo real (reemplazar con Kafka/Airbyte)."""
    df = pd.read_csv('data/financial_timeseries.csv', parse_dates=['date'])
    return df

# -- Tool Functions ----------------------------------------------------------

def run_dcf(cash_flows: str) -> str:
    flows = [float(x) for x in cash_flows.split(',')]
    r = 0.1  # tasa de descuento
    pv = sum(cf / ((1 + r) ** i) for i, cf in enumerate(flows, start=1))
    return f"VAN (DCF) calculado: {pv:.2f}"


def run_monte_carlo(params: str) -> str:
    sims = int(params)
    results = np.random.randn(sims).cumsum()
    return f"MC completado: valor final {results[-1]:.2f}"


def run_decision_tree_analysis(params: str) -> str:
    """Entrena un árbol de decisión sobre datos de ejemplo para prever KPI."""
    # Ejemplo mock: generar datos
    X = np.random.rand(100, 3)
    y = X @ np.array([2.0, -1.0, 0.5]) + 0.1 * np.random.randn(100)
    tree = DecisionTreeRegressor(max_depth=4)
    tree.fit(X, y)
    rules = export_text(tree, feature_names=[f'x{i}' for i in range(3)])
    return f"Árbol de decisión entrenado. Reglas:\n{rules}"


def run_gnn_contagion(params: str) -> str:
    """Simula contagio sistémico usando GNN sobre grafo financiero."""
    # Grafo de ejemplo: 10 nodos aleatorios
    G = nx.erdos_renyi_graph(10, 0.3)
    edge_index = torch.tensor(list(G.edges)).t().contiguous()
    x = torch.randn((10, 4))  # atributos de nodo
    data = Data(x=x, edge_index=edge_index)
    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = GCNConv(4, 8)
            self.conv2 = GCNConv(8, 2)
        def forward(self, data):
            x, edge_index = data.x, data.edge_index
            x = torch.relu(self.conv1(x, edge_index))
            x = self.conv2(x, edge_index)
            return x
    model = GCN()
    out = model(data)
    return f"GNN procesó grafo con shape salida {tuple(out.shape)}"

# -- LangChain Agent Setup --------------------------------------------------
tools = [
    Tool(name="DCF Analysis", func=run_dcf, description="Calcula VAN dado flujos de caja (csv)."),
    Tool(name="Monte Carlo", func=run_monte_carlo, description="Corre simulación Monte Carlo dado N iteraciones."),
    Tool(name="Decision Tree", func=run_decision_tree_analysis, description="Entrena árbol de decisión y muestra reglas."),
    Tool(name="GNN Contagion", func=run_gnn_contagion, description="Simula contagio sistémico con GNN."),
]

llm = OpenAI(temperature=0)
memory = ConversationBufferMemory(memory_key="chat_history")
agent = initialize_agent(tools, llm, agent=AgentType.OPENAI_FUNCTIONS, verbose=True, memory=memory)

# -- Streamlit UI ------------------------------------------------------------
st.set_page_config(page_title="🧠 AI Financial Suite", layout="wide")
st.title("🧠 AI Financial Suite en Tiempo Real")

st.sidebar.header("Data Source")
if st.sidebar.button("Reload Data"):
    data = load_financial_data()
    st.sidebar.success("Datos recargados")
else:
    data = load_financial_data()
    st.sidebar.write(f"{len(data)} registros cargados")

st.header("Consulta tu modelo financiero con IA")
user_input = st.text_input("Escribe tu consulta o comando:")
if user_input:
    with st.spinner("Analizando..."):
        response = agent.run(user_input)
    st.markdown(f"**Respuesta:** {response}")

st.header("Serie de Tiempo Financiera")
st.line_chart(data.set_index('date')['value'])

# -- Deployment: Dockerfile & Kubernetes -------------------------------------
# Dockerfile:
# FROM python:3.10-slim
# WORKDIR /app
# COPY . /app
# RUN pip install -r requirements.txt
# EXPOSE 8501
# CMD ["streamlit", "run", "app.py", "--server.port=8501"]

# k8s-deployment.yaml:
# apiVersion: apps/v1
# kind: Deployment
# metadata:
#   name: ai-financial-suite
# spec:
#   replicas: 2
#   selector:
#     matchLabels:
#       app: ai-financial-suite
#   template:
#     metadata:
#       labels:
#         app: ai-financial-suite
#     spec:
#       containers:
#       - name: suite
#         image: tu-registro/ai-financial-suite:latest
#         ports:
#         - containerPort: 8501
#
# apiVersion: v1
# kind: Service
# metadata:
#   name: ai-financial-service
# spec:
#   type: LoadBalancer
#   ports:
#   - port: 80
#     targetPort: 8501
#   selector:
#     app: ai-financial-suite
