from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import clickhouse_connect

app = FastAPI()

# Load DeepSeek model
model_id = "deepseek-ai/deepseek-coder-6.7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# ClickHouse client
ch_client = clickhouse_connect.get_client(host='localhost', port=8123)

# Input model
class QueryPrompt(BaseModel):
    prompt: str
    run_query: bool = True

# Generate SQL from prompt
def generate_sql(prompt: str) -> str:
    input_text = (
        "You are an expert ClickHouse SQL generator.\n"
        f"Instruction: {prompt}\nSQL:"
    )
    tokens = tokenizer(input_text, return_tensors="pt").to(model.device)
    outputs = model.generate(**tokens, max_length=512, temperature=0.3, do_sample=False)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result.split("SQL:")[-1].strip()

@app.post("/generate_sql/")
def generate_clickhouse_sql(query: QueryPrompt):
    sql = generate_sql(query.prompt)
    if query.run_query:
        try:
            result = ch_client.query(sql)
            return {"sql": sql, "data": result.result_rows}
        except Exception as e:
            return {"sql": sql, "error": str(e)}
    return {"sql": sql}

