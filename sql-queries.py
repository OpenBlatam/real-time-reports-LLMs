from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load DeepSeek Coder model
model_id = "deepseek-ai/deepseek-coder-6.7b-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto")

# Prompt Template for SQL generation
def generate_clickhouse_sql(prompt):
    system_prompt = (
        "You are an expert SQL generator for ClickHouse. "
        "Generate valid ClickHouse SQL queries based on user input.\n\n"
        f"Instruction: {prompt}\n"
        "SQL:"
    )

    inputs = tokenizer(system_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_length=512,
        temperature=0.3,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id
    )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated.split("SQL:")[-1].strip()

# Example usage
prompt = "Get total spend and revenue by platform in the last 7 days"
print(generate_clickhouse_sql(prompt))

