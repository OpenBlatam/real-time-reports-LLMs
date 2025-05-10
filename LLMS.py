from transformers import T5ForConditionalGeneration, T5Tokenizer
from datasets import load_dataset

# Load a pre-trained T5 model and tokenizer
model = T5ForConditionalGeneration.from_pretrained("t5-small")
tokenizer = T5Tokenizer.from_pretrained("t5-small")

# Load your dataset (Example: you would replace this with your actual data)
dataset = load_dataset('csv', data_files='path_to_your_data.csv')

# Preprocessing the data: Format the input as "generate sql for: <instruction>"
def preprocess_data(examples):
    return {
        'input_text': [f'Generate SQL for: {instr}' for instr in examples['instruction']],
        'output_text': examples['sql_query']
    }

dataset = dataset.map(preprocess_data, remove_columns=['instruction', 'sql_query'])

# Tokenize the data
def tokenize_data(examples):
    inputs = tokenizer(examples['input_text'], padding="max_length", truncation=True, max_length=512)
    outputs = tokenizer(examples['output_text'], padding="max_length", truncation=True, max_length=512)
    return {
        'input_ids': inputs['input_ids'],
        'attention_mask': inputs['attention_mask'],
        'labels': outputs['input_ids']
    }

dataset = dataset.map(tokenize_data, batched=True)

# Fine-tuning the model (adjust epochs and batch_size as needed)
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='./results',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset['train'],
    eval_dataset=dataset['test'],
)

trainer.train()

