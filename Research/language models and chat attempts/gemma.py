from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, Trainer, TrainingArguments
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from datasets import Dataset, load_dataset
import pandas as pd
import json

print("running gemma")
# Load tokenizer and model
accelerator = Accelerator()
tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base", device_map="auto", torch_dtype=torch.float16)
#model = AutoModelForSeq2SeqLM.from_pretrained("flan-t5-finetuned", device_map="auto", torch_dtype=torch.float16)

model, tokenizer = accelerator.prepare(model, tokenizer)

def fine_tune():
    with open('intents.json') as f:
        intents = json.load(f)

    # Prepare the data for T5
    data = {'input_text': [], 'target_text': []}
    for intent in intents['intents']:
        for pattern in intent['patterns']:
            data['input_text'].append(pattern)
            data['target_text'].append(intent['tag'])

    df = pd.DataFrame(data)

    train_df, val_df = train_test_split(df, test_size=0.1)

    tokenized_train = tokenize_function(train_df)
    tokenized_val = tokenize_function(val_df)

    train_dataset = Dataset.from_pandas(tokenized_train)
    val_dataset = Dataset.from_pandas(tokenized_val)

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=8)
    val_dataloader = DataLoader(val_dataset, batch_size=8)

    model = T5ForConditionalGeneration.from_pretrained('google/flan-t5-base')

    model, train_dataloader, val_dataloader = accelerator.prepare(
        model, train_dataloader, val_dataloader
    )

    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=30,
        weight_decay=0.01,
        save_total_limit=3,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()

    model.save_pretrained("flan-t5-finetuned")

class T5IntentDataset(Dataset):
    def __init__(self, tokenized_data):
        self.tokenized_data = tokenized_data
    
    def __len__(self):
        return len(self.tokenized_data)
    
    def __getitem__(self, idx):
        if isinstance(self.tokenized_data, pd.DataFrame):
            # Assuming the DataFrame has columns 'input_ids', 'attention_mask', 'labels'
            input_ids = torch.tensor(self.tokenized_data.iloc[idx]['input_ids'])
            attention_mask = torch.tensor(self.tokenized_data.iloc[idx]['attention_mask'])
            labels = torch.tensor(self.tokenized_data.iloc[idx]['labels'])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


    
def tokenize_function(df):
    # Tokenize all texts and prepare them for the model
    inputs = tokenizer(df['input_text'].tolist(), max_length=128, padding='max_length', truncation=True)
    targets = tokenizer(df['target_text'].tolist(), max_length=30, padding='max_length', truncation=True)

    # Convert lists of tokenized inputs into a DataFrame
    tokenized_inputs = {k: v for k, v in inputs.items()}
    tokenized_targets = {"labels": targets["input_ids"]}
    
    # Combine inputs and labels
    tokenized_data = {**tokenized_inputs, **tokenized_targets}
    return pd.DataFrame(tokenized_data)

def recognize_intent(input_text):
    """
    Use Gemma to get a general understanding of the input and potentially recognize the intent.
    This function is a placeholder; in practice, you might add more sophisticated logic here,
    potentially including keyword spotting or even fine-tuning Gemma on your specific intents.
    """
    # Placeholder for simplicity
    if "spend last month" in input_text:
        return "query_last_month_expense"
    elif "spend next month" in input_text:
        return "predict_next_month_expense"
    else:
        return "general_query"

def handle_intent(intent, input_text):
    """
    Execute custom logic based on the recognized intent.
    """
    if intent == "query_last_month_expense":
        # Placeholder function for fetching and summing last month's expenses
        expenses = get_last_month_expenses()
        response = f"Last month, you spent ${expenses}."
    elif intent == "predict_next_month_expense":
        # Placeholder for predictive model logic
        prediction = predict_next_month_expenses()
        response = f"Next month, you might spend around ${prediction}."
    else:
        # Fallback/general queries handled directly by Gemma
        response = generate_response_with_gemma(input_text)
    
    return response

def get_last_month_expenses():
    # Implement fetching and summing last month's expenses from your data
    return 1234.56

def predict_next_month_expenses():
    # Implement prediction logic here, possibly connecting to a forecasting model
    return 1300.00

def generate_response_with_gemma(input_text):
    # Implement calling Gemma model for general query responses
    encoded_input = tokenizer(input_text, return_tensors="pt")
    input_ids = encoded_input["input_ids"].to(accelerator.device)
    output_ids = accelerator.unwrap_model(model).generate(input_ids, max_new_tokens=30, num_return_sequences=1)
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def generate_intent(input_text):
    prompt = "Identify intent for: " + input_text
    encoded_input = tokenizer(prompt, return_tensors="pt")
    input_ids = encoded_input["input_ids"].to(accelerator.device)
    attention_mask = encoded_input["attention_mask"].to(accelerator.device)

    output_ids = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=30, num_return_sequences=1)
    intent = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    return intent
fine_tune()
while(True):
    input_text = input("Ask Ziro: ")
    intent = generate_intent(input_text)
    print(intent)