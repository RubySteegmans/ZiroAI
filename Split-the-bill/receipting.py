import json
import os
from transformers import T5Tokenizer, T5ForConditionalGeneration, BertTokenizer, BertForTokenClassification, BertForMaskedLM
import torch
from torch.utils.data import DataLoader, Dataset

# Load the dataset
with open('another-dataset.json', 'r') as file:
    dataset = json.load(file)

# Prepare the prompts and targets
prompts = [
    f"You are a helpful assistant that extracts structured information from receipts and returns it in JSON format.\nExtract the following JSON structure from the text:\n{entry['ocr']}\nThe JSON structure should be:\n{{\"place_name\": \"PLACE_NAME\",\"timestamp\": \"TIMESTAMP\",\"items\": [{{\"name\": \"ITEM_NAME\",\"quantity\": 1,\"price\": PRICE}}],\"total\": TOTAL,\"currency\": \"CURRENCY\"}}"
    for entry in dataset
]

targets = [
    json.dumps({
        "place_name": entry["place_name"],
        "timestamp": entry["timestamp"],
        "items": entry["items"],
        "total": entry["total"],
        "currency": entry["currency"]
    })
    for entry in dataset
]

class ReceiptQADataset(Dataset):
    def __init__(self, prompts, targets, tokenizer, max_length=512):
        self.prompts = prompts
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        prompt = self.prompts[idx]
        target = self.targets[idx]
        
        inputs = self.tokenizer(prompt, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt")
        labels = self.tokenizer(target, max_length=self.max_length, padding='max_length', truncation=True, return_tensors="pt").input_ids

        return {
            'input_ids': inputs.input_ids.squeeze(),
            'attention_mask': inputs.attention_mask.squeeze(),
            'labels': labels.squeeze()
        }

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create a subset of the dataset for quick fine-tuning
num_samples_to_train = 64
train_dataset = ReceiptQADataset(prompts[:num_samples_to_train], targets[:num_samples_to_train], tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)

from transformers import AdamW, get_linear_schedule_with_warmup

model = BertForMaskedLM.from_pretrained("bert-base-uncased")

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = len(train_dataloader) * 3  # 3 epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_loss = float('inf')
model_save_path = 'best_model'

# Fine-tuning loop
model.train()
for epoch in range(10):  # Number of epochs
    total_loss = 0
    for batch in train_dataloader:
        optimizer.zero_grad()

        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    avg_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}, Average Loss: {avg_loss}")

    if avg_loss < best_loss:
        best_loss = avg_loss
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)
        print(f"Best model saved with loss: {best_loss}")

# Testing the model with the rest of the dataset
model = BertForMaskedLM.from_pretrained(model_save_path)
model.to(device)
model.eval()
results = []

for entry in dataset[:5]:  # Adjust the number of test samples as needed
    ocr_text = entry['ocr']
    expected_json = {
        "place_name": entry["place_name"],
        "timestamp": entry["timestamp"],
        "items": entry["items"],
        "total": entry["total"],
        "currency": entry["currency"]
    }
    
    prompt = f"You are a helpful assistant that extracts structured information from receipts and returns it in JSON format.\nExtract the following JSON structure: \n{{\"place_name\": \"PLACE_NAME\",\"timestamp\": \"TIMESTAMP\",\"items\": [{{\"name\": \"ITEM_NAME\",\"quantity\": 1,\"price\": PRICE}}],\"total\": TOTAL,\"currency\": \"CURRENCY\"}} from the text:\n{ocr_text}"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True).to(device)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)  # Adjust max_new_tokens as needed
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print(f"OCR Text: {ocr_text}\nExpected JSON: {json.dumps(expected_json, indent=2)}\nRaw Response: {response}\n")

    results.append({"OCR Text": ocr_text, "Expected JSON": expected_json, "Raw Response": response})

# Save the results to a file
with open('predicted_results.json', 'w') as file:
    json.dump(results, file, indent=2)