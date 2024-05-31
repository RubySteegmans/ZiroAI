import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, pipeline
from accelerate import Accelerator
from sklearn.model_selection import train_test_split
import pandas as pd
import json
import random
from nltk_utils import bag_of_words, nltk_tokenize, stem
from model import NeuralNet
import pandas as pd
from datetime import datetime
import os

columns = ['User', 'Bot', 'Feedback']
conversation_log = pd.DataFrame(columns=columns)
tiny_llama_pipe = pipeline("text-generation", model="TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch_dtype=torch.bfloat16, device_map="auto")

X_train = []
y_train = []
all_words = []
tags = []
xy = []
chat_history_ids = None
conversation_history = []
bot_name = "Ziro"
username = "Ruby"
# Load and prepare your intents
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Define your Neural Network model and dataset as before
def create_NeuralNet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained simple model
    FILE = "data.pth"
    data = torch.load(FILE, map_location=device)  # Ensure model is loaded to the correct device

    model_simple = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
    model_simple.load_state_dict(data["model_state"])
    model_simple.eval()
    return model_simple

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = X_train
        self.y_data = y_train

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # we can call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

# Tokenization and dataset preparation function
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

def clean_response(response):
    # Split the response into parts based on "</s>"
    parts = response.split("<|assistant|>")
    
    chatbot_response = parts[1].strip()
    
    chatbot_response_lines = chatbot_response.split('\n')
    chatbot_response_cleaned = [line for line in chatbot_response_lines if not line.strip().lower().startswith("user:")]
    
    return chatbot_response

def post_process_response(generated_text, max_length=100):
    sentences = generated_text.split('.')

    # Rebuild the response with sentence limit and maximum character length.
    processed_response = ""
    for sentence in sentences:
        if len(processed_response) + len(sentence) + 1 <= max_length:
            processed_response += sentence + "."
        else:
            break

    # Trim leading and trailing spaces.
    processed_response = processed_response.strip()

    return processed_response

def recognize_intent_simple(input_text, model, all_words, tags):
    if not isinstance(input_text, str):
        input_text = str(input_text)
    sentence = nltk_tokenize(input_text)  # Use the nltk-based tokenize function
    X = bag_of_words(sentence, all_words)
    X = np.array(X).reshape(1, -1)
    X = torch.from_numpy(X).float().to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        return tag
    else:
        return None


# Function to handle the intent
def handle_intent(intent, intents_json):
    for intent_config in intents_json['intents']:
        if intent == intent_config["tag"]:
            return random.choice(intent_config['responses'])
    return "I do not understand..."

def load_data():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    FILE = "data.pth"
    data = torch.load(FILE, map_location=device)

    all_words = data["all_words"]
    tags = data["tags"]
    xy = data["xy"]
    
    for (pattern_sentence, tag) in xy:
        # X: bag of words for each pattern_sentence
        bag = bag_of_words(pattern_sentence, all_words)
        X_train.append(bag)
        # y: PyTorch CrossEntropyLoss needs only class labels, not one-hot
        label = tags.index(tag)
        y_train.append(label)
    
    return all_words, tags, xy

def generate_contextual_prompt(user_input):
    # Extract the last few exchanges to use as context
    context_exchanges = conversation_history[-5:]  # Adjust this number as needed
    context = " ".join([f"{exchange['user']} {exchange['bot']}" for exchange in context_exchanges])
    prompt = f"{context} User: {user_input} Bot:"
    return prompt

def generate_contextual_messages(user_input):
    # messages = [
    #     {"role": "system", "content": "You are a friendly chatbot that provides helpful responses and information."},
    #     *[
    #         {"role": "user", "content": exchange.get("user", "")},
    #         {"role": "bot", "content": exchange.get("bot", "")}
    #         for exchange in conversation_history[-2:]  # Use the last 5 exchanges for context
    #     ],
    #     {"role": "user", "content": user_input}
    # ]
    messages = [
        {"role": "system", "content": "You are a friendly chatbot that provides helpful responses and information."},
    ]
    return messages

def generate_response_with_tinyllama(user_input):
    global conversation_history
    messages = generate_contextual_messages(user_input)
    prompt = tiny_llama_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = tiny_llama_pipe(prompt, max_new_tokens=20, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    # Extract the generated text
    generated_text = outputs[0]["generated_text"]
    # Clean and limit the response length if needed
    cleaned_response = clean_response(generated_text)
    processed_response = post_process_response(cleaned_response)

    conversation_history.append({"user": user_input})
    conversation_history.append({"bot": cleaned_response})
    
    return processed_response

def request_feedback():
    # This is a simple example. You could implement a more sophisticated method.
    feedback = input("Was this response helpful? (yes/no): ")
    return feedback.lower().strip()

def log_interaction(user_input, bot_response, feedback='N/A'):
    global conversation_log
    # Append the interaction to the DataFrame
    
    conversation_log.loc[len(conversation_log.index)] = [user_input, bot_response, feedback]
    
def save_conversation_to_csv():
    # Generate a unique filename for each conversation based on the current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_{username}_{timestamp}.csv"
    
    # Save the DataFrame to a CSV file
    conversation_log.to_csv(f'conversations/{filename}', index=False)
    print(f"Conversation saved to 'conversations/{filename}'")

def chat(do_feedback):
    while True:
        input_text = input("You: ")
        if input_text == "quit":
            save_conversation_to_csv()
            break
        
        intent = recognize_intent_simple(input_text, model_simple, all_words, tags)
        if intent:
            response = handle_intent(intent, intents)
        else:
            response = generate_response_with_tinyllama(input_text)
            if response == "":
                response = handle_intent("fallback",  intents)
        
        
        print(f"{bot_name}: {response}")

        if do_feedback:
            feedback = request_feedback()
            log_interaction(input_text, response, feedback)

def chat_with_tinyllama(do_feedback):
    global conversation_history
    conversation_history = []  # Initialize conversation history
    
    while True:
        input_text = input("You: ")
        if input_text == "quit":
            save_conversation_to_csv()
            break
        
        response = generate_response_with_tinyllama(input_text)
        print(f"Chatbot: {response}")

        if do_feedback:
            feedback = request_feedback()
            log_interaction(input_text, response, feedback)

def generate_response(input_text):
    if input_text == "quit":
        save_conversation_to_csv()
        
    intent = recognize_intent_simple(input_text, model_simple, all_words, tags)
    if intent:
        response = handle_intent(intent, intents)
    else:
        response = generate_response_with_tinyllama(input_text)
        if response == "":
            response = handle_intent("fallback",  intents)
        
    return response

def prep():
    all_words, tags, xy = load_data()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_simple = create_NeuralNet()
    model_simple.load_state_dict(torch.load("data.pth")['model_state'])

if __name__ == "__main__":
    print("Loading data...")
    all_words, tags, xy = load_data()
    print("Welcome, ", username)
    print("-----------------------------------")
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Prepare simple model (NeuralNet) as before
    # Load trained simple model weights here
    model_simple = create_NeuralNet()
    model_simple.load_state_dict(torch.load("data.pth")['model_state'])

    print("Let's chat! (type 'quit' to exit)")
    conversation_log = pd.DataFrame(columns=columns)
    chat(False)
