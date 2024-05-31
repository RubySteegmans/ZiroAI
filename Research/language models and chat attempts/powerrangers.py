import numpy as np
import torch
import pandas as pd
from datetime import datetime
import os
import json
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, pipeline
from sklearn.model_selection import train_test_split
from model import NeuralNet 
from nltk_utils import bag_of_words, nltk_tokenize
from accelerate import Accelerator

columns = ['User', 'Bot', 'Feedback']
conversation_log = pd.DataFrame(columns=columns)
X_train = []
y_train = []
all_words = []
tags = []
xy = []
chat_history_ids = None
bot_name = "Ziro"
username = "Ruby"

def create_NeuralNet():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the trained simple model
    FILE = "data.pth"
    data = torch.load(FILE, map_location=device)  # Ensure model is loaded to the correct device

    model_simple = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
    model_simple.load_state_dict(data["model_state"])
    model_simple.eval()
    return model_simple

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

def recognize_intent_simple(input_text, model, all_words, tags):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    prob = probs[0][predicted.item()].item()  # Convert to Python float

    # Set a confidence threshold. Adjust this based on your dataset and needs.
    confidence_threshold = 0.75
    if prob > confidence_threshold:
        return tag, prob
    else:
        return None, prob

def handle_intent(intent, intents_json):
    for intent_config in intents_json['intents']:
        if intent == intent_config["tag"]:
            return random.choice(intent_config['responses'])
    return "I do not understand..."

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

def generate_response_distilbert(input_text):
    inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=128).to(device)
    outputs = model(**inputs)
    response_id = torch.argmax(outputs.logits, dim=-1).item()
    # Map the response_id to a predefined response or further generate it based on your application
    return "Your generic response or further processing here"

def chat_with_feedback(feedback):
    while True:
        input_text = input("You: ")
        if input_text.lower() == "quit":
            save_conversation_to_csv()
            break

        intent, confidence = recognize_intent_simple(input_text, model_simple, all_words, tags)
        if confidence >= 85:
            response = handle_intent(intent, intents)
        else:
            response = generate_response_distilbert(input_text)
        
        print(f"{bot_name}: {response}")
        if feedback:
            feedback = request_feedback()
            log_interaction(input_text, response, feedback)

if __name__ == "__main__":
    print("Loading data...")
    all_words, tags, xy = load_data()
    print("Operating Pangea, all systems go")
    print("Welcome, user")
    print("-----------------------------------")
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert/distilbert-base-uncased')
    model = DistilBertForSequenceClassification.from_pretrained('distilbert/distilbert-base-uncased')
    model.to(device)

    # Prepare simple model (NeuralNet) as before
    # Load trained simple model weights here
    model_simple = create_NeuralNet()
    model_simple.load_state_dict(torch.load("data.pth")['model_state'])

    accelerator = Accelerator()
    model, tokenizer = accelerator.prepare(model, tokenizer)

    print("Let's chat! (type 'quit' to exit)")
    conversation_log = pd.DataFrame(columns=columns)
    chat_with_feedback(False)