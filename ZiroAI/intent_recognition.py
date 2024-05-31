# intent_recognition.py

import numpy as np
import torch
import json
import random
from nltk_utils import bag_of_words, nltk_tokenize
from model import NeuralNet

def load_model(model_path, device):
    data = torch.load(model_path, map_location=device)
    model = NeuralNet(data["input_size"], data["hidden_size"], data["output_size"]).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model

def recognize_intent(input_text, model, all_words, tags, device):
    input_text = input_text if isinstance(input_text, str) else str(input_text)
    sentence = nltk_tokenize(input_text)
    X = bag_of_words(sentence, all_words)
    X = torch.from_numpy(X).reshape(1, -1).float().to(device)
    output = model(X)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    return tag if prob.item() > 0.75 else None

def handle_intent(intent):
    with open('intents.json', 'r') as f:
        intents = json.load(f)

    for intent_config in intents['intents']:
        if intent == intent_config["tag"]:
            return random.choice(intent_config['responses'])
    return "I do not understand..."