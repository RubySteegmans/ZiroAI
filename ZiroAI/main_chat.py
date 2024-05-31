# main_chat.py

from intent_recognition import load_model, recognize_intent, handle_intent
from response_generation import setup_pipeline, generate_response
from data_management import init_conversation_log, log_interaction, save_conversation_to_csv
from api_client import ZiroPayAPI
import torch

def main():
    print("Initializing ZiroPay Chat Assistant")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model("data.pth", device)
    chat_pipeline = setup_pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch.bfloat16, "auto")
    conversation_log = init_conversation_log(['User', 'Bot', 'Feedback'])

    # Example interaction loop
    while True:
        user_input = input("You: ")
        if user_input == "quit":
            print(save_conversation_to_csv(conversation_log, "Ruby"))
            break

        # API example: Fetch user profile
        if user_input.startswith("fetch profile"):
            user_id = user_input.split()[-1]  # Assuming last part of input is user_id
            user_profile = api_client.fetch_user_profile(user_id)
            print("User Profile:", user_profile)
            continue

        # Intent recognition and response generation
        intent = recognize_intent(user_input, model, all_words, tags, device)
        if intent:
            response = handle_intent(intent);
        else:
            response = generate_response(chat_pipeline, user_input)

        print(f"Ziro: {response}")

        # Optional feedback collection
        feedback = input("Was this response helpful? (yes/no): ")
        conversation_log = log_interaction(conversation_log, user_input, response, feedback)

def handle_chat_interaction(user_input):
    # Initialize your models and other setups just once
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model("data.pth", device)
    chat_pipeline = setup_pipeline("TinyLlama/TinyLlama-1.1B-Chat-v1.0", torch.bfloat16, "auto")

    FILE = "data.pth"
    data = torch.load(FILE, map_location=device)

    all_words = data["all_words"]
    tags = data["tags"]
    xy = data["xy"]

    # Assuming 'user_input' is the text from the user
    intent = recognize_intent(user_input, model, all_words, tags, device)
    if intent:
        response = handle_intent(intent)
    else:
        response = generate_response(chat_pipeline, user_input)
    return response

if __name__ == "__main__":
    main()
