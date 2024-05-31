import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def chat():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-large")
    model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-large").to(device)

    chat_history_ids = None
    print("Chatbot is ready to talk! Type 'quit' to exit.")

    while True:
        input_text = input("You: ")
        if input_text.lower() == "quit":
            break

        # Encode and concatenate new user input
        new_user_input_ids = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors='pt').to(device)
        bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if chat_history_ids is not None else new_user_input_ids

        # Generate a response
        chat_history_ids = model.generate(bot_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

        # Decode and print the response
        response = tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True)
        print("Chatbot:", response)

if __name__ == "__main__":
    chat()
