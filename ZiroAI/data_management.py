# data_management.py

import pandas as pd
from datetime import datetime

def init_conversation_log(columns):
    return pd.DataFrame(columns=columns)

def log_interaction(conversation_log, user_input, bot_response, feedback='N/A'):
    conversation_log.loc[len(conversation_log.index)] = [user_input, bot_response, feedback]
    return conversation_log

def save_conversation_to_csv(conversation_log, username):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"conversation_{username}_{timestamp}.csv"
    conversation_log.to_csv(f'conversations/{filename}', index=False)
    return f"Conversation saved to 'conversations/{filename}'"
