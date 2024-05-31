import re
from flask import Flask, request, jsonify
import spacy
import logging

# Initialize the app and load the NER model
app = Flask(__name__)
nlp = spacy.load("best_ner_model")

logging.basicConfig(level=logging.DEBUG)

@app.route('/extract_entities', methods=['POST'])
def extract_entities():
    try:
        data = request.get_json()
        ocr_text = data['ocr_text']
        logging.debug(f"Received request with data: {data}")

        # Preprocess the OCR text
        cleaned_text = preprocess_text(ocr_text)
        logging.debug(f"Cleaned OCR text: {cleaned_text}")

        # Process the cleaned text with the NER model
        doc = nlp(cleaned_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        logging.debug(f"Extracted entities: {entities}")

        # Post-process the extracted entities
        cleaned_entities = post_process_entities(entities)
        return jsonify({"entities": cleaned_entities})

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": str(e)}), 500

def preprocess_text(text):
    # Replace newline characters with spaces
    text = text.replace('\n', ' ')
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    
    # Handle common OCR artifacts (e.g., misplaced numbers or punctuation)
    text = re.sub(r'(\d)\s+(\d)', r'\1\2', text)
    text = re.sub(r'(\w)\s+(\w)', r'\1 \2', text)  # Ensure words are separated by single space
    
    return text

def post_process_entities(entities):
    # Initialize dictionaries to store final entities
    items = []
    total = None
    timestamp = None

    for text, label in entities:
        if label == 'ITEM':
            items.append(text.strip())
        elif label == 'TOTAL':
            total = text.strip()
        elif label == 'TIMESTAMP':
            timestamp = text.strip()
        # Add other entity types as necessary

    # Create a dictionary to hold the final entities
    processed_entities = {
        "items": items,
        "total": total,
        "timestamp": timestamp
    }
    return processed_entities

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
