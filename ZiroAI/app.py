from flask import Flask, request, jsonify
from main_chat import handle_chat_interaction
import werkzeug
import os
from speech_recognition import transcribe_audio, load_model

app = Flask(__name__)

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get('message')
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    response = handle_chat_interaction(user_input)
    return jsonify({'response': response})

@app.route('/voice', methods=['POST'])
def handle_voice_input():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    filename = werkzeug.utils.secure_filename(audio_file.filename)
    audio_path = os.path.join('audio', filename)
    audio_file.save(audio_path)
    
    processor, model = load_model()

    # Assuming your transcription function and models are accessible
    transcription = transcribe_audio(audio_path, processor, model)
    os.remove(audio_path)  # Clean up the stored file after processing

    # Here, you could pass the transcription to your chat function
    response = handle_chat_interaction(transcription)
    return jsonify({'transcription': transcription, 'response': response})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)