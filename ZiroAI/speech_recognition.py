from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
from datasets import load_dataset

def load_model():
    # Load the Whisper model and processor
    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    return processor, model

def transcribe_audio(audio_path, processor, model):
    # Load audio
    waveform, sample_rate = torchaudio.load(audio_path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    if waveform.ndim > 1:
        waveform = waveform.squeeze(0)

    # Process audio
    inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt")

    # Generate tokens
    predicted_ids = model.generate(inputs.input_features)

    # Decode tokens to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    return transcription[0]