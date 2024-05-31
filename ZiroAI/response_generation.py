# response_generation.py

from transformers import pipeline

def setup_pipeline(model_name, torch_dtype, device_map):
    return pipeline("text-generation", model=model_name, torch_dtype=torch_dtype, device_map=device_map)

def generate_response(model, user_input):
    response = model(f"{user_input}", max_new_tokens=20, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
    return response[0].get("generated_text", "").strip()
