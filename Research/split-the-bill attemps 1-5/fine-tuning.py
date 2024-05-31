import json
import pandas as pd
import spacy
from spacy.training.example import Example
from flask import Flask, request, jsonify

# Load the dataset
with open('dataset.json') as f:
    dataset = json.load(f)

# Create a DataFrame from the dataset
df = pd.DataFrame(dataset)

# Function to check and fix overlapping entities
def resolve_overlaps(entities):
    sorted_entities = sorted(entities, key=lambda x: x[0])
    non_overlapping_entities = []

    for start, end, label in sorted_entities:
        if not non_overlapping_entities:
            non_overlapping_entities.append((start, end, label))
        else:
            last_start, last_end, last_label = non_overlapping_entities[-1]
            if start >= last_end:  # No overlap
                non_overlapping_entities.append((start, end, label))
            else:
                # Resolve overlap by keeping the longer entity
                if end > last_end:
                    non_overlapping_entities[-1] = (start, end, label)

    return non_overlapping_entities

# Function to align entities with text
def align_entities(text, entities):
    aligned_entities = []
    for entity in entities:
        start = text.find(entity[0])
        if start != -1:
            end = start + len(entity[0])
            aligned_entities.append((start, end, entity[1]))
    return aligned_entities

# Convert the entities into the required format for spaCy
def convert_to_spacy_format(df):
    training_data = []
    for _, row in df.iterrows():
        text = row['text']
        entities = []
        for entity in row['entities']:
            entity_text = entity[0]
            start = text.find(entity_text)
            if start != -1:
                end = start + len(entity_text)
                entities.append((start, end, entity[1]))
        # Resolve any overlapping entities
        entities = resolve_overlaps(entities)
        training_data.append((text, {'entities': entities}))
    return training_data

training_data = convert_to_spacy_format(df)

# Load the pre-trained spaCy model
nlp = spacy.load("en_core_web_sm")

# Disable other pipelines to only train NER
nlp.disable_pipes(*[pipe for pipe in nlp.pipe_names if pipe != "ner"])

# Add new entities labels to the existing NER pipeline
ner = nlp.get_pipe("ner")
for _, annotations in training_data:
    for ent in annotations.get("entities"):
        ner.add_label(ent[2])

# Convert training data to spaCy format
def create_training_examples(data):
    examples = []
    for text, annotations in data:
        examples.append(Example.from_dict(nlp.make_doc(text), annotations))
    return examples

# Create training examples
examples = create_training_examples(training_data)

# Initialize training
optimizer = nlp.resume_training()
for epoch in range(10):  # Adjust the number of epochs as needed
    losses = {}
    nlp.update(examples, drop=0.35, sgd=optimizer, losses=losses)
    print(f"Epoch {epoch}, Losses: {losses}")

# Save the trained model
nlp.to_disk("trained_ner_model")