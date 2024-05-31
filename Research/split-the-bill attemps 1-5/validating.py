import json
with open('dataset.json', 'r') as f:
    receipts = json.load(f)
    
def find_overlapping_entities(data):
    overlapping_entities = []
    for receipt in data:
        text = receipt["text"]
        entities = receipt["entities"]
        spans = []
        for entity in entities:
            start_idx = text.find(entity[0])
            if start_idx != -1:
                end_idx = start_idx + len(entity[0])
                spans.append((start_idx, end_idx, entity[1]))
        spans = sorted(spans, key=lambda x: x[0])
        for i in range(len(spans) - 1):
            if spans[i][1] > spans[i + 1][0]:
                overlapping_entities.append((text, spans[i], spans[i + 1]))
    return overlapping_entities

overlapping_entities = find_overlapping_entities(receipts)
if overlapping_entities:
    print("Overlapping entities found:")
    for text, ent1, ent2 in overlapping_entities:
        print(f"Text: {text}")
        print(f"Overlap between: {ent1} and {ent2}")
else:
    print("No overlapping entities found.")

def adjust_overlapping_entities(data):
    adjusted_data = []
    for receipt in data:
        text = receipt["text"]
        entities = receipt["entities"]
        spans = []
        for entity in entities:
            start_idx = text.find(entity[0])
            if start_idx != -1:
                end_idx = start_idx + len(entity[0])
                spans.append((start_idx, end_idx, entity[1]))
        spans = sorted(spans, key=lambda x: x[0])
        adjusted_spans = []
        for i in range(len(spans)):
            if i > 0 and spans[i][0] < adjusted_spans[-1][1]:
                # Adjust or remove overlapping entity
                if spans[i][1] <= adjusted_spans[-1][1]:
                    # If the current entity is fully within the previous one, ignore it
                    continue
                else:
                    # Adjust the start of the current entity to avoid overlap
                    spans[i] = (adjusted_spans[-1][1], spans[i][1], spans[i][2])
            adjusted_spans.append(spans[i])
        adjusted_entities = [(text[start:end], label) for start, end, label in adjusted_spans]
        adjusted_data.append({"text": text, "entities": adjusted_entities})
    return adjusted_data

receipts = adjust_overlapping_entities(receipts)

# Verify again if there are any overlapping entities
overlapping_entities = find_overlapping_entities(receipts)
if overlapping_entities:
    print("Overlapping entities found:")
    for text, ent1, ent2 in overlapping_entities:
        print(f"Text: {text}")
        print(f"Overlap between: {ent1} and {ent2}")
else:
    print("No overlapping entities found.")
