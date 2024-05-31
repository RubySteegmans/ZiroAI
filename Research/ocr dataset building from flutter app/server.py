from flask import Flask, request
import csv

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_text():
    text_data = request.data.decode('utf-8')
    # Define the CSV file path
    csv_file_path = 'ocr.csv'
    # Open the CSV file in append mode
    with open(csv_file_path, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # Write the data along with a timestamp
        writer.writerow([text_data])
    return 'Data written to file', 200

@app.route('/')
def home():
    return "Hello, World!"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)