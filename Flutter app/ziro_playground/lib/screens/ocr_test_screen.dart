import 'package:flutter/material.dart';
import 'dart:convert';
import 'package:google_mlkit_text_recognition/google_mlkit_text_recognition.dart';
import 'package:image_picker/image_picker.dart';
import 'package:http/http.dart' as http;

class OCRScreen extends StatefulWidget {
  @override
  _OCRScreenState createState() => _OCRScreenState();
}

class _OCRScreenState extends State<OCRScreen> {
  String _extractedText = 'Scan a receipt to extract text';

  void _pickImage() async {
    final ImagePicker _picker = ImagePicker();
    final XFile? image = await _picker.pickImage(source: ImageSource.camera);
    if (image != null) {
      _recognizeText(image.path);
    }
  }

  Future<void> _recognizeText(String imagePath) async {
    final inputImage = InputImage.fromFilePath(imagePath);
    final textRecognizer = TextRecognizer(script: TextRecognitionScript.latin);

    try {
      final RecognizedText recognizedText =
          await textRecognizer.processImage(inputImage);
      String text = recognizedText.text;
      sendToPC(text); // Send text to PC
      setState(() {
        _extractedText = text;
      });
    } catch (e) {
      print('Error occurred while recognizing text: $e');
    } finally {
      textRecognizer.close();
    }
  }

  void sendToPC(String text) async {
    final url = 'http://192.168.237.217:5000/upload';
    final headers = {'Content-Type': 'application/json'};
    final body = json.encode({'ocr_text': text});

    print('Sending request to $url with headers $headers and body $body');

    final response = await http.post(
      Uri.parse(url),
      headers: headers,
      body: body,
    );

    if (response.statusCode == 200) {
      final extractedEntities = json.decode(response.body);
      print('Extracted Entities: $extractedEntities');
    } else {
      print(
          'Failed to extract entities with status code: ${response.statusCode}');
      throw Exception('Failed to extract entities');
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('OCR with ML Kit'),
      ),
      body: SingleChildScrollView(
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Padding(
                padding: const EdgeInsets.all(8.0),
                child: Text(_extractedText, textAlign: TextAlign.center),
              ),
              ElevatedButton(
                onPressed: _pickImage,
                child: Text('Scan Image'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
