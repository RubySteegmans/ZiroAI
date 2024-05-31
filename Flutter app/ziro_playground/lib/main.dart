// File: lib/main.dart
import 'package:flutter/material.dart';
import 'screens/ocr_test_screen.dart'; // Import OCR screen

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'OCR App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: OCRScreen(), // Set OCRScreen as the home widget
    );
  }
}
