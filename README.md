FastAPI Invoice OCR API

This project is a FastAPI application that processes invoices using OCR and 
integrates with Google Cloud Vision API to extract key information from PDF 
and image files. It extracts data such as invoice numbers, customer details, 
and consumption values and provides an HTML-based web interface.

Features

- OCR Processing: Extracts text from PDF and image files using Google Cloud Vision API.
- Flexible Data Extraction: Identifies and processes key information such as invoices, contracts, customer details, and energy consumption data.
- HTML Interface: Provides a web interface for users to upload files and view processing results.
- Supported File Formats: .pdf, .jpeg, .jpg, .png.

Dependencies



- Google Cloud Vision API
  - Google Cloud Vision API is used for OCR processing.
  - You must create a service account in your Google Cloud account and download the API key.

- Python Packages
  - The following Python packages are required:
    - fastapi: Required for creating the FastAPI web application.
    - jinja2: Used as the HTML templating engine.
    - pdf2image: Converts PDF files into images. 
    - google-cloud-vision: Communicates with the Google Vision API.
    - uvicorn: ASGI server used to run the FastAPI application.

  - To install all required packages:
    pip install fastapi jinja2 pdf2image google-cloud-vision uvicorn



Installation

1. Ensure Python 3.8+ is installed.
2. Install the required Python packages:
   pip install fastapi jinja2 pdf2image google-cloud-vision uvicorn
3. Google Cloud Vision API Setup:
   - Create a service account in Google Cloud Console.
   - Download your API key (.json) file.

Configuration

1. Set up the path to your service account key in the application for Google Cloud Vision:
  path/for/.json/key



Running the Application

1. To run the FastAPI application:
   uvicorn invoice_api_web:app --reload
2. Access the application at: http://127.0.0.1:8000

Usage

Users can upload .pdf, .jpeg, .jpg, or .png files on the home page and view OCR processing results in HTML format.

Project Structure

project-root/
├── invoice_api_web.py              
├── templates/    
├── static          
└── README.txt           
