# Face Authentication System

A secure face authentication system built with Python, OpenCV, and SQLite.

## Features

- Face detection and recognition
- Anti-spoofing protection
- Location-based authentication
- SQLite database integration (offline)
- Streamlit web interface

## Prerequisites

- Python 3.8+
- Webcam

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ayush021-Dev/Face_Authenticaion_App.git
cd Face_Authenticaion_App
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. The SQLite database file (`face_auth.db`) will be automatically created when you run the application for the first time.

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
Face_Authenticaion_App/
├── app.py                  # Main Streamlit application
├── database/
│   ├── __init__.py
│   └── db_manager.py       # SQLite database operations
├── face_utils/
│   ├── __init__.py
│   ├── detector.py         # Face detection
│   ├── recognizer.py       # Face recognition
│   └── anti_spoof.py       # Anti-spoofing module
├── utils/
│   ├── __init__.py
│   └── location.py         # Get geolocation data
├── models/                 # Store trained models
├── data/                   # Store face embeddings
└── requirements.txt        # Dependencies
```

