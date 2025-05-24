# Face Authentication System

A secure face authentication system built with Python, OpenCV, and MySQL.

## Features

- Face detection and recognition
- Anti-spoofing protection
- Location-based authentication
- MySQL database integration
- Streamlit web interface

## Prerequisites

- Python 3.8+
- MySQL Server
- Webcam

## Installation

1. Clone the repository:
```bash
git clone <your-repo-url>
cd face_auth_system
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up MySQL database:
```sql
CREATE DATABASE face_auth_new;
CREATE USER 'faceuser'@'localhost' IDENTIFIED BY '1234';
GRANT ALL PRIVILEGES ON face_auth_new.* TO 'faceuser'@'localhost';
FLUSH PRIVILEGES;
```

## Usage

1. Start the application:
```bash
streamlit run app.py
```

2. Open your browser and navigate to `http://localhost:8501`

## Project Structure

```
face_auth_system/
├── app.py                  # Main Streamlit application
├── database/
│   ├── __init__.py
│   └── db_manager.py       # MySQL database operations
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

## License

MIT License 