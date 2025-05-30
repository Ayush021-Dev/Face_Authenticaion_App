import cv2
import os
import time
import numpy as np
from datetime import datetime

def collect_training_data():
    # Create directories for real and fake faces
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "training_data")
    real_dir = os.path.join(base_dir, "real")
    fake_dir = os.path.join(base_dir, "fake")
    
    os.makedirs(real_dir, exist_ok=True)
    os.makedirs(fake_dir, exist_ok=True)
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Initialize camera with specific settings
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer size
    
    # Warm up the camera
    for _ in range(5):
        cap.read()
        time.sleep(0.1)
    
    print("\n=== Training Data Collection ===")
    print("Press 'r' to capture a real face")
    print("Press 'f' to capture a fake face (show photo/video)")
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read from camera")
            break
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw rectangle around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Training Data Collection', frame)
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('r') and len(faces) > 0:
            # Capture real face
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"real_{timestamp}.jpg"
            cv2.imwrite(os.path.join(real_dir, filename), face_img)
            print(f"Saved real face: {filename}")
            
        elif key == ord('f') and len(faces) > 0:
            # Capture fake face
            x, y, w, h = faces[0]
            face_img = frame[y:y+h, x:x+w]
            
            # Save image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fake_{timestamp}.jpg"
            cv2.imwrite(os.path.join(fake_dir, filename), face_img)
            print(f"Saved fake face: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    # Print summary
    real_count = len(os.listdir(real_dir))
    fake_count = len(os.listdir(fake_dir))
    print(f"\nCollection complete!")
    print(f"Real faces collected: {real_count}")
    print(f"Fake faces collected: {fake_count}")

if __name__ == "__main__":
    collect_training_data() 