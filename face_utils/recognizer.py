# face_rec/face_utils/recognizer.py
import cv2
import numpy as np
import pickle
import os
import urllib.request

class FaceRecognizer:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_ids = []
        
        # Load OpenCV's face recognition model
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = os.path.join(model_dir, "openface_nn4.small2.v1.t7")
        if not os.path.exists(model_file):
            print("Downloading face recognition model...")
            try:
                urllib.request.urlretrieve(
                    "https://github.com/pyannote/pyannote-data/raw/master/openface.nn4.small2.v1.t7",
                    model_file
                )
            except Exception as e:
                print(f"Could not download face recognition model: {e}")
        
        # Load the model
        try:
            self.face_net = cv2.dnn.readNetFromTorch(model_file)
            print("Face recognition model loaded successfully")
        except Exception as e:
            print(f"Could not load face recognition model: {e}")
            self.face_net = None
    
    def get_face_encoding(self, face_img):
        """
        Get face encoding using OpenCV DNN
        
        Args:
            face_img: Face image (cropped to face region)
            
        Returns:
            128-dimensional face encoding
        """
        if self.face_net is None:
            # Return random encoding if model not available (for testing)
            return np.random.rand(128).astype(np.float32)
        
        # Resize and preprocess face
        face_blob = cv2.dnn.blobFromImage(
            face_img, 1.0/255, (96, 96), (0, 0, 0), swapRB=True, crop=False
        )
        
        # Get face embedding
        self.face_net.setInput(face_blob)
        encoding = self.face_net.forward()
        
        return encoding.flatten()
    
    def add_face(self, face_encoding, emp_id):
        """
        Add a face encoding with its associated employee ID
        
        Args:
            face_encoding: Face encoding numpy array
            emp_id: Employee ID string
        """
        self.known_face_encodings.append(face_encoding)
        self.known_face_ids.append(emp_id)
    
    def recognize_face(self, face_encoding, tolerance=0.6):
        """
        Recognize a face by comparing with known faces
        
        Args:
            face_encoding: Face encoding to recognize
            tolerance: Threshold for face comparison (lower is stricter)
            
        Returns:
            Employee ID if recognized, None otherwise
        """
        if len(self.known_face_encodings) == 0:
            return None
        
        # Compare face with known faces using Euclidean distance
        distances = []
        for known_encoding in self.known_face_encodings:
            # Calculate Euclidean distance
            dist = np.linalg.norm(known_encoding - face_encoding)
            distances.append(dist)
        
        min_distance_idx = np.argmin(distances)
        min_distance = distances[min_distance_idx]
        
        # If distance is less than tolerance, return the employee ID
        if min_distance < tolerance:
            return self.known_face_ids[min_distance_idx]
        
        return None
    
    def load_from_database(self, employees):
        """
        Load face encodings from database records
        
        Args:
            employees: List of tuples (emp_id, name, face_encoding)
        """
        self.known_face_encodings = []
        self.known_face_ids = []
        
        for emp_id, name, face_encoding in employees:
            # Decode the face encoding from bytes to numpy array
            try:
                decoded_encoding = pickle.loads(face_encoding)
                self.add_face(decoded_encoding, emp_id)
            except Exception as e:
                print(f"Error loading encoding for {emp_id}: {e}")