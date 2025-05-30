# face_rec/face_utils/detector.py
import cv2
import os
import urllib.request

class FaceDetector:
    def __init__(self):
        # Initialize Haar Cascade classifier for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try to load DNN model for better detection
        self.net = None
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_file = os.path.join(model_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        config_file = os.path.join(model_dir, "deploy.prototxt")
        
        # Download files if they don't exist
        if not os.path.exists(model_file):
            print("Downloading face detection model...")
            try:
                urllib.request.urlretrieve(
                    "https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20180205_fp16/res10_300x300_ssd_iter_140000_fp16.caffemodel",
                    model_file
                )
            except Exception as e:
                print(f"Could not download model: {e}")
        
        if not os.path.exists(config_file):
            print("Downloading model configuration...")
            try:
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    config_file
                )
            except Exception as e:
                print(f"Could not download configuration: {e}")
        
        # Load DNN model if available
        if os.path.exists(model_file) and os.path.exists(config_file):
            try:
                self.net = cv2.dnn.readNet(model_file, config_file)
                print("DNN face detector loaded successfully")
            except Exception as e:
                print(f"Could not load DNN model: {e}")
    
    def detect_face(self, frame):
        """
        Detect faces in the frame using Haar Cascade
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face locations (x, y, w, h)
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
        return faces
    
    def detect_face_dnn(self, frame):
        """
        Detect faces using DNN if available, otherwise use Haar Cascade
        
        Args:
            frame: Input image
            
        Returns:
            List of face locations (x, y, w, h)
        """
        if self.net is None:
            return self.detect_face(frame)
        
        # Use DNN for better detection
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
        self.net.setInput(blob)
        detections = self.net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.4:
                x1 = int(detections[0, 0, i, 3] * width)
                y1 = int(detections[0, 0, i, 4] * height)
                x2 = int(detections[0, 0, i, 5] * width)
                y2 = int(detections[0, 0, i, 6] * height)
                faces.append((x1, y1, x2-x1, y2-y1))
        
        return faces
    
    def convert_rect_format(self, x, y, w, h):
        """
        Convert (x,y,w,h) format to (top, right, bottom, left) format
        
        Args:
            x, y, w, h: Rectangle coordinates
            
        Returns:
            (top, right, bottom, left) tuple
        """
        return (y, x+w, y+h, x)