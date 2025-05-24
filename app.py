# face_rec/app.py
import streamlit as st
import cv2
import numpy as np
import pickle
from datetime import datetime
import time
import os

from database.db_manager import DatabaseManager
from face_utils.detector import FaceDetector
from face_utils.recognizer import FaceRecognizer
from face_utils.anti_spoof import AntiSpoofing
from utils.location import get_location, is_within_area

# Set page configuration
st.set_page_config(
    page_title="Face Authentication System",
    page_icon="🔐",
    layout="wide"
)

# Initialize components
@st.cache_resource
def load_resources():
    db_manager = DatabaseManager()
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    anti_spoof = AntiSpoofing()
    return db_manager, face_detector, face_recognizer, anti_spoof

db_manager, face_detector, face_recognizer, anti_spoof = load_resources()

# Load existing employees
employees = db_manager.get_all_employees()
if employees:
    face_recognizer.load_from_database(employees)

# App title
st.title("🔐 Face Authentication System")

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Go to", ["Home", "Sign Up", "Login"])

if option == "Home":
    st.header("Welcome to Face Authentication System")
    st.write("Use the sidebar to navigate to Sign Up or Login.")
    
    # Add project description
    st.markdown("""
    ### Features:
    - Facial recognition-based authentication
    - Anti-spoofing protection against photo and screen attacks
    - Employee registration with face scan
    - Login tracking with timestamp and location
    - Geofencing to restrict logins to defined areas
    
    ### Security Features:
    - Advanced anti-spoofing using multiple techniques
    - Multi-factor authentication
    - Secure database storage
    - Location-based access control
    """)
    
elif option == "Sign Up":
    st.header("Employee Registration")
    
    # Input fields
    emp_id = st.text_input("Employee ID")
    name = st.text_input("Employee Name")
    
    # Face capture section
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Instructions:")
        st.info("1. Look directly at the camera\n2. Ensure good lighting\n3. Remove glasses or obstructions\n4. Keep a neutral expression")
        
        if st.button("Capture Face"):
            if not emp_id or not name:
                st.error("Please provide Employee ID and Name before capturing face")
            else:
                progress = st.progress(0)
                status_text = st.empty()
                
                status_text.text("Starting camera...")
                cap = cv2.VideoCapture(0)
                
                # Warm up the camera
                for i in range(10):
                    cap.read()
                    progress.progress(i * 0.05)
                    time.sleep(0.1)
                
                status_text.text("Looking for face...")
                face_encodings = []
                frames_with_face = 0
                max_frames = 10
                
                # Capture multiple frames for better accuracy
                while frames_with_face < max_frames:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("Could not access camera")
                        break
                        
                    # Display the frame in the second column
                    with col2:
                        st.image(frame, channels="BGR", caption="Camera Feed", use_column_width=True)
                    
                    # Find faces using DNN
                    faces = face_detector.detect_face_dnn(frame)
                    
                    if faces:
                        # Get the first face
                        x, y, w, h = faces[0]
                        face_location = face_detector.convert_rect_format(x, y, w, h)
                        
                        # Check for spoofing
                        is_real = anti_spoof.check_liveness(frame, face_location)
                        
                        if is_real:
                            # Extract face region
                            face_img = frame[y:y+h, x:x+w]
                            
                            # Get encoding
                            encoding = face_recognizer.get_face_encoding(face_img)
                            face_encodings.append(encoding)
                            frames_with_face += 1
                            progress.progress(0.5 + frames_with_face * 0.05)
                            status_text.text(f"Captured frame {frames_with_face}/{max_frames}")
                        else:
                            status_text.text("⚠️ Spoof detected! Please use a real face.")
                            time.sleep(1)
                    
                    time.sleep(0.2)
                
                cap.release()
                
                if len(face_encodings) >= 5:
                    # Average the encodings for more robust representation
                    avg_encoding = np.mean(face_encodings, axis=0)
                    
                    # Check if face already exists in database
                    face_exists, existing_id, existing_name = db_manager.face_exists(avg_encoding)
                    
                    if face_exists:
                        progress.progress(1.0)
                        status_text.text("Face already registered!")
                        st.error(f"This face is already registered as {existing_name} (ID: {existing_id})")
                    else:
                        # Save to database
                        # Serialize the face encoding
                        serialized_encoding = pickle.dumps(avg_encoding)
                        if db_manager.register_employee(emp_id, name, serialized_encoding):
                            # Update recognizer
                            face_recognizer.add_face(avg_encoding, emp_id)
                            
                            progress.progress(1.0)
                            status_text.text("Registration successful!")
                            st.success(f"Employee {name} registered successfully!")
                            
                            # Show area definition section
                            st.subheader("Define Allowed Login Area")
                            st.info("This will restrict login to a specific geographic area")
                            
                            # Get current location as default
                            default_lat, default_lon = get_location()
                            
                            area_lat = st.number_input("Area Center Latitude", 
                                                    value=default_lat if default_lat else 0.0,
                                                    format="%.6f")
                            area_lon = st.number_input("Area Center Longitude", 
                                                    value=default_lon if default_lon else 0.0,
                                                    format="%.6f")
                            area_radius = st.number_input("Allowed Radius (km)", 
                                                        min_value=0.1, value=1.0, step=0.1)
                            
                            if st.button("Set Area Restriction"):
                                if db_manager.update_employee_area(emp_id, area_lat, area_lon, area_radius):
                                    st.success(f"Area restriction set for {name}!")
                                    st.info(f"Login will only be allowed within {area_radius} km of {area_lat}, {area_lon}")
                                    st.markdown(f"[View on Map](https://www.google.com/maps?q={area_lat},{area_lon})")
                                else:
                                    st.error("Failed to set area restriction. Please try again.")
                        else:
                            st.error("Failed to register employee. Please try again.")
                else:
                    st.error("Could not capture enough clear face images. Please try again.")

elif option == "Login":
    st.header("Employee Login")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Instructions:")
        st.info("1. Look directly at the camera\n2. Ensure good lighting\n3. Position your face in the frame")
        
        if st.button("Verify Face"):
            progress = st.progress(0)
            status_text = st.empty()
            
            status_text.text("Starting camera...")
            cap = cv2.VideoCapture(0)
            
            # Warm up camera
            for i in range(5):
                cap.read()
                progress.progress(i * 0.1)
                time.sleep(0.1)
            
            status_text.text("Looking for face...")
            login_successful = False
            recognized_emp_id = None
            
            # Try for a few seconds
            start_time = time.time()
            while time.time() - start_time < 10 and not login_successful:
                ret, frame = cap.read()
                if not ret:
                    st.error("Could not access camera")
                    break
                
                # Find faces using DNN
                faces = face_detector.detect_face_dnn(frame)
                
                # Display the frame in the second column
                with col2:
                    if faces:
                        # Draw rectangle around face
                        for (x, y, w, h) in faces:
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    st.image(frame, channels="BGR", caption="Camera Feed", use_column_width=True)
                
                if faces:
                    # Get the first face
                    x, y, w, h = faces[0]
                    face_location = face_detector.convert_rect_format(x, y, w, h)
                    
                    # Check for spoofing
                    is_real = anti_spoof.check_liveness(frame, face_location)
                    
                    if is_real:
                        # Extract face region
                        face_img = frame[y:y+h, x:x+w]
                        
                        # Get encoding
                        encoding = face_recognizer.get_face_encoding(face_img)
                        
                        # Try to recognize
                        recognized_emp_id = face_recognizer.recognize_face(encoding)
                        if recognized_emp_id:
                            login_successful = True
                            status_text.text("Face verified!")
                            break
                        else:
                            status_text.text("Face not recognized")
                    else:
                        status_text.text("⚠️ Spoof detected! Please use a real face.")
                
                progress.progress(min(1.0, (time.time() - start_time) / 10))
                time.sleep(0.1)
            
            cap.release()
            
            if login_successful:
                # Get location
                latitude, longitude = get_location()
                
                # Get employee details including area info
                employee = db_manager.get_employee(recognized_emp_id)
                
                # Check if within allowed area
                area_lat = employee[3]  # index 3 for area_lat
                area_lon = employee[4]  # index 4 for area_lon
                area_radius = employee[5]  # index 5 for area_radius
                
                if is_within_area(latitude, longitude, area_lat, area_lon, area_radius):
                    # Log the login
                    db_manager.log_login(recognized_emp_id, latitude, longitude)
                    
                    progress.progress(1.0)
                    with col1:
                        st.success(f"Welcome, {employee[1]}!")
                        st.write(f"Logged in at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        if latitude and longitude:
                            st.write(f"Location: {latitude}, {longitude}")
                            st.markdown(f"[View on Map](https://www.google.com/maps?q={latitude},{longitude})")
                else:
                    progress.progress(1.0)
                    st.error("Login denied: You are not in the authorized location.")
                    st.info(f"Your location: {latitude}, {longitude}")
                    if area_lat and area_lon:
                        st.info(f"Authorized area: Within {area_radius} km of {area_lat}, {area_lon}")
                        st.markdown(f"[View Authorized Area](https://www.google.com/maps?q={area_lat},{area_lon})")
            else:
                st.error("Login failed. Please try again.")

# Add footer
st.markdown("---")
st.markdown("Face Authentication System | Built with OpenCV & Streamlit")