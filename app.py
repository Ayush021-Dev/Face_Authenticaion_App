# face_rec/app.py
import subprocess
import sys
import streamlit as st
import cv2
import numpy as np
import pickle
from datetime import datetime
import time
import os
import warnings
import tracemalloc

# Suppress specific warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*coroutine.*')

# Enable tracemalloc
tracemalloc.start()

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

# Check for redirection after successful login
if st.session_state.get('redirect_to_logbook'):
    st.session_state.redirect_to_logbook = False  # Reset the flag
    st.switch_page("pages/logbook.py")

# Initialize components
@st.cache_resource
def load_resources():
    db_manager = DatabaseManager()
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    anti_spoof = AntiSpoofing()
    
    # Create default admin if no admins exist
    try:
        db_manager.add_admin("admin", "System Administrator", "admin123")
    except:
        pass  # Admin might already exist
    
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
option = st.sidebar.radio("Go to", ["Home", "Areas", "Sign Up", "Login"])

if option == "Home":
    st.header("Welcome to Face Authentication System")
    st.write("Use the sidebar to navigate to Equipment Areas, Sign Up, or Login.")
    
    # Add project description
    st.markdown("""
    ### Features:
    - Facial recognition-based authentication
    - Anti-spoofing protection against photo and screen attacks
    - Equipment area-based access control
    - Employee registration with face scan
    - Login tracking with timestamp and location
    
    ### Security Features:
    - Advanced anti-spoofing using multiple techniques
    - Multi-factor authentication
    - Secure database storage
    - Equipment area-based access control
    """)

elif option == "Areas":
    st.header("Manage Equipment Areas")
    
    # Admin Authentication
    st.subheader("Admin Authentication")
    with st.form(key="admin_auth_form"):
        admin_id = st.text_input("Admin ID", help="Enter your admin ID")
        admin_password = st.text_input("Admin Password", type="password", help="Enter your admin password")
        
        if st.form_submit_button("Login"):
            if db_manager.verify_admin(admin_id, admin_password):
                st.session_state.admin_authenticated = True
                st.session_state.admin_id = admin_id
                st.success("Admin authentication successful!")
                st.rerun()
            else:
                st.error("Invalid admin credentials!")
    
    # Only show equipment management if admin is authenticated
    if st.session_state.get('admin_authenticated', False):
        st.success(f"Logged in as Admin: {st.session_state.admin_id}")
        
        # Create tabs for Equipment Areas and Employee Management
        tab1, tab2 = st.tabs(["Areas", "Employee Management"])
        
        with tab1:
            # Add new equipment area
            st.subheader("Add New Area")
            with st.form(key="equipment_area_form"):
                col1, col2 = st.columns(2)
                
                with col1:
                    equipment_name = st.text_input("Area Name", help="Enter equipment name")
                    num_equipment = st.number_input("Number of Equipment", min_value=1, value=1, help="Enter number of equipment in this area")
                
                with col2:
                    st.write("**Location Settings:**")
                    # Get default location
                    default_lat, default_lon = get_location()
                    
                    area_lat = st.number_input(
                        "Area Center Latitude", 
                        value=default_lat if default_lat else 0.0,
                        format="%.6f",
                        help="Enter the latitude of the center point"
                    )
                    area_lon = st.number_input(
                        "Area Center Longitude", 
                        value=default_lon if default_lon else 0.0,
                        format="%.6f",
                        help="Enter the longitude of the center point"
                    )
                    area_radius = st.number_input(
                        "Allowed Radius (km)", 
                        min_value=0.1, 
                        value=1.0,
                        step=0.1,
                        help="Enter the radius in kilometers"
                    )
                
                # Location preview
                if area_lat and area_lon:
                    st.write("**Location Preview:**")
                    st.markdown(f"[View on Map](https://www.google.com/maps?q={area_lat},{area_lon})")
                    st.info(f"Access will be allowed within {area_radius} km of this location")
                
                if st.form_submit_button("Add Area"):
                    if not equipment_name:
                        st.error("Please provide Equipment Name")
                    else:
                        if db_manager.add_equipment_area(equipment_name, area_lat, area_lon, area_radius, num_equipment):
                            st.success(f"Equipment area '{equipment_name}' added successfully!")
                            st.rerun()
                        else:
                            st.error("Failed to add equipment area. Please try again.")
            
            # Display existing equipment areas
            st.subheader("Existing Equipment Areas")
            equipment_areas = db_manager.get_all_equipment_areas()
            if equipment_areas:
                for area in equipment_areas:
                    with st.expander(f"{area[0]} ({area[4]} equipment)"):
                        st.write(f"Location: {area[1]}, {area[2]}")
                        st.write(f"Radius: {area[3]} km")
                        st.markdown(f"[View on Map](https://www.google.com/maps?q={area[1]},{area[2]})")
            else:
                st.info("No equipment areas added yet.")
        
        with tab2:
            st.subheader("Employee Management")
            
            # Get all employees
            employees = db_manager.get_all_employees()
            if employees:
                # Create a form for each employee
                for emp in employees:
                    with st.expander(f"{emp[1]} (ID: {emp[0]})"):
                        # Get current equipment assignment
                        current_equipment = db_manager.get_employee(emp[0])
                        current_equipment_id = current_equipment[3] if current_equipment else None
                        
                        # Get all equipment areas for selection
                        equipment_areas = db_manager.get_all_equipment_areas()
                        if equipment_areas:
                            equipment_options = {f"{area[0]} ({area[4]} equipment)": area[0] for area in equipment_areas}
                            current_selection = next((k for k, v in equipment_options.items() if v == current_equipment_id), None)
                            
                            selected_equipment = st.selectbox(
                                "Assigned Equipment Area",
                                options=["None"] + list(equipment_options.keys()),
                                index=0 if not current_selection else list(equipment_options.keys()).index(current_selection) + 1,
                                key=f"equip_{emp[0]}"
                            )
                            
                            if st.button("Update Assignment", key=f"update_{emp[0]}"):
                                if selected_equipment == "None":
                                    # Remove equipment assignment
                                    if db_manager.update_employee_equipment(emp[0], None):
                                        st.success("Equipment assignment removed!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update assignment")
                                else:
                                    # Update equipment assignment
                                    equipment_id = equipment_options[selected_equipment]
                                    if db_manager.update_employee_equipment(emp[0], equipment_id):
                                        st.success("Equipment assignment updated!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update assignment")
                        else:
                            st.warning("No equipment areas available. Please add equipment areas first.")
            else:
                st.info("No employees registered yet.")
        
        # Add logout button
        if st.button("Logout"):
            st.session_state.admin_authenticated = False
            st.session_state.admin_id = None
            st.rerun()
    else:
        st.warning("Please login as admin to manage equipment areas.")

elif option == "Sign Up":
    st.header("Employee Registration")
    
    # Initialize session state for face capture
    if 'face_captured' not in st.session_state:
        st.session_state.face_captured = False
        st.session_state.face_encodings = []
        st.session_state.avg_encoding = None
    
    # Step 1: Basic Information
    st.subheader("Step 1: Employee Information")
    with st.form(key="employee_info_form"):
        emp_id = st.text_input("Employee ID", help="Enter unique employee ID")
        name = st.text_input("Employee Name", help="Enter full name")
        
        # Store information in session state when form is submitted
        info_submitted = st.form_submit_button("Confirm Information & Proceed to Face Capture")
        
        if info_submitted:
            if not emp_id or not name:
                st.error("Please provide all required information")
            else:
                # Check if employee ID already exists
                existing_employee = db_manager.get_employee(emp_id)
                if existing_employee:
                    st.error(f"Employee ID '{emp_id}' already exists in the database")
                else:
                    # Store in session state
                    st.session_state.emp_id = emp_id
                    st.session_state.emp_name = name
                    st.session_state.info_confirmed = True
                    st.success("Information confirmed! Now proceed to face capture.")
                    st.rerun()
    
    # Step 2: Face Capture (only show if info is confirmed)
    if st.session_state.get('info_confirmed', False):
        st.subheader("Step 2: Face Capture")
        
        # Display confirmed information
        st.info(f"**Employee:** {st.session_state.emp_name} (ID: {st.session_state.emp_id})")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Face Capture Instructions:**")
            st.info("1. Look directly at the camera\n2. Ensure good lighting\n3. Remove glasses or obstructions\n4. Keep a neutral expression")
            
            if not st.session_state.face_captured:
                if st.button("Capture Face", key="capture_face_btn"):
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
                            # Check for multiple faces
                            if len(faces) > 1:
                                status_text.text("⚠️ Multiple faces detected! Please ensure only one face is in frame.")
                                time.sleep(1)
                                continue
                                
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
                            # Store face data in session state
                            st.session_state.face_encodings = face_encodings
                            st.session_state.avg_encoding = avg_encoding
                            st.session_state.face_captured = True
                            
                            progress.progress(1.0)
                            status_text.text("Face captured successfully!")
                            st.success("Face captured successfully! You can now complete the registration.")
                            st.rerun()
                    else:
                        st.error("Could not capture enough clear face images. Please try again.")
            else:
                st.success("✅ Face captured successfully!")
                if st.button("Recapture Face", key="recapture_btn"):
                    st.session_state.face_captured = False
                    st.session_state.face_encodings = []
                    st.session_state.avg_encoding = None
                    st.rerun()
    
    # Step 3: Final Registration (only show if both info confirmed and face captured)
    if st.session_state.get('info_confirmed', False) and st.session_state.get('face_captured', False):
        st.subheader("Step 3: Complete Registration")
        
        # Summary of all information
        st.write("**Registration Summary:**")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Employee ID:** {st.session_state.emp_id}")
            st.write(f"**Name:** {st.session_state.emp_name}")
            st.write(f"**Face:** ✅ Captured")
        
        # Final registration button
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Complete Registration", key="final_registration", type="primary"):
                try:
                    # Serialize the face encoding
                    serialized_encoding = pickle.dumps(st.session_state.avg_encoding)
                    
                    # Register employee in database
                    if db_manager.register_employee(
                        st.session_state.emp_id,
                        st.session_state.emp_name,
                        serialized_encoding,
                        None  # No equipment assignment during registration
                    ):
                        # Update recognizer
                        face_recognizer.add_face(st.session_state.avg_encoding, st.session_state.emp_id)
                        
                        # Success message
                        st.success(f"🎉 Registration completed successfully!")
                        st.success(f"Employee {st.session_state.emp_name} has been registered.")
                        
                        # Clear session state
                        for key in ['info_confirmed', 'face_captured', 'face_encodings', 'avg_encoding', 
                                   'emp_id', 'emp_name']:
                            if key in st.session_state:
                                del st.session_state[key]
                        
                        st.balloons()
                        time.sleep(2)
                        st.rerun()
                    else:
                        st.error("Failed to register employee. Please try again.")
                        
                except Exception as e:
                    st.error(f"Registration failed: {str(e)}")
        
        # Option to start over
        if st.button("🔄 Start Over", key="start_over"):
            # Clear all session state
            for key in ['info_confirmed', 'face_captured', 'face_encodings', 'avg_encoding', 
                       'emp_id', 'emp_name']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

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
                        # Check for multiple faces
                        if len(faces) > 1:
                            status_text.text("⚠️ Multiple faces detected! Please ensure only one face is in frame.")
                            time.sleep(1)
                            continue
                            
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
                
                # Get employee details including equipment area info
                employee = db_manager.get_employee(recognized_emp_id)
                
                if employee:
                    # Check if within allowed area
                    area_lat = employee[4]  # area_lat from joined query
                    area_lon = employee[5]  # area_lon from joined query
                    area_radius = employee[6]  # area_radius from joined query
                    
                    # Calculate distance and check if within area
                    is_within, distance = is_within_area(latitude, longitude, area_lat, area_lon, area_radius)
                    
                    if is_within:
                        # Log the login
                        db_manager.log_login(recognized_emp_id, latitude, longitude)
                        
                        progress.progress(1.0)
                        with col1:
                            st.success(f"Welcome, {employee[1]}!")
                            st.write(f"Logged in at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                            if latitude and longitude:
                                st.write(f"Your location: {latitude}, {longitude}")
                                st.write(f"Distance from equipment: {distance:.2f} km")
                                st.markdown(f"[View on Map](https://www.google.com/maps?q={latitude},{longitude})")
                        
                        # Set session state for logged in user and redirection
                        st.session_state.logged_in_user = recognized_emp_id
                        st.session_state.user_name = employee[1]
                        st.session_state.redirect_to_logbook = True
                        st.success("🚀 Redirecting to Operations Logbook...")
                        time.sleep(2)
                        st.rerun()
                    else:
                        progress.progress(1.0)
                        st.error("Login denied: You are not in the authorized equipment area.")
                        st.info(f"Your location: {latitude}, {longitude}")
                        if area_lat and area_lon:
                            st.info(f"Distance from equipment: {distance:.2f} km")
                            st.info(f"Authorized area: Within {area_radius} km of {area_lat}, {area_lon}")
                            st.markdown(f"[View Authorized Area](https://www.google.com/maps?q={area_lat},{area_lon})")
                else:
                    st.error("Employee not found in the database.")
            else:
                st.error("Login failed. Please try again.")

# Add footer
st.markdown("---")
st.markdown("Face Authentication System | Built with OpenCV & Streamlit")