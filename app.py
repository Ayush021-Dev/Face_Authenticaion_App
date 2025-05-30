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
    db_manager = DatabaseManager()  # This will now use SQLite
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
option = st.sidebar.radio("Go to", ["Home", "Areas", "Sign Up", "Login", "Database Info"])

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
    - SQLite database for reliable data storage
    
    ### Security Features:
    - Advanced anti-spoofing using multiple techniques
    - Multi-factor authentication
    - Secure SQLite database storage
    - Equipment area-based access control
    """)

elif option == "Database Info":
    st.header("📊 Database Statistics")
    
    # Get database statistics
    stats = db_manager.get_database_stats()
    
    if stats:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("👥 Employees", stats.get('employees', 0))
            st.metric("🔑 Admins", stats.get('admins', 0))
        
        with col2:
            st.metric("🏭 Areas", stats.get('equipment_areas', 0))
            st.metric("⚙️ Equipment", stats.get('equipment', 0))
        
        with col3:
            st.metric("📊 Parameters", stats.get('parameters', 0))
            st.metric("📈 Readings", stats.get('equipment_readings', 0))
        
        with col4:
            st.metric("📝 Login Logs", stats.get('login_logs', 0))
    
    # Database backup option
    st.subheader("🔄 Database Backup")
    if st.button("Create Backup"):
        backup_filename = f"face_auth_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.db"
        if db_manager.backup_database(backup_filename):
            st.success(f"✅ Database backed up to: {backup_filename}")
        else:
            st.error("❌ Failed to create backup")

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
                    with st.expander(f"{area['equipment_name']} ({area['num_equipment']} equipment)"):
                        st.write(f"**Location:** {area['area_lat']}, {area['area_lon']}")
                        st.write(f"**Radius:** {area['area_radius']} km")
                        st.markdown(f"[View on Map](https://www.google.com/maps?q={area['area_lat']},{area['area_lon']})")
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
                        current_employee = db_manager.get_employee(emp[0])
                        current_equipment_area = current_employee['equipment_name'] if current_employee else None
                        
                        # Get all equipment areas for selection
                        equipment_areas = db_manager.get_all_equipment_areas()
                        if equipment_areas:
                            # Create equipment options
                            equipment_options = {area['equipment_name']: area['equipment_name'] for area in equipment_areas}
                            current_selection = current_equipment_area if current_equipment_area in equipment_options else None
                            
                            # Calculate the index for selectbox
                            options_list = ["None"] + list(equipment_options.keys())
                            current_index = 0
                            if current_selection:
                                try:
                                    current_index = options_list.index(current_selection)
                                except ValueError:
                                    current_index = 0
                            
                            selected_equipment = st.selectbox(
                                "Assigned Equipment Area",
                                options=options_list,
                                index=current_index,
                                key=f"equip_{emp[0]}"
                            )
                            
                            if st.button("Update Assignment", key=f"update_{emp[0]}"):
                                if selected_equipment == "None":
                                    # Remove equipment assignment
                                    if db_manager.update_employee_equipment(emp[0], None):
                                        st.success(f"Equipment assignment removed for {emp[1]}")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update assignment")
                                else:
                                    # Update equipment assignment
                                    if db_manager.update_employee_equipment(emp[0], selected_equipment):
                                        st.success(f"Equipment assignment updated for {emp[1]} to {selected_equipment}")
                                        st.rerun()
                                    else:
                                        st.error("Failed to update assignment")
                        else:
                            st.info("No equipment areas available. Please add areas first.")
            else:
                st.info("No employees registered yet.")

elif option == "Sign Up":
    st.header("Employee Registration")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Registration form
        with st.form(key="registration_form"):
            emp_id = st.text_input("Employee ID", help="Enter unique employee ID")
            emp_name = st.text_input("Full Name", help="Enter employee full name")
            
            # Equipment area selection
            equipment_areas = db_manager.get_all_equipment_areas()
            if equipment_areas:
                area_options = ["None"] + [area['equipment_name'] for area in equipment_areas]
                selected_area = st.selectbox(
                    "Equipment Area Assignment (optional)",
                    options=area_options,
                    help="Select the equipment area this employee will be assigned to"
                )
                if selected_area == "None":
                    selected_area = None
            else:
                selected_area = None
                st.info("No equipment areas available. Employee will be registered without area assignment.")
            
            submit_registration = st.form_submit_button("📸 Capture Face & Register")
    
    with col2:
        # Camera feed
        st.subheader("Face Capture")
        camera_placeholder = st.empty()
        
        # Initialize camera session state
        if 'camera_active' not in st.session_state:
            st.session_state.camera_active = False
        
        # Camera controls
        col_cam1, col_cam2 = st.columns(2)
        with col_cam1:
            if st.button("📷 Start Camera"):
                st.session_state.camera_active = True
        
        with col_cam2:
            if st.button("⏹️ Stop Camera"):
                st.session_state.camera_active = False
    
    # Handle registration submission
    if submit_registration:
        if not emp_id or not emp_name:
            st.error("Please fill in all required fields!")
        else:
            # Check if employee ID already exists
            existing_employee = db_manager.get_employee(emp_id)
            if existing_employee:
                st.error("Employee ID already exists! Please use a different ID.")
            else:
                # Capture face
                st.info("📸 Capturing face... Please look at the camera!")
                
                # Initialize camera
                cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    st.error("Could not open camera. Please check your camera connection.")
                else:
                    face_captured = False
                    capture_attempts = 0
                    max_attempts = 50  # 5 seconds at ~10 FPS
                    
                    while not face_captured and capture_attempts < max_attempts:
                        ret, frame = cap.read()
                        if ret:
                            # Detect faces
                            face_locations = face_detector.detect_face_dnn(frame)
                            
                            if face_locations:
                                # Find the largest face
                                largest_face = None
                                largest_area = 0
                                for (x, y, w, h) in face_locations:
                                    area = w * h
                                    if area > largest_area:
                                        largest_area = area
                                        largest_face = (x, y, w, h)

                                if largest_face is not None:
                                    # Add a minimum size check for the detected face
                                    min_face_area = 5000 # Example threshold, adjust as needed
                                    if largest_area > min_face_area:
                                        left, top, width, height = largest_face
                                        right = left + width
                                        bottom = top + height

                                        # Get coordinates from the largest face bounding box
                                        # Ensure coordinates are within image bounds before cropping
                                        h_frame, w_frame = frame.shape[:2]
                                        top_bounded = max(0, top)
                                        left_bounded = max(0, left)
                                        bottom_bounded = min(h_frame, bottom)
                                        right_bounded = min(w_frame, right)

                                        # Crop face
                                        face_img = frame[top_bounded:bottom_bounded, left_bounded:right_bounded]

                                        # Only proceed if the cropped image is valid
                                        if face_img.size != 0:
                                            # Get face encoding from the largest face
                                            face_encoding = face_recognizer.get_face_encoding(face_img)

                                            # Check if face encoding is valid before recognition
                                            if face_encoding is not None:
                                                # Check if face already exists
                                                exists, existing_id, existing_name = db_manager.face_exists(face_encoding)
                                                
                                                if exists:
                                                    st.error(f"Face already registered for {existing_name} (ID: {existing_id})")
                                                    cap.release()
                                                    break
                                                
                                                # Serialize face encoding
                                                face_encoding_blob = pickle.dumps(face_encoding)
                                                
                                                # Register employee
                                                if db_manager.register_employee(emp_id, emp_name, face_encoding_blob, selected_area):
                                                    st.success(f"✅ Employee {emp_name} registered successfully!")
                                                    
                                                    # Update face recognizer with new employee
                                                    employees = db_manager.get_all_employees()
                                                    face_recognizer.load_from_database(employees)
                                                    
                                                    face_captured = True
                                                    cap.release()
                                                    st.rerun()
                                                else:
                                                    st.error("Failed to register employee. Please try again.")
                                                    cap.release()
                                                    break
                            
                            # Display frame with face detection
                            display_frame = frame.copy()
                            for face_location in face_locations:
                                top, right, bottom, left = face_location
                                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            
                            # Convert BGR to RGB for Streamlit
                            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                            camera_placeholder.image(display_frame, channels="RGB", use_column_width=True)
                            
                            capture_attempts += 1
                            time.sleep(0.1)
                    
                    if not face_captured and capture_attempts >= max_attempts:
                        st.error("Could not detect face clearly. Please try again with better lighting and positioning.")
                    
                    cap.release()
    
    # Live camera feed
    if st.session_state.camera_active:
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Detect faces for preview
                face_locations = face_detector.detect_face_dnn(frame)

                # Find the largest face for drawing the bounding box
                largest_face = None
                largest_area = 0
                if face_locations:
                    for (x, y, w, h) in face_locations:
                        area = w * h
                        if area > largest_area:
                            largest_area = area
                            largest_face = (x, y, w, h)

                # Draw rectangle around the largest detected face
                display_frame = frame.copy()
                if largest_face is not None:
                    left, top, width, height = largest_face
                    right = left + width
                    bottom = top + height
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)

                # Convert BGR to RGB for Streamlit
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            cap.release()

elif option == "Login":
    st.header("Face Authentication Login")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Instructions")
        st.info("""
        1. Click 'Start Authentication' button
        2. Look directly at the camera
        3. Keep your face steady and well-lit
        4. Wait for the system to recognize you
        """)
        
        # Authentication controls
        if st.button("🔐 Start Face Authentication", type="primary"):
            st.session_state.auth_in_progress = True
        
        if st.button("⏹️ Stop Authentication"):
            st.session_state.auth_in_progress = False
    
    with col2:
        # Camera feed
        st.subheader("Camera Feed")
        auth_camera_placeholder = st.empty()
        auth_status_placeholder = st.empty()
    
    # Handle authentication
    if st.session_state.get('auth_in_progress', False):
        # Get current location
        current_lat, current_lon = get_location()
        
        # Initialize camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open camera. Please check your camera connection.")
            st.session_state.auth_in_progress = False
        else:
            auth_attempts = 0
            max_auth_attempts = 100  # 10 seconds at ~10 FPS
            authenticated = False
            
            while st.session_state.get('auth_in_progress', False) and auth_attempts < max_auth_attempts and not authenticated:
                ret, frame = cap.read()
                if ret:
                    # Detect faces
                    face_locations = face_detector.detect_face_dnn(frame)
                    
                    if face_locations:
                        auth_status_placeholder.info("👤 Face detected! Verifying identity...")

                        # Find the largest face
                        largest_face = None
                        largest_area = 0
                        for (x, y, w, h) in face_locations:
                            area = w * h
                            if area > largest_area:
                                largest_area = area
                                largest_face = (x, y, w, h)

                        if largest_face is not None:
                            # Add a minimum size check for the detected face
                            min_face_area = 5000 # Example threshold, adjust as needed
                            if largest_area > min_face_area:
                                left, top, width, height = largest_face
                                right = left + width
                                bottom = top + height

                                # Get coordinates from the largest face bounding box
                                # Ensure coordinates are within image bounds before cropping
                                h_frame, w_frame = frame.shape[:2]
                                top_bounded = max(0, top)
                                left_bounded = max(0, left)
                                bottom_bounded = min(h_frame, bottom)
                                right_bounded = min(w_frame, right)

                                # Crop face
                                face_img = frame[top_bounded:bottom_bounded, left_bounded:right_bounded]

                                # Only proceed if the cropped image is valid
                                if face_img.size != 0:
                                    # Get face encoding from the largest face
                                    face_encoding = face_recognizer.get_face_encoding(face_img)

                                    # Check if face encoding is valid before recognition
                                    if face_encoding is not None:
                                        # Recognize face
                                        try:
                                            employee_id, min_distance = face_recognizer.recognize_face(face_encoding)

                                            if employee_id and min_distance is not None and min_distance < 0.6:  # Use distance and tolerance for recognition check
                                                # Anti-spoofing check
                                                is_real = anti_spoof.detect_real_face(frame, face_locations[0])
                                                
                                                if is_real:
                                                    # Get employee details including area assignment
                                                    employee_details = db_manager.get_employee(employee_id)
                                                    
                                                    if employee_details:
                                                        # Check location if employee has area assignment
                                                        if employee_details['equipment_name'] and employee_details['area_lat'] and employee_details['area_lon']:
                                                            if current_lat and current_lon:
                                                                within_area = is_within_area(
                                                                    current_lat, current_lon,
                                                                    employee_details['area_lat'], employee_details['area_lon'],
                                                                    employee_details['area_radius']
                                                                )
                                                                
                                                                if not within_area:
                                                                    auth_status_placeholder.warning(f"⚠️ You are not within the allowed area for {employee_details['equipment_name']}")
                                                                    time.sleep(3)
                                                                    continue
                                                            else:
                                                                auth_status_placeholder.warning("⚠️ Could not determine your location. Please enable location services.")
                                                                time.sleep(3)
                                                                continue
                                                        
                                                        # Log successful login
                                                        db_manager.log_login(employee_id, current_lat, current_lon)
                                                        
                                                        # Set session state for successful login
                                                        st.session_state.logged_in_user = employee_id
                                                        st.session_state.user_name = employee_details['employee_name']
                                                        st.session_state.user_info = employee_details
                                                        st.session_state.login_time = datetime.now()
                                                        st.session_state.auth_in_progress = False
                                                        
                                                        # Success message
                                                        auth_status_placeholder.success(f"✅ Welcome {employee_details['employee_name']}! Authentication successful!")
                                                        
                                                        # Set redirect flag
                                                        st.session_state.redirect_to_logbook = True
                                                        
                                                        authenticated = True
                                                        cap.release()
                                                        
                                                        # Brief pause before redirect
                                                        time.sleep(2)
                                                        st.rerun()
                                                    else:
                                                        auth_status_placeholder.warning("⚠️ Possible spoofing attempt detected! Use your real face.")
                                                else:
                                                    auth_status_placeholder.warning("👤 Face not recognized. Please try again.")
                                        except Exception:
                                            st.warning("Could not process face recognition result.")
                    else:
                        auth_status_placeholder.info("📷 Looking for face... Please position yourself in front of the camera.")
                    
                    # Display frame with face detection
                    display_frame = frame.copy()
                    for face_location in face_locations:
                        top, right, bottom, left = face_location
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Convert BGR to RGB for Streamlit
                    display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    auth_camera_placeholder.image(display_frame, channels="RGB", use_column_width=True)
                    
                    auth_attempts += 1
                    time.sleep(0.1)
            
            if not authenticated and auth_attempts >= max_auth_attempts:
                auth_status_placeholder.error("❌ Authentication timeout. Please try again.")
                st.session_state.auth_in_progress = False
            
            cap.release()

# Display login status in sidebar
if st.session_state.get('logged_in_user'):
    st.sidebar.success(f"✅ Logged in as: {st.session_state.get('user_name', 'Unknown')}")
    st.sidebar.info(f"🆔 ID: {st.session_state.logged_in_user}")
    if st.session_state.get('login_time'):
        st.sidebar.info(f"🕐 Login time: {st.session_state.login_time.strftime('%H:%M:%S')}")
    
    # Add logout button
    if st.sidebar.button("🚪 Logout"):
        # Clear session state
        for key in ['logged_in_user', 'user_name', 'user_info', 'login_time', 'redirect_to_logbook']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()
    
    # Add logbook navigation button
    if st.sidebar.button("📝 Go to Logbook"):
        st.session_state.redirect_to_logbook = True
        st.rerun()
else:
    st.sidebar.info("Please login to access the system")

# Footer
st.markdown("---")
st.markdown("🔐 **Face Authentication System** - Powered by SQLite Database")