from database.db_manager import DatabaseManager
import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

import sounddevice as sd
import numpy as np
import tempfile
import wave
import threading
import time
from io import BytesIO
from faster_whisper import WhisperModel

# Set page configuration
st.set_page_config(
    page_title="Equipment Logbook",
    page_icon="📝",
    layout="wide"
)

# Initialize session state
if 'readings' not in st.session_state:
    st.session_state.readings = {}
if 'transcript' not in st.session_state:
    st.session_state.transcript = ''
if 'is_listening' not in st.session_state:
    st.session_state.is_listening = False
if 'error' not in st.session_state:
    st.session_state.error = None
if 'current_step' not in st.session_state:
    st.session_state.current_step = 1
if 'whisper_model' not in st.session_state:
    st.session_state.whisper_model = None
if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'area_equipment_info' not in st.session_state:
    st.session_state.area_equipment_info = {}

# Load Whisper model with caching for efficiency
@st.cache_resource
def load_whisper_model(model_size="base"):
    """Load Whisper model with caching to avoid reloading"""
    try:
        model = WhisperModel(model_size)
        return model
    except Exception as e:
        st.error(f"Failed to load Whisper model: {e}")
        return None

# Check if user is logged in first
if not st.session_state.get('logged_in_user'):
    st.warning("⚠️ Please login first to access the logbook!")
    st.info("Click the button below to go back to the main page and login.")
    if st.button("🔙 Go to Login Page"):
        st.switch_page("app.py")
    st.stop()

# Initialize database manager
db_manager = DatabaseManager()
user_id = st.session_state.get('logged_in_user')
user_name = st.session_state.get('user_name', 'Unknown User')

# Get user info from session state or database
user_info = st.session_state.get('user_info')
if not user_info and user_id:
    user_info = db_manager.get_employee(user_id)
    if user_info:
        st.session_state.user_info = user_info

# If user info is still not found, show error
if not user_info:
    st.error("❌ User information not found!")
    st.info("Please logout and login again.")
    if st.button("🚪 Logout"):
        for key in ['logged_in_user', 'user_name', 'user_info', 'login_time']:
            if key in st.session_state:
                del st.session_state[key]
        st.switch_page("app.py")
    st.stop()

assigned_area_name = user_info.get('equipment_name')

# Check if user has an assigned area
if not assigned_area_name:
    st.warning("⚠️ You are not assigned to any equipment area!")
    st.info("Please contact your administrator to assign you to an equipment area.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔙 Back to Main Page"):
            st.switch_page("app.py")
    with col2:
        if st.button("🚪 Logout"):
            for key in ['logged_in_user', 'user_name', 'user_info', 'login_time']:
                if key in st.session_state:
                    del st.session_state[key]
            st.switch_page("app.py")
    st.stop()

# Fetch equipment details for the assigned area
area_equipment = db_manager.get_equipment_by_area(assigned_area_name)
if not area_equipment:
    st.error(f"❌ No equipment found for assigned area: {assigned_area_name}!")
    st.info("Please contact your administrator to set up equipment for your area.")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🔙 Back to Main Page"):
            st.switch_page("app.py")
    with col2:
        if st.button("🚪 Logout"):
            for key in ['logged_in_user', 'user_name', 'user_info', 'login_time']:
                if key in st.session_state:
                    del st.session_state[key]
            st.switch_page("app.py")
    st.stop()

num_equipment = area_info[4]

def initialize_excel():
    """Initialize the Excel file if it doesn't exist"""
    if not os.path.exists('bina_refinery_log.xlsx'):
        df = pd.DataFrame(columns=['Timestamp', 'User', 'Area', 'Equipment', 'Parameter', 'Value'])
        df.to_excel('bina_refinery_log.xlsx', index=False)

def append_to_excel(equipment, parameter, value):
    """Append a new reading to the Excel file"""
    df = pd.read_excel('bina_refinery_log.xlsx')
    new_row = {
        'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'User': f"{st.session_state.user_name} (ID: {st.session_state.logged_in_user})",
        'Area': assigned_area,
        'Equipment': equipment,
        'Parameter': parameter,
        'Value': value
    }
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_excel('bina_refinery_log.xlsx', index=False)
    return True

def get_voice_input():
    """Capture voice input using speech_recognition with enhanced error handling"""
    recognizer = sr.Recognizer()
    
    # Adjust for ambient noise
    try:
        with sr.Microphone() as source:
            st.info("🎤 Adjusting for ambient noise... Please wait.")
            recognizer.adjust_for_ambient_noise(source, duration=2)
            st.info("🎤 Listening... Please speak now.")
            
            # Set timeout and phrase time limit
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            
            try:
                text = recognizer.recognize_google(audio)
                return text
            except sr.UnknownValueError:
                st.error("❌ Could not understand audio. Please speak clearly and try again.")
                return None
            except sr.RequestError as e:
                st.error(f"❌ Could not request results from speech recognition service; {e}")
                return None
    except Exception as e:
        st.error(f"❌ Error transcribing audio: {str(e)}")
        return None

def get_whisper_voice_input(model, duration=10):
    """Complete voice input pipeline using Whisper"""
    # Record audio
    audio_data, sample_rate = record_audio(duration)
    if audio_data is None:
        return None
    
    # Save to temporary file
    temp_audio_file = save_audio_to_temp_file(audio_data, sample_rate)
    if temp_audio_file is None:
        return None

def extract_parameter_value_pairs(text):
    """Extract multiple parameter-value pairs from text"""
    import re
    
    # Convert text to lowercase for better matching
    text = text.lower()
    
    # Split the text by common separators
    separators = [',', 'and', '&', ';']
    for sep in separators:
        text = text.replace(sep, '|')
    parts = text.split('|')
    
    # Process each part
    results = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
            
        # Extract numeric value
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", part)
        if not numbers:
            continue
            
        value = float(numbers[0])
        
        # Extract parameter name
        # Remove common words and clean up
        words_to_remove = ['is', 'are', 'was', 'were', 'equals', 'equal', 'to', 'at']
        parameter = part
        for word in words_to_remove:
            parameter = parameter.replace(word, '')
        
        # Get everything before the number
        parameter = parameter.split(str(numbers[0]))[0].strip()
        
        # Capitalize first letter of each word
        parameter = ' '.join(word.capitalize() for word in parameter.split())
        
        if parameter:  # Only add if we found a parameter name
            results.append((parameter, value))
    
    return results

def check_value_range(value, min_val, max_val):
    """Check if value is within range and return status"""
    if value < min_val:
        return "Below Range"
    elif value > max_val:
        return "Above Range"
    return "Normal"

def remove_last_entry_from_excel():
    """Removes the last entry from the Excel file"""
    try:
        if os.path.exists('bina_refinery_log.xlsx'):
            df = pd.read_excel('bina_refinery_log.xlsx')
            if not df.empty:
                df = df[:-1] # Drop the last row
                df.to_excel('bina_refinery_log.xlsx', index=False)
                return True
            else:
                return False # File is empty
        else:
            return False # File does not exist
    except Exception as e:
        st.error(f"An error occurred while removing the last entry: {e}")
        return False

# Initialize the Excel file
initialize_excel()

# Check if user is logged in
if not st.session_state.get('logged_in_user'):
    st.warning("Please login first!")
    st.stop()

# Streamlit UI
st.title("Bina Refinery Operations Logbook")
st.write(f"Welcome to {assigned_area} Operations Logbook")

# Display equipment cards
st.subheader("Equipment Readings")
cols = st.columns(min(3, num_equipment))  # Show max 3 columns

for i in range(num_equipment):
    with cols[i % 3]:
        st.markdown(f"""
        <div style='background-color: #1f2937; padding: 20px; border-radius: 10px; margin-bottom: 20px;'>
            <h3 style='color: #60a5fa;'>Equipment {i+1}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create a form for each equipment
        with st.form(key=f"equipment_{i+1}"):
            # Form submit button for voice input
            if st.form_submit_button("🎤 Start Voice Input"):
                with st.spinner("🎤 Listening..."):
                    voice_input = get_voice_input()
                    if voice_input:
                        st.write(f"🎯 Voice Input: {voice_input}")
                        parameter_value_pairs = extract_parameter_value_pairs(voice_input)
                        
                        if parameter_value_pairs:
                            # Add to session state
                            if f"Equipment {i+1}" not in st.session_state.readings:
                                st.session_state.readings[f"Equipment {i+1}"] = []
                            
                            # Process each parameter-value pair
                            for parameter, value in parameter_value_pairs:
                                st.session_state.readings[f"Equipment {i+1}"].append({
                                    'parameter': parameter,
                                    'value': value,
                                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                                })
                                
                                # Append to Excel
                                append_to_excel(f"Equipment {i+1}", parameter, value)
                                st.success(f"✅ Successfully logged: {parameter} = {value}")
                        else:
                            st.error("❌ Could not extract any parameter-value pairs from the voice input")
                            st.info("💡 Try speaking clearly, for example: 'Temperature is 120, Current is 200, Pressure is 2.5'")

# Display current readings
st.subheader("Current Readings")
for equipment, readings in st.session_state.readings.items():
    with st.expander(f"{equipment} Readings"):
        for reading in readings:
            st.write(f"Parameter: {reading['parameter']}")
            st.write(f"Value: {reading['value']}")
            st.write(f"Time: {reading['timestamp']}")
            st.markdown("---")

# Export button
if st.button("📥 Export Readings as Excel"):
    try:
        df_export = pd.read_excel('bina_refinery_log.xlsx')
        if not df_export.empty:
            csv = df_export.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Excel",
                data=csv,
                file_name="readings.csv",
                mime="text/csv",
                key="download_csv"
            )
        else:
            st.warning("No readings to export.")
    except Exception as e:
        st.error(f"An error occurred during export: {e}")

# Clear readings button
if st.button("🗑️ Clear All Readings"):
    st.session_state.readings = {}
    st.success("All readings cleared!")
    st.rerun() 