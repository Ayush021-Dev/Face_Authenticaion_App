from database.db_manager import DatabaseManager
import streamlit as st
import pandas as pd
import os
import re
from datetime import datetime

import sounddevice as sd
import numpy as np
import tempfile
import wave
import threading
import time
from io import BytesIO
from faster_whisper import WhisperModel

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

# Get user's assigned area and equipment count
db_manager = DatabaseManager()
user_id = st.session_state.get('logged_in_user')
user_info = db_manager.get_employee(user_id)
if not user_info:
    st.error("User information not found!")
    st.stop()

assigned_area = user_info[3]  # equipment_name
area_info = db_manager.get_equipment_area(assigned_area)
if not area_info:
    st.error("Assigned area information not found!")
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
        st.error(f"❌ Error accessing microphone: {str(e)}")
        st.info("💡 Troubleshooting tips:")
        st.markdown("""
        1. Make sure your microphone is properly connected
        2. Check if your microphone is not being used by another application
        3. Try refreshing the page
        4. Make sure you've granted microphone permissions to the browser
        """)
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

def record_audio(duration=10, sample_rate=16000):
    """Record audio using sounddevice with improved error handling"""
    try:
        # Check available audio devices
        devices = sd.query_devices()
        default_input = sd.default.device[0]
        
        st.info(f"🎤 Using microphone: {devices[default_input]['name']}")
        st.info(f"🎤 Recording for {duration} seconds... Speak now!")
        
        # Record audio with better settings
        audio_data = sd.rec(int(duration * sample_rate), 
                           samplerate=sample_rate, 
                           channels=1, 
                           dtype=np.float32,
                           device=default_input)
        
        # Show recording progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(duration):
            time.sleep(1)
            progress_bar.progress((i + 1) / duration)
            status_text.info(f"🎤 Recording... {duration - i - 1} seconds remaining")
        
        sd.wait()  # Wait for recording to complete
        progress_bar.empty()
        status_text.empty()
        
        # Check if audio was recorded
        if np.max(np.abs(audio_data)) < 0.001:
            st.warning("⚠️ Very quiet audio detected. Please speak louder and try again.")
            return None, None
        
        st.success("✅ Audio recorded successfully!")
        return audio_data.flatten(), sample_rate
        
    except Exception as e:
        st.error(f"❌ Error during recording: {str(e)}")
        st.info("💡 Troubleshooting tips:")
        st.markdown("""
        1. Make sure your microphone is properly connected
        2. Check if your microphone is not being used by another application
        3. Try refreshing the page and allow microphone permissions
        4. Check your system's audio input settings
        5. Try a different microphone if available
        """)
        return None, None

def save_audio_to_temp_file(audio_data, sample_rate):
    """Save audio data to a temporary WAV file"""
    try:
        # Create temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Convert float32 to int16 for WAV format
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(temp_file.name, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 2 bytes per sample (int16)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
        
        return temp_file.name
    except Exception as e:
        st.error(f"Error saving audio file: {e}")
        return None

def transcribe_with_whisper(audio_file_path, model):
    """Transcribe audio using Whisper model - Fixed version without verbose parameter"""
    try:
        # Load and transcribe audio - removed both fp16 and verbose parameters for compatibility
        segments, info = model.transcribe(audio_file_path, language="en")
        
        # Extract text from segments
        transcribed_text = ""
        for segment in segments:
            transcribed_text += segment.text + " "
        
        return transcribed_text.strip()
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None
    finally:
        # Clean up temporary file
        try:
            if os.path.exists(audio_file_path):
                os.unlink(audio_file_path)
        except:
            pass

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
    
    # Transcribe with Whisper
    st.info("🔄 Processing speech with Whisper... Please wait.")
    transcribed_text = transcribe_with_whisper(temp_audio_file, model)
    
    return transcribed_text

def check_value_range(value, min_val, max_val):
    """Check if value is within range and return status"""
    if value < min_val:
        return "Below Range"
    elif value > max_val:
        return "Above Range"
    return "Normal"

def parse_readings(text, area_parameters):
    """Enhanced parsing to extract parameter values from speech - works without units"""
    readings = {}
    text = text.lower().strip()
    
    # Simplified patterns that work without units - just parameter name + number
    parameter_patterns = {
        "Temperature": [
            r"(?:temperature|temp)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"temp\s*(\d+\.?\d*)",
            r"temperature\s*(\d+\.?\d*)",
        ],
        "Pressure": [
            r"(?:pressure|press)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"press\s*(\d+\.?\d*)",
            r"pressure\s*(\d+\.?\d*)",
        ],
        "Water": [
            r"(?:water)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"water\s*(\d+\.?\d*)",
        ],
        "Flow": [
            r"(?:flow|rate)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"flow\s*(\d+\.?\d*)",
            r"rate\s*(\d+\.?\d*)",
        ],
        "Steam": [
            r"(?:steam)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"steam\s*(\d+\.?\d*)",
        ],
        "Power": [
            r"(?:power)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"power\s*(\d+\.?\d*)",
        ],
        "Efficiency": [
            r"(?:efficiency)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"efficiency\s*(\d+\.?\d*)",
        ],
        "pH": [
            r"(?:ph|p\.?h\.?)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"ph\s*(\d+\.?\d*)",
            r"p\.?h\.?\s*(\d+\.?\d*)",
        ],
        "COD": [
            r"(?:cod|c\.?o\.?d\.?)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"cod\s*(\d+\.?\d*)",
            r"c\.?o\.?d\.?\s*(\d+\.?\d*)",
        ],
        "Oil": [
            r"(?:oil)\s*(?:is|at|equals?|:|=)?\s*(\d+\.?\d*)",
            r"oil\s*(\d+\.?\d*)",
        ]
    }
    
    # Also try to find numbers in sequence and match them to available parameters
    # This helps when speech is unclear or parameters are mentioned in order
    numbers_in_text = re.findall(r'\d+\.?\d*', text)
    param_names = list(area_parameters.keys())
    
    # Try pattern matching first
    for param_name in area_parameters.keys():
        if param_name in parameter_patterns:
            patterns = parameter_patterns[param_name]
            for pattern in patterns:
                match = re.search(pattern, text)
                if match:
                    try:
                        value = float(match.group(1))
                        readings[param_name] = value
                        break  # Stop after first match for this parameter
                    except (ValueError, IndexError):
                        continue
    
    # If no pattern matches found and we have numbers, try sequential assignment
    if not readings and numbers_in_text and len(numbers_in_text) <= len(param_names):
        st.info("🔄 Pattern matching failed, trying sequential number assignment...")
        for i, number_str in enumerate(numbers_in_text):
            if i < len(param_names):
                try:
                    value = float(number_str)
                    param_name = param_names[i]
                    readings[param_name] = value
                    st.info(f"📍 Assigned {value} to {param_name} (position {i+1})")
                except ValueError:
                    continue
    
    return readings

# Initialize the Excel file
initialize_excel()

# Check if user is logged in
if not st.session_state.get('logged_in_user'):
    st.warning("Please login first!")
    st.stop()

# Load Whisper model
st.sidebar.header("🎙️ Whisper Settings")
model_size = st.sidebar.selectbox(
    "Select Whisper Model Size",
    ["tiny", "base", "small", "medium", "large"],
    index=1,  # Default to "base"
    help="Larger models are more accurate but slower. 'base' is recommended for most uses."
)

recording_duration = st.sidebar.slider(
    "Recording Duration (seconds)",
    min_value=5,
    max_value=30,
    value=15,
    help="How long to record audio. Longer recordings capture more data but take more time."
)

# Load the selected model
if st.session_state.whisper_model is None or getattr(st.session_state, 'model_size', None) != model_size:
    with st.spinner(f"Loading Whisper {model_size} model... This may take a moment."):
        st.session_state.whisper_model = load_whisper_model(model_size)
        st.session_state.model_size = model_size
    
    if st.session_state.whisper_model is not None:
        st.success(f"✅ Whisper {model_size} model loaded successfully!")
    else:
        st.error("❌ Failed to load Whisper model. Please try again.")
        st.stop()

# Streamlit UI
st.title("🏭 Bina Refinery Operations Logbook (Offline)")
st.write(f"Welcome to {assigned_area} Operations Logbook")
st.info("🔄 **Offline Mode**: Using Whisper for speech recognition - no internet required!")

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