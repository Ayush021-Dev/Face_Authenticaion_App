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

# Store equipment info in session state for easy access
st.session_state.area_equipment_info = {eq[1]: eq for eq in area_equipment}
num_equipment = len(area_equipment)

# Audio recording functions
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
                           dtype='float64')
        
        # Create progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(duration):
            time.sleep(1)
            progress = (i + 1) / duration
            progress_bar.progress(progress)
            status_text.text(f"Recording... {duration - i - 1} seconds remaining")
        
        sd.wait()  # Wait until recording is finished
        progress_bar.empty()
        status_text.empty()
        
        st.success("✅ Recording completed!")
        return audio_data, sample_rate
        
    except Exception as e:
        st.error(f"❌ Error recording audio: {str(e)}")
        return None, None

def save_audio_to_temp_file(audio_data, sample_rate):
    """Save audio data to temporary WAV file"""
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        
        # Normalize audio data
        audio_data = np.clip(audio_data, -1.0, 1.0)
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        # Write WAV file
        with wave.open(temp_file.name, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())
        
        return temp_file.name
        
    except Exception as e:
        st.error(f"❌ Error saving audio file: {str(e)}")
        return None

def transcribe_with_whisper(audio_file_path, model):
    """Transcribe audio using Whisper model"""
    try:
        if model is None:
            st.error("❌ Whisper model not loaded")
            return None
        
        # Transcribe audio
        segments, info = model.transcribe(audio_file_path, language="en")
        
        # Combine all segments
        transcript = ""
        for segment in segments:
            transcript += segment.text + " "
        
        # Clean up temporary file
        try:
            os.unlink(audio_file_path)
        except:
            pass
        
        return transcript.strip()
        
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
    
    # Transcribe with Whisper
    st.info("🔄 Processing speech with Whisper... Please wait.")
    transcribed_text = transcribe_with_whisper(temp_audio_file, model)
    
    return transcribed_text

def parse_voice_readings(transcript, equipment_parameters):
    """Parse voice transcript to extract equipment readings"""
    readings = {}
    
    if not transcript:
        return readings
    
    # Convert transcript to lowercase for easier matching
    text = transcript.lower()
    
    # Common patterns for numbers
    number_patterns = [
        r'(\d+\.?\d*)',  # Basic numbers (123, 12.5)
        r'(\d+\s*point\s*\d+)',  # "20 point 5" format
        r'(zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred)'
    ]
    
    # Word to number mapping
    word_to_num = {
        'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5,
        'six': 6, 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10,
        'eleven': 11, 'twelve': 12, 'thirteen': 13, 'fourteen': 14, 'fifteen': 15,
        'sixteen': 16, 'seventeen': 17, 'eighteen': 18, 'nineteen': 19, 'twenty': 20,
        'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
        'eighty': 80, 'ninety': 90, 'hundred': 100
    }
    
    for param_id, param_name, unit, min_val, max_val in equipment_parameters:
        param_lower = param_name.lower()
        
        # Look for parameter mentions
        param_patterns = [
            param_lower,
            param_lower.replace(' ', ''),
            param_lower.replace('_', ' ')
        ]
        
        for pattern in param_patterns:
            # Find mentions of this parameter
            if pattern in text:
                # Look for numbers near this parameter mention
                param_index = text.find(pattern)
                
                # Search in a window around the parameter mention
                search_window = text[max(0, param_index-50):param_index+100]
                
                # Try to find numbers in this window
                for num_pattern in number_patterns:
                    matches = re.findall(num_pattern, search_window)
                    for match in matches:
                        try:
                            # Convert match to number
                            if match in word_to_num:
                                value = word_to_num[match]
                            elif 'point' in match:
                                # Handle "20 point 5" format
                                parts = match.split('point')
                                if len(parts) == 2:
                                    value = float(parts[0].strip()) + float(parts[1].strip()) / 10
                                else:
                                    continue
                            else:
                                value = float(match)
                            
                            # Validate range if specified
                            if min_val is not None and value < min_val:
                                continue
                            if max_val is not None and value > max_val:
                                continue
                            
                            readings[param_id] = {
                                'name': param_name,
                                'value': value,
                                'unit': unit
                            }
                            break
                        except ValueError:
                            continue
                
                if param_id in readings:
                    break
    
    return readings

# Main App Layout
st.title("📝 Equipment Logbook")
st.subheader(f"Welcome, {user_name}!")
st.info(f"📍 Assigned Area: **{assigned_area_name}** ({num_equipment} equipment units)")

# Navigation tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Record Readings", "🎤 Voice Input", "📈 View Data", "📜 History"])

with tab1:
    st.header("📊 Manual Equipment Readings")
    
    # Equipment selection
    equipment_options = {eq[1]: eq[0] for eq in area_equipment}  # name: id
    selected_equipment_name = st.selectbox(
        "Select Equipment",
        options=list(equipment_options.keys()),
        key="manual_equipment_select"
    )
    
    if selected_equipment_name:
        equipment_id = equipment_options[selected_equipment_name]
        
        # Get parameters for selected equipment
        parameters = db_manager.get_equipment_parameters(equipment_id)
        
        if parameters:
            st.subheader(f"Parameters for {selected_equipment_name}")
            
            # Create form for readings
            with st.form(key="manual_readings_form"):
                readings_data = {}
                
                # Create input fields for each parameter
                for param_id, param_name, unit, min_val, max_val in parameters:
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        # Create number input with validation
                        help_text = f"Enter {param_name}"
                        if min_val is not None and max_val is not None:
                            help_text += f" (Range: {min_val} - {max_val} {unit})"
                        elif min_val is not None:
                            help_text += f" (Min: {min_val} {unit})"
                        elif max_val is not None:
                            help_text += f" (Max: {max_val} {unit})"
                        
                        value = st.number_input(
                            f"{param_name} ({unit})",
                            min_value=min_val if min_val is not None else 0.0,
                            max_value=max_val if max_val is not None else 10000.0,
                            step=0.1,
                            format="%.2f",
                            help=help_text,
                            key=f"manual_{param_id}"
                        )
                        readings_data[param_id] = value
                    
                    with col2:
                        st.write(f"**{unit}**")
                
                # Submit button
                if st.form_submit_button("💾 Save Readings", type="primary"):
                    success_count = 0
                    error_count = 0
                    
                    # Save each reading
                    for param_id, value in readings_data.items():
                        if db_manager.add_equipment_reading(equipment_id, param_id, value, user_id):
                            success_count += 1
                        else:
                            error_count += 1
                    
                    if success_count > 0:
                        st.success(f"✅ Successfully saved {success_count} readings!")
                    if error_count > 0:
                        st.error(f"❌ Failed to save {error_count} readings!")
                    
                    if success_count > 0:
                        st.rerun()
        else:
            st.info("No parameters configured for this equipment.")

with tab2:
    st.header("🎤 Voice Input for Equipment Readings")
    
    # Load Whisper model
    if st.session_state.whisper_model is None:
        with st.spinner("Loading Whisper model..."):
            st.session_state.whisper_model = load_whisper_model("base")
    
    if st.session_state.whisper_model is None:
        st.error("❌ Could not load speech recognition model. Voice input is not available.")
    else:
        st.success("✅ Speech recognition model loaded successfully!")
        
        # Equipment selection for voice input
        voice_equipment_name = st.selectbox(
            "Select Equipment for Voice Input",
            options=list(equipment_options.keys()),
            key="voice_equipment_select"
        )
        
        if voice_equipment_name:
            voice_equipment_id = equipment_options[voice_equipment_name]
            voice_parameters = db_manager.get_equipment_parameters(voice_equipment_id)
            
            if voice_parameters:
                st.subheader(f"Voice Input for {voice_equipment_name}")
                
                # Show available parameters
                st.info("**Available Parameters:** " + ", ".join([f"{p[1]} ({p[2]})" for p in voice_parameters]))
                
                # Voice input controls
                col1, col2 = st.columns(2)
                
                with col1:
                    duration = st.slider("Recording Duration (seconds)", 5, 30, 10)
                
                with col2:
                    if st.button("🎤 Start Voice Recording", type="primary"):
                        transcript = get_whisper_voice_input(st.session_state.whisper_model, duration)
                        if transcript:
                            st.session_state.transcript = transcript
                            st.rerun()
                
                # Show transcript
                if st.session_state.transcript:
                    st.subheader("📝 Transcript")
                    st.text_area("Recognized Speech", st.session_state.transcript, height=100)
                    
                    # Parse readings from transcript
                    parsed_readings = parse_voice_readings(st.session_state.transcript, voice_parameters)
                    
                    if parsed_readings:
                        st.subheader("🔍 Extracted Readings")
                        
                        # Display parsed readings
                        for param_id, reading_info in parsed_readings.items():
                            st.write(f"**{reading_info['name']}:** {reading_info['value']} {reading_info['unit']}")
                        
                        # Confirmation and saving
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("✅ Confirm & Save", type="primary"):
                                success_count = 0
                                for param_id, reading_info in parsed_readings.items():
                                    if db_manager.add_equipment_reading(voice_equipment_id, param_id, reading_info['value'], user_id):
                                        success_count += 1
                                
                                if success_count > 0:
                                    st.success(f"✅ Successfully saved {success_count} readings from voice input!")
                                    st.session_state.transcript = ""  # Clear transcript
                                    st.rerun()
                                else:
                                    st.error("❌ Failed to save readings!")
                        
                        with col2:
                            if st.button("🔄 Record Again"):
                                st.session_state.transcript = ""
                                st.rerun()
                        
                        with col3:
                            if st.button("❌ Clear"):
                                st.session_state.transcript = ""
                                st.rerun()
                    
                    else:
                        st.warning("⚠️ No equipment readings were detected in the speech. Please try again or use manual input.")
                        st.info("💡 **Tips for better voice recognition:**\n- Speak clearly and slowly\n- Use parameter names (e.g., 'Temperature is 25 degrees')\n- Include units when possible\n- Avoid background noise")
                
                # Voice input examples
                with st.expander("💡 Voice Input Examples"):
                    st.markdown("""
                    **Good examples:**
                    - "Temperature is 25 degrees, pressure is 2.5 bar"
                    - "The flow rate is 150 cubic meters per hour"
                    - "Level is at 75 percent, temperature 30 degrees"
                    
                    **Tips:**
                    - Speak clearly and at moderate pace
                    - Use the exact parameter names shown above
                    - Include numbers clearly (avoid "uh" or "um")
                    - Pause briefly between different readings
                    """)

with tab3:
    st.header("📈 Equipment Data Visualization")
    
    # Equipment selection for data view
    view_equipment_name = st.selectbox(
        "Select Equipment to View",
        options=list(equipment_options.keys()),
        key="view_equipment_select"
    )
    
    if view_equipment_name:
        view_equipment_id = equipment_options[view_equipment_name]
        
        # Get recent readings
        readings = db_manager.get_equipment_readings(view_equipment_id, 100)
        
        if readings:
            # Convert to DataFrame
            df_data = []
            for reading in readings:
                df_data.append({
                    'Reading ID': reading[0],
                    'Equipment ID': reading[1],
                    'Parameter ID': reading[2],
                    'Value': reading[3],
                    'Time': reading[4],
                    'Parameter': reading[5],
                    'Unit': reading[6]
                })
            
            df = pd.DataFrame(df_data)
            df['Time'] = pd.to_datetime(df['Time'])
            
            # Show summary statistics
            st.subheader(f"📊 Recent Readings for {view_equipment_name}")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Readings", len(df))
            with col2:
                st.metric("Parameters", df['Parameter'].nunique())
            with col3:
                if not df.empty:
                    st.metric("Latest Reading", df.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M'))
            
            # Parameter-wise visualization
            parameters = df['Parameter'].unique()
            
            for param in parameters:
                param_data = df[df['Parameter'] == param].sort_values('Time')
                
                if len(param_data) > 1:
                    st.subheader(f"📈 {param} Trend")
                    
                    # Create time series plot
                    fig = px.line(param_data, x='Time', y='Value', 
                                title=f"{param} over Time",
                                labels={'Value': f"{param} ({param_data.iloc[0]['Unit']})"})
                    
                    fig.update_layout(
                        xaxis_title="Time",
                        yaxis_title=f"{param} ({param_data.iloc[0]['Unit']})",
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Current", f"{param_data.iloc[0]['Value']:.2f}")
                    with col2:
                        st.metric("Average", f"{param_data['Value'].mean():.2f}")
                    with col3:
                        st.metric("Min", f"{param_data['Value'].min():.2f}")
                    with col4:
                        st.metric("Max", f"{param_data['Value'].max():.2f}")
            
            # Show raw data table
            with st.expander("📋 Raw Data Table"):
                st.dataframe(df[['Time', 'Parameter', 'Value', 'Unit']].sort_values('Time', ascending=False))
        
        else:
            st.info("No readings available for this equipment yet.")

with tab4:
    st.header("📜 Reading History & Export")
    
    # Get all readings for the area
    all_readings = db_manager.get_all_readings_for_area(assigned_area_name, 1000)
    
    if all_readings:
        # Convert to DataFrame
        history_data = []
        for reading in all_readings:
            history_data.append({
                'Reading ID': reading[0],
                'Equipment ID': reading[1],
                'Parameter ID': reading[2],
                'Value': reading[3],
                'Time': reading[4],
                'Parameter': reading[5],
                'Unit': reading[6],
                'Equipment': reading[7]
            })
        
        history_df = pd.DataFrame(history_data)
        history_df['Time'] = pd.to_datetime(history_df['Time'])
        
        # Summary statistics
        st.subheader(f"📊 Area Summary: {assigned_area_name}")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Readings", len(history_df))
        with col2:
            st.metric("Equipment Count", history_df['Equipment'].nunique())
        with col3:
            st.metric("Parameters", history_df['Parameter'].nunique())
        with col4:
            if not history_df.empty:
                st.metric("Latest Entry", history_df.iloc[0]['Time'].strftime('%Y-%m-%d %H:%M'))
        
        # Filters
        st.subheader("🔍 Filter Data")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            equipment_filter = st.multiselect(
                "Select Equipment",
                options=history_df['Equipment'].unique(),
                default=history_df['Equipment'].unique()
            )
        
        with col2:
            parameter_filter = st.multiselect(
                "Select Parameters",
                options=history_df['Parameter'].unique(),
                default=history_df['Parameter'].unique()
            )
        
        with col3:
            days_back = st.number_input("Days to show", min_value=1, max_value=365, value=7)
        
        # Apply filters
        filtered_df = history_df[
            (history_df['Equipment'].isin(equipment_filter)) &
            (history_df['Parameter'].isin(parameter_filter)) &
            (history_df['Time'] >= (datetime.now() - pd.Timedelta(days=days_back)))
        ].sort_values('Time', ascending=False)
        
        # Display filtered data
        st.subheader("📋 Filtered Results")
        st.info(f"Showing {len(filtered_df)} readings")
        
        # Display table
        display_df = filtered_df[['Time', 'Equipment', 'Parameter', 'Value', 'Unit']].copy()
        display_df['Time'] = display_df['Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
        st.dataframe(display_df, use_container_width=True)
        
        # Export functionality
        st.subheader("📤 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📊 Download as CSV"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV File",
                    data=csv,
                    file_name=f"equipment_readings_{assigned_area_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📋 Generate Report"):
                # Create summary report
                report_data = {
                    'Area': assigned_area_name,
                    'Report Generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'Total Readings': len(filtered_df),
                    'Equipment Count': filtered_df['Equipment'].nunique(),
                    'Date Range': f"{filtered_df['Time'].min().strftime('%Y-%m-%d')} to {filtered_df['Time'].max().strftime('%Y-%m-%d')}"
                }
                
                # Parameter summaries
                param_summary = filtered_df.groupby(['Equipment', 'Parameter']).agg({
                    'Value': ['count', 'mean', 'min', 'max', 'std']
                }).round(2)
                
                st.json(report_data)
                st.subheader("📈 Parameter Statistics")
                st.dataframe(param_summary)
    
    else:
        st.info("No reading history available for your assigned area yet.")

# Footer
st.markdown("---")
st.markdown(f"👤 **Logged in as:** {user_name} | 📍 **Area:** {assigned_area_name}")

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