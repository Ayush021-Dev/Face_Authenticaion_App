import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

import streamlit as st
import pandas as pd
import speech_recognition as sr
from datetime import datetime
import os
from database.db_manager import DatabaseManager

# Check if user is logged in
if 'logged_in_user' not in st.session_state:
    st.error("Please login first to access the Operations Logbook")
    st.stop()

# Get logged in user info
logged_in_user = st.session_state.logged_in_user
user_name = st.session_state.user_name

# Display header with user information
st.markdown(f"""
<div class="main-header">
    <h1>Operations Logbook</h1>
    <p>Welcome, {user_name} (ID: {logged_in_user})</p>
</div>
""", unsafe_allow_html=True)

# Enhanced CSS styling
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Root variables for consistent theming */
:root {
    --primary-color: #60a5fa; /* Light blue */
    --secondary-color: #facc15; /* Yellow */
    --success-color: #34d399; /* Green */
    --danger-color: #f87171; /* Red */
    --warning-color: #facc15; /* Yellow */
    --info-color: #60a5fa; /* Light blue */
    --background-primary: #1f2937; /* Dark grey */
    --background-secondary: #111827; /* Even darker grey */
    --background-accent: #374151; /* Medium grey */
    --text-primary: #f3f4f6; /* Light grey */
    --text-secondary: #d1d5db; /* Lighter grey */
    --border-color: #4b5563; /* Grey border */
    --shadow-sm: 0 1px 2px 0 rgb(0 0 0 / 0.2);
    --shadow-md: 0 4px 6px -1px rgb(0 0 0 / 0.2), 0 2px 4px -2px rgb(0 0 0 / 0.2);
    --shadow-lg: 0 10px 15px -3px rgb(0 0 0 / 0.3), 0 4px 6px -4px rgb(0 0 0 / 0.3);
    --border-radius: 0.75rem;
}

/* General styling */
.stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    background: linear-gradient(135deg, var(--background-secondary) 0%, var(--background-primary) 100%); /* Dark gradient background */
    min-height: 100vh;
    color: var(--text-primary); /* Default text color */
}

.main .block-container {
    padding: 1.5rem;
    max-width: 1200px;
}

/* Header styling */
.main-header {
    background: linear-gradient(135deg, var(--background-accent) 0%, var(--background-primary) 100%); /* Dark grey gradient */
    color: var(--text-primary);
    padding: 2rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-lg);
    text-align: center;
    position: relative;
    overflow: hidden;
    border: 1px solid var(--border-color);
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="50" cy="50" r="2" fill="#4b5563" opacity="0.1"/></svg>') repeat;
    background-size: 20px 20px;
}

.main-header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    z-index: 1;
    position: relative;
    color: var(--text-primary); /* Header text color */
}

.main-header p {
    font-size: 1.1rem;
    opacity: 0.9;
    z-index: 1;
    position: relative;
    color: var(--text-secondary); /* Header text color */
}

/* Progress container styling */
.progress-container {
    background: var(--background-primary);
    padding: 1.5rem;
    border-radius: var(--border-radius);
    margin-bottom: 2rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
}

.step-indicator {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 1rem;
}

.step-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 500;
    color: var(--text-secondary);
}

.step-item.active {
    color: var(--primary-color);
}

.step-item.completed .step-number {
    background: var(--success-color);
    color: white;
    border-color: var(--success-color);
}

.step-number {
    width: 2rem;
    height: 2rem;
    border-radius: 50%;
    background: var(--background-accent);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.875rem;
    font-weight: 600;
    border: 2px solid var(--border-color);
    color: var(--text-primary);
}

.step-item.active .step-number {
    background: var(--primary-color);
    color: white;
    border-color: var(--primary-color);
}


/* Card styling */
.card {
    background: var(--background-primary);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}

.card:hover {
    transform: translateY(-2px);
    box-shadow: var(--shadow-lg);
}

.area-card {
    cursor: pointer;
    border: 2px solid transparent;
    transition: all 0.3s ease;
    background: var(--background-secondary);
    color: var(--text-secondary);
}

.area-card:hover {
    border-color: var(--primary-color);
    background: var(--background-accent);
    color: var(--text-primary);
}

.area-card.selected {
    border-color: var(--primary-color);
    background: var(--background-accent);
    color: var(--text-primary);
}

.area-icon {
    width: 3rem;
    height: 3rem;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--primary-color), var(--info-color));
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1.5rem;
    margin-bottom: 1rem;
}

.parameter-card {
    background: var(--background-secondary);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    border: 1px solid var(--border-color);
    margin-bottom: 1rem;
    transition: all 0.3s ease;
    position: relative;
    color: var(--text-primary);
}

.parameter-card:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
    background: var(--background-accent);
}

.parameter-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 1rem;
}

.parameter-title {
    font-size: 1.1rem;
    font-weight: 600;
    color: var(--text-primary);
    margin: 0;
}

.range-badge {
    background: linear-gradient(135deg, var(--secondary-color), #b45309); /* Darker yellow */
    color: white;
    padding: 0.25rem 0.75rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 500;
}

.voice-button {
    width: 100%;
    background: linear-gradient(135deg, var(--success-color), #059669); /* Green gradient for action */
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    box-shadow: var(--shadow-sm);
}

.voice-button:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
    background: linear-gradient(135deg, #059669, #047857); /* Darker green */
}

.voice-button:active {
    transform: translateY(0);
}

/* Status indicators */
.status-normal {
    color: var(--success-color);
    background: #065f46; /* Dark green */
    border: 1px solid #34d399;
    padding: 0.25rem 0.75rem;
    border-radius: var(--border-radius);
    font-size: 0.875rem;
}

.status-warning {
    color: var(--warning-color);
    background: #78350f; /* Dark yellow/orange */
    border: 1px solid #facc15;
    padding: 0.25rem 0.75rem;
    border-radius: var(--border-radius);
    font-size: 0.875rem;
}

.status-danger {
    color: var(--danger-color);
    background: #991b1b; /* Dark red */
    border: 1px solid #f87171;
    padding: 0.25rem 0.75rem;
    border-radius: var(--border-radius);
    font-size: 0.875rem;
}

/* Button styling */
.btn-primary {
    background: linear-gradient(135deg, var(--primary-color), var(--info-color));
    color: white;
    border: none;
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.btn-primary:hover {
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

.btn-secondary {
    background: var(--background-primary);
    color: var(--text-primary);
    border: 1px solid var(--border-color);
    border-radius: var(--border-radius);
    padding: 0.75rem 1.5rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: var(--shadow-sm);
}

.btn-secondary:hover {
    background: var(--background-accent);
    transform: translateY(-1px);
    box-shadow: var(--shadow-md);
}

/* Streamlit component overrides */
.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, var(--primary-color), var(--info-color)) !important; /* Blue gradient */
    color: white !important;
    border: none !important;
    border-radius: var(--border-radius) !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    box-shadow: var(--shadow-sm) !important;
}

.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: var(--shadow-md) !important;
}

.stRadio > div {
    gap: 1rem;
}

.stAlert {
    border-radius: var(--border-radius) !important;
    border: none !important;
    box-shadow: var(--shadow-sm) !important;
    color: var(--text-primary) !important;
}

.stAlert.info {
    background: #1e3a8a !important; /* Darker blue */
    color: #bfdbfe !important; /* Light blue text */
}

.stAlert.success {
    background: #065f46 !important; /* Dark green */
    color: #a7f3d0 !important; /* Light green text */
}

.stAlert.error {
    background: #991b1b !important; /* Dark red */
    color: #fecaida !important; /* Light red text */
}

/* Data display */
.metric-card {
    background: var(--background-secondary);
    border-radius: var(--border-radius);
    padding: 1.5rem;
    text-align: center;
    box-shadow: var(--shadow-md);
    border: 1px solid var(--border-color);
    color: var(--text-primary);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--primary-color);
    margin-bottom: 0.5rem;
}

.metric-label {
    color: var(--text-secondary);
    font-size: 0.875rem;
    font-weight: 500;
}

/* Responsive design */
@media (max-width: 768px) {
    .main-header h1 {
        font-size: 2rem;
    }
    
    .step-indicator {
        flex-direction: column;
        gap: 1rem;
    }
    
    .main .block-container {
        padding: 1rem;
    }
    
    .card,
    .progress-container {
        padding: 1rem;
    }
    
    .parameter-header {
        flex-direction: column;
        gap: 0.5rem;
    }
    
    .parameter-title {
        font-size: 1rem;
    }
    
    .range-badge {
        font-size: 0.7rem;
    }
    
    .voice-button {
        font-size: 0.9rem;
        padding: 0.6rem 0.8rem;
    }
    
    .btn-primary,
    .btn-secondary {
        font-size: 0.9rem;
        padding: 0.6rem 1rem;
    }
}

/* Loading spinner */
.loading-spinner {
    display: inline-block;
    width: 1rem;
    height: 1rem;
    border: 2px solid #f3f3f3;
    border-top: 2px solid var(--primary-color);
    border-radius: 50%;
    animation: spin 1s linear infinite;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(135deg, var(--background-primary) 0%, var(--background-accent) 100%);
    color: var(--text-primary);
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

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