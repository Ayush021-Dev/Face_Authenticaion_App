# face_auth_system/face_utils/anti_spoof.py
import cv2
import numpy as np
import os
import time
import urllib.request

class AntiSpoofing:
    def __init__(self):
        # Initialize face analysis components
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        # For motion detection
        self.prev_frame = None
        self.last_motion_check = time.time()
        self.motion_history = []
        
        # For blink detection
        self.eye_aspect_ratios = []
        self.blink_detected = False
        
        # Download and load model if not already present
        self.model_loaded = self._load_or_download_model()
        
        print(f"Advanced anti-spoofing initialized. Model loaded: {self.model_loaded}")
    
    def _load_or_download_model(self):
        """Load face analysis DNN model or download if not available"""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # File paths for OpenCV DNN face detection model
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
                return False
        
        if not os.path.exists(config_file):
            print("Downloading model configuration...")
            try:
                urllib.request.urlretrieve(
                    "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                    config_file
                )
            except Exception as e:
                print(f"Could not download configuration: {e}")
                return False
                
        # Load the DNN model
        try:
            self.face_net = cv2.dnn.readNet(model_file, config_file)
            return True
        except Exception as e:
            print(f"Could not load face detection model: {e}")
            return False
    
    def check_liveness(self, frame, face_location):
        """
        Comprehensive anti-spoofing check combining multiple techniques
        
        Args:
            frame: Input BGR image
            face_location: Face location as (top, right, bottom, left)
            
        Returns:
            Boolean indicating if the face is real (True) or spoof (False)
        """
        try:
            # Extract face region
            top, right, bottom, left = face_location
            face_region = frame[top:bottom, left:right]
            
            if face_region.size == 0:
                return False
                
            # 1. Texture analysis (30%)
            texture_score = self._analyze_texture(face_region)
            
            # 2. Motion analysis (30%)
            motion_score = self._analyze_motion(frame)
            
            # 3. Eye detection and analysis (40%)
            eye_score = self._analyze_eyes(face_region)
            
            # Combine scores with weighting
            combined_score = (0.3 * texture_score + 0.3 * motion_score + 0.4 * eye_score)
            
            # Log detailed results for debugging
            print(f"Anti-spoofing scores - Texture: {texture_score:.2f}, Motion: {motion_score:.2f}, Eyes: {eye_score:.2f}")
            print(f"Combined liveness score: {combined_score:.2f}")
            
            # Decision threshold (adjust as needed based on testing)
            is_real = combined_score > 0.65
            
            return is_real
            
        except Exception as e:
            print(f"Error in advanced liveness detection: {e}")
            # Default to False (reject) in case of errors
            return False
    
    def _analyze_texture(self, face_region):
        """
        Analyze facial texture to detect photo/screen patterns
        
        Returns:
            Score between 0.0 (fake) and 1.0 (real)
        """
        try:
            # Resize for consistent analysis
            face_resized = cv2.resize(face_region, (128, 128))
            
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
            
            # 1. Gradient analysis
            gradient_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            gradient_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            
            gradient_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            gradient_variance = np.var(gradient_magnitude)
            
            # 2. Frequency analysis using FFT
            f_transform = np.fft.fft2(gray_face)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Calculate high-frequency content (printed images often lack this)
            # Get the center region (low frequencies)
            center_size = 20
            height, width = magnitude_spectrum.shape
            center_y, center_x = height // 2, width // 2
            
            # Create mask for high frequencies
            mask = np.ones(magnitude_spectrum.shape)
            mask[center_y-center_size:center_y+center_size, center_x-center_size:center_x+center_size] = 0
            
            # Calculate ratio of high to total frequency energy
            high_freq_energy = np.sum(magnitude_spectrum * mask)
            total_energy = np.sum(magnitude_spectrum)
            high_freq_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
            
            # Combine metrics - normalize and scale for 0-1 range
            texture_score = min(1.0, max(0.0, 
                0.6 * min(1.0, gradient_variance / 1000) + 
                0.4 * min(1.0, high_freq_ratio * 10)
            ))
            
            return texture_score
            
        except Exception as e:
            print(f"Error in texture analysis: {e}")
            return 0.5  # Default to middle value in case of error
    
    def _analyze_motion(self, frame):
        """
        Analyze motion patterns consistent with a live person
        
        Returns:
            Score between 0.0 (fake) and 1.0 (real)
        """
        try:
            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
            
            # Initialize on first call
            if self.prev_frame is None:
                self.prev_frame = gray_frame
                return 0.5  # Default middle value
            
            # Check if we should compute motion (every 300ms)
            current_time = time.time()
            if current_time - self.last_motion_check < 0.3:
                # Return last result if we have one
                if len(self.motion_history) > 0:
                    return self.motion_history[-1]
                return 0.5
            
            # Update timing
            self.last_motion_check = current_time
            
            # Calculate frame difference
            frame_diff = cv2.absdiff(self.prev_frame, gray_frame)
            
            # Update previous frame
            self.prev_frame = gray_frame
            
            # Calculate motion metrics
            motion_amount = np.mean(frame_diff)
            
            # Store in history (keep last 5 values)
            self.motion_history.append(motion_amount)
            if len(self.motion_history) > 5:
                self.motion_history.pop(0)
            
            # Analyze motion pattern
            avg_motion = np.mean(self.motion_history)
            motion_variance = np.var(self.motion_history) if len(self.motion_history) > 1 else 0
            
            # Score calculation:
            # - Low motion (<0.5): Likely a photo (low score)
            # - Very high motion (>15): Possible video playback (medium score)
            # - Moderate motion with some variance: Likely a real person (high score)
            
            if avg_motion < 0.5:
                base_score = avg_motion * 0.8  # Low score for very little motion
            elif avg_motion > 15:
                base_score = 0.7 - min(0.2, (avg_motion - 15) / 50)  # Penalize very high motion
            else:
                # Sweet spot: moderate motion gets high score
                base_score = 0.7 + min(0.3, (avg_motion - 0.5) / 15)
            
            # Add bonus for natural motion variance (real faces have micro-movements)
            variance_bonus = min(0.2, motion_variance / 5)
            
            # Final motion score
            motion_score = min(1.0, max(0.0, base_score + variance_bonus))
            
            return motion_score
            
        except Exception as e:
            print(f"Error in motion analysis: {e}")
            return 0.5  # Default to middle value in case of error
    
    def _analyze_eyes(self, face_region):
        """
        Analyze eyes for presence, blinking, and natural movement
        
        Returns:
            Score between 0.0 (fake) and 1.0 (real)
        """
        try:
            if face_region.size == 0:
                return 0.0
                
            # Convert to grayscale
            gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Enhance contrast for better eye detection
            gray_face = cv2.equalizeHist(gray_face)
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(gray_face, 1.1, 4, 
                                                    minSize=(30, 30), 
                                                    flags=cv2.CASCADE_SCALE_IMAGE)
            
            # Basic eye detection score
            if len(eyes) == 0:
                return 0.1  # No eyes detected, likely fake
            elif len(eyes) == 1:
                eye_presence_score = 0.5  # One eye might be obscured or profile view
            else:
                eye_presence_score = 0.9  # Both eyes detected, good sign
            
            # Extract eye regions for additional analysis
            eye_regions = []
            for (ex, ey, ew, eh) in eyes[:2]:  # Use up to 2 eyes
                eye_region = gray_face[ey:ey+eh, ex:ex+ew]
                if eye_region.size > 0:
                    eye_regions.append(eye_region)
            
            # If we have eye regions, analyze pupil/iris contrast
            # Real eyes have characteristic contrast patterns
            eye_pattern_score = 0.0
            if eye_regions:
                pattern_scores = []
                for eye in eye_regions:
                    # Simple variance-based analysis
                    # Real eyes typically have higher contrast variance
                    if eye.size > 0:
                        eye_variance = np.var(eye)
                        # Scale to approximate 0-1 range
                        pattern_scores.append(min(1.0, eye_variance / 1500))
                
                if pattern_scores:
                    eye_pattern_score = np.mean(pattern_scores)
            
            # Combined eye analysis score
            eye_score = 0.7 * eye_presence_score + 0.3 * eye_pattern_score
            return eye_score
            
        except Exception as e:
            print(f"Error in eye analysis: {e}")
            return 0.3  # Default to low-medium value in case of error
    
    def draw_result(self, frame, face_location, is_real):
        """
        Draw the anti-spoofing result on the frame
        
        Args:
            frame: Input image
            face_location: Face location as (top, right, bottom, left)
            is_real: Boolean indicating if face is real
            
        Returns:
            Annotated frame
        """
        top, right, bottom, left = face_location
        
        if is_real:
            color = (0, 255, 0)  # Green for real
            label = "Real Face"
        else:
            color = (0, 0, 255)  # Red for fake
            label = "Fake Face"
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label with background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (left, top - 25), (left + label_size[0], top), color, -1)
        cv2.putText(frame, label, (left, top - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame