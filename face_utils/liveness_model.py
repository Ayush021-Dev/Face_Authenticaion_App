import cv2
import numpy as np
import os
import urllib.request
import tensorflow as tf  # type: ignore
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout  # type: ignore
from tensorflow.keras.optimizers import Adam  # type: ignore

class LivenessDetector:
    def __init__(self):
        self.model = None
        self.model_loaded = self._load_or_download_model()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        print(f"Liveness detection model loaded: {self.model_loaded}")

    def _load_or_download_model(self):
        """Load or download the liveness detection model"""
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
        os.makedirs(model_dir, exist_ok=True)
        
        model_path = os.path.join(model_dir, "liveness_model.h5")
        
        # If model doesn't exist, download it
        if not os.path.exists(model_path):
            print("Downloading liveness detection model...")
            try:
                # Download the pre-trained model
                urllib.request.urlretrieve(
                    "https://github.com/your-repo/liveness_model/raw/main/liveness_model.h5",
                    model_path
                )
            except Exception as e:
                print(f"Could not download model: {e}")
                return False
        
        try:
            # Load the model
            self.model = tf.keras.models.load_model(model_path)
            return True
        except Exception as e:
            print(f"Could not load model: {e}")
            return False

    def preprocess_face(self, face_img):
        """Preprocess face image for model input"""
        try:
            # Resize to model input size
            face_resized = cv2.resize(face_img, (128, 128))
            
            # Convert to RGB
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values
            face_normalized = face_rgb.astype('float32') / 255.0
            
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            
            return face_batch
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None

    def check_liveness(self, frame, face_location):
        """
        Check if the face is real using the deep learning model
        
        Args:
            frame: Input BGR image
            face_location: Face location as (top, right, bottom, left)
            
        Returns:
            Boolean indicating if the face is real (True) or spoof (False)
        """
        try:
            if not self.model_loaded:
                return False
                
            # Extract face region
            top, right, bottom, left = face_location
            face_region = frame[top:bottom, left:right]
            
            if face_region.size == 0:
                return False
            
            # Preprocess face
            face_batch = self.preprocess_face(face_region)
            if face_batch is None:
                return False
            
            # Get model prediction with reduced verbosity
            prediction = self.model.predict(face_batch, verbose=0)[0]
            confidence = prediction[1]  # Probability of being real
            
            # Only log prediction if it's significantly different from previous
            if not hasattr(self, '_last_confidence'):
                self._last_confidence = 0
                print(f"Liveness confidence: {confidence:.2f}")
            elif abs(confidence - self._last_confidence) > 0.05:  # Only log if change > 5%
                print(f"Liveness confidence: {confidence:.2f}")
            
            self._last_confidence = confidence
            
            # Threshold for real face
            is_real = confidence > 0.80
        
            
            return is_real
            
        except Exception as e:
            print(f"Error in liveness detection: {e}")
            return False

    def train_model(self, real_faces_dir, fake_faces_dir, epochs=50, batch_size=64, validation_data=None):
        """
        Train the liveness detection model
        
        Args:
            real_faces_dir: Directory containing real face images
            fake_faces_dir: Directory containing fake face images
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_data: Dictionary containing validation data paths
        """
        try:
            print("\nStarting model training process...")
            
            # Load and preprocess training data
            print("\nLoading training data...")
            real_images = self._load_images(real_faces_dir, label=1)
            fake_images = self._load_images(fake_faces_dir, label=0)
            
            if len(real_images[0]) == 0 or len(fake_images[0]) == 0:
                print("Error: Could not load training data. Please check the image directories.")
                return
            
            print(f"\nTraining data loaded:")
            print(f"Real faces: {len(real_images[0])}")
            print(f"Fake faces: {len(fake_images[0])}")
            
            # Combine datasets
            X = np.concatenate([real_images[0], fake_images[0]])
            y = np.concatenate([real_images[1], fake_images[1]])
            
            print(f"\nCombined training data shape:")
            print(f"X shape: {X.shape}")
            print(f"y shape: {y.shape}")
            
            # Shuffle the data
            indices = np.random.permutation(len(X))
            X = X[indices]
            y = y[indices]
            
            # Prepare validation data if provided
            validation_data_tuple = None
            if validation_data:
                print("\nLoading validation data...")
                val_real_images = self._load_images(validation_data['real_dir'], label=1)
                val_fake_images = self._load_images(validation_data['fake_dir'], label=0)
                
                if len(val_real_images[0]) > 0 and len(val_fake_images[0]) > 0:
                    X_val = np.concatenate([val_real_images[0], val_fake_images[0]])
                    y_val = np.concatenate([val_real_images[1], val_fake_images[1]])
                    validation_data_tuple = (X_val, y_val)
                    print(f"\nValidation data loaded:")
                    print(f"X_val shape: {X_val.shape}")
                    print(f"y_val shape: {y_val.shape}")
                else:
                    print("Warning: Could not load validation data. Proceeding without validation.")
            
            # Create model architecture
            print("\nCreating model architecture...")
            model = Sequential([
                # First convolutional block
                Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                # Second convolutional block
                Conv2D(64, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                # Third convolutional block
                Conv2D(128, (3, 3), activation='relu'),
                MaxPooling2D(2, 2),
                Dropout(0.25),
                
                # Dense layers
                Flatten(),
                Dense(256, activation='relu'),
                Dropout(0.5),
                Dense(2, activation='softmax')
            ])
            
            # Compile model
            print("\nCompiling model...")
            model.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Create data generator
            print("\nSetting up data augmentation...")
            datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                rotation_range=20,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest'
            )
            
            # Calculate steps per epoch
            steps_per_epoch = max(1, len(X) // batch_size)
            print(f"\nTraining parameters:")
            print(f"Batch size: {batch_size}")
            print(f"Steps per epoch: {steps_per_epoch}")
            print(f"Epochs: {epochs}")
            
            # Train model
            print("\nStarting training...")
            model.fit(
                datagen.flow(X, y, batch_size=batch_size),
                epochs=epochs,
                validation_data=validation_data_tuple,
                steps_per_epoch=steps_per_epoch,
                callbacks=[
                    tf.keras.callbacks.EarlyStopping(
                        monitor='val_loss' if validation_data_tuple else 'loss',
                        patience=10,
                        restore_best_weights=True
                    )
                ]
            )
            
            # Save model
            print("\nSaving model...")
            model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
            os.makedirs(model_dir, exist_ok=True)
            model.save(os.path.join(model_dir, "liveness_model.h5"))
            
            self.model = model
            self.model_loaded = True
            
            print("\nModel training completed successfully!")
            
        except Exception as e:
            print(f"\nError training model: {e}")
            import traceback
            traceback.print_exc()

    def _load_images(self, directory, label):
        """Load and preprocess images from directory"""
        images = []
        labels = []
        
        print(f"\nLoading images from {directory}")
        print(f"Label: {label}")
        
        if not os.path.exists(directory):
            print(f"Error: Directory does not exist: {directory}")
            return np.array([]), np.array([])
            
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"Found {len(image_files)} image files")
        
        for filename in image_files:
            try:
                # Load image
                img_path = os.path.join(directory, filename)
                img = cv2.imread(img_path)
                
                if img is None:
                    print(f"Warning: Could not load image {filename}")
                    continue
                
                # Preprocess
                img = cv2.resize(img, (128, 128))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype('float32') / 255.0
                
                images.append(img)
                
                # Create one-hot encoded label
                label_one_hot = np.zeros(2)
                label_one_hot[label] = 1
                labels.append(label_one_hot)
                
            except Exception as e:
                print(f"Error loading image {filename}: {e}")
                continue
        
        if not images:
            print(f"Warning: No valid images loaded from {directory}")
            return np.array([]), np.array([])
            
        print(f"Successfully loaded {len(images)} images")
        return np.array(images), np.array(labels) 