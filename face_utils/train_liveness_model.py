import os
import cv2
import numpy as np
from liveness_model import LivenessDetector

def extract_frames_from_videos(video_dir, output_dir, label, max_frames_per_video=50):
    """Extract frames from videos and save them as images"""
    print(f"\nExtracting frames from videos in {video_dir}")
    print(f"Output directory: {output_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all video files
    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith(('.mp4', '.avi', '.mov'))]
    print(f"Found {len(video_files)} video files")
    
    total_frames = 0
    for video_file in video_files:
        try:
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                print(f"Warning: Could not open video {video_file}")
                continue
            
            # Get video properties
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Calculate frame interval to get desired number of frames
            frame_interval = max(1, frame_count // max_frames_per_video)
            
            frame_idx = 0
            saved_frames = 0
            
            while saved_frames < max_frames_per_video:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # Save frame
                    output_path = os.path.join(output_dir, f"{label}_{video_file}_{saved_frames:04d}.jpg")
                    cv2.imwrite(output_path, frame)
                    saved_frames += 1
                    total_frames += 1
                
                frame_idx += 1
            
            cap.release()
            print(f"Extracted {saved_frames} frames from {video_file}")
            
        except Exception as e:
            print(f"Error processing video {video_file}: {e}")
            continue
    
    print(f"\nTotal frames extracted: {total_frames}")
    return total_frames

def train_liveness_model():
    # Initialize liveness detector
    detector = LivenessDetector()
    
    # Set paths to your dataset
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "videos")
    
    # Video directories
    train_real_videos = os.path.join(base_dir, "train", "real")
    train_fake_videos = os.path.join(base_dir, "train", "fake")
    test_real_videos = os.path.join(base_dir, "test", "real")
    test_fake_videos = os.path.join(base_dir, "test", "fake")
    
    # Output directories for extracted frames
    train_real_dir = os.path.join(base_dir, "train", "real_frames")
    train_fake_dir = os.path.join(base_dir, "train", "fake_frames")
    test_real_dir = os.path.join(base_dir, "test", "real_frames")
    test_fake_dir = os.path.join(base_dir, "test", "fake_frames")
    
    # Check if video directories exist
    for dir_path in [train_real_videos, train_fake_videos, test_real_videos, test_fake_videos]:
        if not os.path.exists(dir_path):
            print(f"Error: Video directory not found: {dir_path}")
            return
    
    print("\n=== Extracting Frames from Videos ===")
    
    # Extract frames from training videos
    print("\nProcessing training videos...")
    train_real_frames = extract_frames_from_videos(train_real_videos, train_real_dir, "real")
    train_fake_frames = extract_frames_from_videos(train_fake_videos, train_fake_dir, "fake")
    
    # Extract frames from test videos
    print("\nProcessing test videos...")
    test_real_frames = extract_frames_from_videos(test_real_videos, test_real_dir, "real")
    test_fake_frames = extract_frames_from_videos(test_fake_videos, test_fake_dir, "fake")
    
    print(f"\n=== Dataset Summary ===")
    print("Training Data:")
    print(f"  Real faces: {train_real_frames}")
    print(f"  Fake faces: {train_fake_frames}")
    print("\nTest Data:")
    print(f"  Real faces: {test_real_frames}")
    print(f"  Fake faces: {test_fake_frames}")
    
    if train_real_frames < 50 or train_fake_frames < 50:
        print("\nWarning: Not enough training data!")
        print("Please ensure you have enough training samples.")
        return
    
    print("\nStarting model training...")
    
    # Train the model
    detector.train_model(
        real_faces_dir=train_real_dir,
        fake_faces_dir=train_fake_dir,
        epochs=50,
        batch_size=32,  # Reduced batch size for smaller dataset
        validation_data={
            'real_dir': test_real_dir,
            'fake_dir': test_fake_dir
        }
    )
    
    print("\nTraining complete!")
    print("Model saved to models/liveness_model.h5")

if __name__ == "__main__":
    train_liveness_model() 