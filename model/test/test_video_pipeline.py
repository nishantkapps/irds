#!/usr/bin/env python3
"""
Test the video-to-gesture pipeline with a sample video
"""

import os
import cv2
import numpy as np
from video_to_gesture import VideoToSkeleton, GestureClassifier

def create_sample_video(output_path: str, duration: int = 5, fps: int = 30):
    """Create a sample video for testing"""
    print(f"Creating sample video: {output_path}")
    
    # Video properties
    width, height = 640, 480
    total_frames = duration * fps
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    for frame_num in range(total_frames):
        # Create a simple frame with moving circle (simulating a person)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Add some movement
        center_x = int(width/2 + 50 * np.sin(2 * np.pi * frame_num / fps))
        center_y = int(height/2 + 30 * np.cos(2 * np.pi * frame_num / fps))
        
        # Draw a simple "person" (circle for head, rectangle for body)
        cv2.circle(frame, (center_x, center_y - 50), 20, (255, 255, 255), -1)
        cv2.rectangle(frame, (center_x - 30, center_y - 30), (center_x + 30, center_y + 50), (255, 255, 255), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame {frame_num}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(frame)
    
    out.release()
    print(f"Sample video created: {output_path}")

def test_skeleton_extraction(video_path: str):
    """Test skeleton extraction from video"""
    print(f"\n=== Testing Skeleton Extraction ===")
    
    skeleton_extractor = VideoToSkeleton()
    
    # Process video
    skeleton_data = skeleton_extractor.process_video(video_path)
    
    print(f"Skeleton data shape: {skeleton_data.shape}")
    print(f"Number of frames: {len(skeleton_data)}")
    print(f"Features per frame: {skeleton_data.shape[1] if len(skeleton_data.shape) > 1 else 0}")
    
    # Show sample skeleton data
    if len(skeleton_data) > 0:
        print(f"Sample skeleton (first frame): {skeleton_data[0][:10]}...")  # Show first 10 values
    
    return skeleton_data

def test_gesture_classification(skeleton_data: np.ndarray):
    """Test gesture classification (mock)"""
    print(f"\n=== Testing Gesture Classification ===")
    
    # Mock classification (replace with actual model)
    gesture_names = {
        "0": "Elbow Flexion Left",
        "1": "Elbow Flexion Right", 
        "2": "Shoulder Abduction Left",
        "3": "Shoulder Abduction Right",
        "4": "Knee Flexion Left"
    }
    
    # Simulate classification
    predicted_class = np.random.randint(0, len(gesture_names))
    confidence = np.random.uniform(0.7, 0.95)
    
    gesture_name = gesture_names[str(predicted_class)]
    
    print(f"Predicted Gesture: {gesture_name}")
    print(f"Confidence: {confidence:.3f}")
    
    return gesture_name, confidence

def main():
    print("=== Video-to-Gesture Pipeline Test ===")
    
    # Create sample video
    sample_video = "sample_gesture_video.mp4"
    if not os.path.exists(sample_video):
        create_sample_video(sample_video, duration=3, fps=30)
    
    # Test skeleton extraction
    skeleton_data = test_skeleton_extraction(sample_video)
    
    # Test gesture classification
    gesture_name, confidence = test_gesture_classification(skeleton_data)
    
    print(f"\n=== Test Results ===")
    print(f"Video processed: {sample_video}")
    print(f"Skeleton data extracted: {skeleton_data.shape}")
    print(f"Predicted gesture: {gesture_name}")
    print(f"Confidence: {confidence:.3f}")
    
    # Cleanup
    if os.path.exists(sample_video):
        os.remove(sample_video)
        print(f"Cleaned up sample video: {sample_video}")

if __name__ == "__main__":
    main()
