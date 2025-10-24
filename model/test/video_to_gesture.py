#!/usr/bin/env python3
"""
Video to Gesture Classification Pipeline
Converts video to skeleton data and classifies gestures using trained CLIP model
"""

import cv2
import numpy as np
import torch
import torch.nn as nn
import joblib
import json
import os
import argparse
from typing import List, Tuple, Optional
import mediapipe as mp
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

class VideoToSkeleton:
    """Convert video frames to skeleton data using MediaPipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        
    def extract_skeleton_from_frame(self, frame):
        """Extract skeleton keypoints from a single frame"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Extract 25 keypoints (MediaPipe provides 33, we need 25)
            landmarks = results.pose_landmarks.landmark
            
            # Map MediaPipe landmarks to our 25-joint format
            # This mapping needs to be adjusted based on your joint definitions
            keypoint_indices = [
                11, 12, 13, 14, 15, 16,  # Arms
                23, 24, 25, 26, 27, 28,  # Legs
                0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,  # Torso and head
                17, 18, 19, 20, 21, 22   # Additional points
            ]
            
            skeleton_data = []
            for idx in keypoint_indices[:25]:  # Take first 25
                if idx < len(landmarks):
                    landmark = landmarks[idx]
                    skeleton_data.extend([landmark.x, landmark.y, landmark.z])
                else:
                    skeleton_data.extend([0.0, 0.0, 0.0])  # Pad if missing
            
            return np.array(skeleton_data)
        else:
            # Return zeros if no pose detected
            return np.zeros(75)  # 25 joints * 3 coordinates
    
    def process_video(self, video_path: str, output_fps: int = 30) -> np.ndarray:
        """Process entire video and extract skeleton sequences"""
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / original_fps
        
        print(f"Video: {video_path}")
        print(f"Original FPS: {original_fps:.2f}, Duration: {duration:.2f}s")
        print(f"Total frames: {total_frames}")
        
        skeleton_sequences = []
        frame_count = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract skeleton from frame
            skeleton = self.extract_skeleton_from_frame(frame)
            skeleton_sequences.append(skeleton)
            frame_count += 1
            
            if frame_count % 100 == 0:
                print(f"Processed {frame_count}/{total_frames} frames")
        
        cap.release()
        
        # Convert to numpy array
        skeleton_data = np.array(skeleton_sequences)
        print(f"Extracted skeleton data shape: {skeleton_data.shape}")
        
        return skeleton_data

class GestureClassifier:
    """Classify gestures using trained CLIP model"""
    
    def __init__(self, model_path: str, scaler_path: str, info_path: str):
        """Load trained model and preprocessing"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load model info
        with open(info_path, 'r') as f:
            self.model_info = json.load(f)
        
        self.gesture_names = self.model_info['gesture_names']
        self.gesture_descriptions = self.model_info['gesture_descriptions']
        self.sequence_length = self.model_info['sequence_length']
        
        # Load scaler
        self.scaler = joblib.load(scaler_path)
        
        # Load model (you'll need to import your model class)
        # For now, we'll create a placeholder
        self.model = self._load_model(model_path)
        self.model.eval()
        
        print(f"Loaded model for {len(self.gesture_names)} gestures")
        print(f"Gesture names: {list(self.gesture_names.values())}")
    
    def _load_model(self, model_path: str):
        """Load the trained model"""
        # This needs to be implemented based on your model architecture
        # For now, return a placeholder
        print(f"Loading model from {model_path}")
        # model = YourModelClass()
        # model.load_state_dict(torch.load(model_path, map_location=self.device))
        # return model.to(self.device)
        return None  # Placeholder
    
    def preprocess_skeleton_data(self, skeleton_data: np.ndarray) -> torch.Tensor:
        """Preprocess skeleton data for model input"""
        # Normalize skeleton data
        skeleton_normalized = self.scaler.transform(skeleton_data)
        
        # Create sequences of specified length
        sequences = []
        for i in range(len(skeleton_normalized) - self.sequence_length + 1):
            sequence = skeleton_normalized[i:i + self.sequence_length]
            sequences.append(sequence)
        
        if len(sequences) == 0:
            # Pad if video is too short
            sequence = np.tile(skeleton_normalized, (self.sequence_length, 1))
            sequences = [sequence]
        
        # Convert to tensor
        sequences = np.array(sequences)
        sequences_tensor = torch.FloatTensor(sequences).to(self.device)
        
        return sequences_tensor
    
    def classify_gesture(self, skeleton_data: np.ndarray) -> Tuple[str, str, float]:
        """Classify gesture from skeleton data"""
        # Preprocess data
        sequences = self.preprocess_skeleton_data(skeleton_data)
        
        # Get predictions for all sequences
        predictions = []
        with torch.no_grad():
            for sequence in sequences:
                # Add batch dimension
                sequence = sequence.unsqueeze(0)
                
                # Get model prediction (placeholder)
                # output = self.model(sequence)
                # prediction = torch.softmax(output, dim=1)
                # predictions.append(prediction.cpu().numpy())
                
                # Placeholder prediction
                predictions.append(np.random.random((1, len(self.gesture_names))))
        
        # Average predictions across all sequences
        avg_prediction = np.mean(predictions, axis=0)
        predicted_class = np.argmax(avg_prediction)
        confidence = float(avg_prediction[0, predicted_class])
        
        # Get gesture name and description
        gesture_name = self.gesture_names[str(predicted_class)]
        gesture_description = self.gesture_descriptions[str(predicted_class)]
        
        return gesture_name, gesture_description, confidence

def visualize_skeleton_on_video(video_path: str, output_path: str, gesture_name: str, confidence: float):
    """Create output video with skeleton overlay and gesture label"""
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize MediaPipe
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw skeleton
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)
        
        if results.pose_landmarks:
            # Draw pose landmarks
            mp.solutions.drawing_utils.draw_landmarks(
                frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Add gesture label
        label = f"Gesture: {gesture_name} ({confidence:.2f})"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        out.write(frame)
        frame_count += 1
    
    cap.release()
    out.release()
    print(f"Output video saved: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Video to Gesture Classification')
    parser.add_argument('--video', type=str, required=True, help='Input video path')
    parser.add_argument('--model', type=str, default='clip_gesture_model.pth', help='Model path')
    parser.add_argument('--scaler', type=str, default='clip_gesture_scaler.pkl', help='Scaler path')
    parser.add_argument('--info', type=str, default='clip_gesture_info.json', help='Model info path')
    parser.add_argument('--output', type=str, default='output_video.mp4', help='Output video path')
    parser.add_argument('--visualize', action='store_true', help='Create output video with skeleton')
    
    args = parser.parse_args()
    
    print("=== Video to Gesture Classification ===")
    
    # Check if files exist
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Initialize components
    print("\n=== Initializing Components ===")
    skeleton_extractor = VideoToSkeleton()
    classifier = GestureClassifier(args.model, args.scaler, args.info)
    
    # Process video
    print("\n=== Processing Video ===")
    skeleton_data = skeleton_extractor.process_video(args.video)
    
    # Classify gesture
    print("\n=== Classifying Gesture ===")
    gesture_name, gesture_description, confidence = classifier.classify_gesture(skeleton_data)
    
    # Print results
    print(f"\n=== Results ===")
    print(f"Predicted Gesture: {gesture_name}")
    print(f"Description: {gesture_description}")
    print(f"Confidence: {confidence:.3f}")
    
    # Create output video if requested
    if args.visualize:
        print(f"\n=== Creating Output Video ===")
        visualize_skeleton_on_video(args.video, args.output, gesture_name, confidence)
    
    print("\n=== Classification Complete ===")

if __name__ == "__main__":
    main()
