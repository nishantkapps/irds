#!/usr/bin/env python3
"""
Video to Gesture Classification Pipeline
Converts video to skeleton data and classifies gestures using trained CLIP model
"""

import cv2
import torch
import json
import os
import sys
import argparse
from typing import List, Tuple, Optional
import mediapipe as mp

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_architectures import get_model
from tensor_utils import tensor_scaler_transform

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
    
    def process_video(self, video_path: str) -> torch.Tensor:
        """Extract skeleton data from video"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        skeleton_sequences = []
        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with MediaPipe
            results = self.pose.process(frame_rgb)
            
            if results.pose_landmarks:
                # Extract 25 landmarks (x, y, z) = 75 features
                # MediaPipe outputs normalized coords [0,1], but training data uses world coordinates
                # Convert to match training data range (centered around 0, scaled appropriately)
                landmarks = []
                for landmark in results.pose_landmarks.landmark[:25]:
                    # Convert from [0,1] to centered coordinates matching training data
                    # Training data has range roughly [-0.5, 0.5] for x,y and [2, 3] for z
                    x = (landmark.x - 0.5) * 1.0  # Center and scale x
                    y = (landmark.y - 0.5) * 1.0  # Center and scale y  
                    z = landmark.z * 1.0 + 2.5    # Scale z to match training range
                    landmarks.extend([x, y, z])
                
                skeleton_sequences.append(torch.tensor(landmarks, dtype=torch.float32))
                frame_count += 1
        
        cap.release()
        
        if len(skeleton_sequences) == 0:
            raise ValueError(f"No skeleton data extracted from video: {video_path}")
        
        skeleton_data = torch.stack(skeleton_sequences)
        print(f"Extracted {frame_count} frames with skeleton data")
        print(f"Skeleton data shape: {skeleton_data.shape}")
        
        return skeleton_data

class GestureClassifier:
    """Classify gestures using trained CLIP model"""
    
    def __init__(self, model_path: str):
        """Load trained model and preprocessing"""
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.gesture_names = checkpoint['gesture_names']
        self.gesture_descriptions = checkpoint['gesture_descriptions']
        self.sequence_length = checkpoint.get('sequence_length', 50)
        # Move scaler params to device
        scaler_params = checkpoint['scaler_params']
        self.scaler_params = {
            'mean': scaler_params['mean'].to(self.device),
            'std': scaler_params['std'].to(self.device)
        }
        
        # Load model
        model_arch = checkpoint.get('architecture', 'medium')
        self.model = get_model(
            model_arch,
            input_size=75,
            num_classes=len(self.gesture_names),
            device=str(self.device)
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        print(f"Loaded model architecture: {model_arch}")
        print(f"Number of gestures: {len(self.gesture_names)}")
        print(f"Gesture names: {self.gesture_names}")
    
    def preprocess_skeleton_data(self, skeleton_data: torch.Tensor) -> torch.Tensor:
        """Preprocess skeleton data for model input"""
        # Create sequences of specified length first
        sequences = []
        for i in range(skeleton_data.size(0) - self.sequence_length + 1):
            sequence = skeleton_data[i:i + self.sequence_length]
            sequences.append(sequence)
        
        if len(sequences) == 0:
            # Pad if video is too short
            if skeleton_data.size(0) < self.sequence_length:
                # Repeat frames to reach sequence length
                repeats = (self.sequence_length // skeleton_data.size(0)) + 1
                sequence = skeleton_data.repeat(repeats, 1)[:self.sequence_length]
            else:
                sequence = skeleton_data[:self.sequence_length]
            sequences = [sequence]
        
        # Stack and flatten sequences
        sequences_tensor = torch.stack(sequences)  # [num_seq, seq_len, 75]
        batch_size = sequences_tensor.size(0)
        flattened = sequences_tensor.view(batch_size, -1).to(self.device)  # [num_seq, seq_len * 75]
        
        # Normalize flattened sequences
        normalized = tensor_scaler_transform(flattened, self.scaler_params)
        
        # Reshape back to [batch, seq_len, features]
        sequences_normalized = normalized.view(batch_size, self.sequence_length, -1).to(self.device)
        
        return sequences_normalized
    
    def classify_gesture(self, skeleton_data: torch.Tensor) -> Tuple[str, str, float]:
        """Classify gesture from skeleton data"""
        # Preprocess data
        sequences = self.preprocess_skeleton_data(skeleton_data)
        
        # Get predictions for all sequences
        with torch.no_grad():
            outputs = self.model(sequences)
            probs = torch.softmax(outputs, dim=1)
            
            # Average predictions across all sequences
            avg_probs = probs.mean(dim=0)
            pred_idx = avg_probs.argmax().item()
            confidence = avg_probs[pred_idx].item()
        
        gesture_name = self.gesture_names[pred_idx] if pred_idx < len(self.gesture_names) else f"Unknown ({pred_idx})"
        gesture_desc = self.gesture_descriptions[pred_idx] if pred_idx < len(self.gesture_descriptions) else "No description"
        
        return gesture_name, gesture_desc, confidence

def main():
    parser = argparse.ArgumentParser(description='Classify gestures from video')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--model', required=True, help='Path to trained model (.pth)')
    parser.add_argument('--output', help='Path to save results (optional)')
    args = parser.parse_args()
    
    print("=== Video to Gesture Classification ===")
    
    # Check if files exist
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        return
    
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Extract skeleton data from video
    print(f"\n=== Extracting Skeleton Data ===")
    print(f"Video: {args.video}")
    skeleton_extractor = VideoToSkeleton()
    skeleton_data = skeleton_extractor.process_video(args.video)
    
    # Classify gesture
    print(f"\n=== Classifying Gesture ===")
    print(f"Model: {args.model}")
    classifier = GestureClassifier(args.model)
    gesture_name, gesture_desc, confidence = classifier.classify_gesture(skeleton_data)
    
    # Print results
    print(f"\n=== Results ===")
    print(f"Gesture: {gesture_name}")
    print(f"Description: {gesture_desc}")
    print(f"Confidence: {confidence:.2%}")
    
    # Save results if requested
    if args.output:
        results = {
            'video': args.video,
            'gesture_name': gesture_name,
            'gesture_description': gesture_desc,
            'confidence': float(confidence)
        }
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

if __name__ == '__main__':
    main()
