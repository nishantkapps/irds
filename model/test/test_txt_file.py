#!/usr/bin/env python3
"""
Test trained model on skeleton .txt files directly
"""

import torch
import os
import sys
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from model.model_architectures import get_model
from tensor_utils import tensor_scaler_transform

class GestureClassifier:
    """Classify gestures using trained model"""
    
    def __init__(self, model_path: str):
        """Load trained model"""
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
        
        print(f"Loaded model: {model_arch}")
        print(f"Gestures: {self.gesture_names}")
    
    def load_skeleton_file(self, txt_path: str) -> torch.Tensor:
        """Load skeleton data from .txt file"""
        try:
            # Read file line by line (comma-separated)
            data = []
            with open(txt_path, 'r') as f:
                for line in f:
                    # Split by comma and convert to float
                    values = []
                    for val in line.strip().split(','):
                        try:
                            values.append(float(val))
                        except ValueError:
                            values.append(0.0)  # Replace non-numeric with 0
                    if values:
                        data.append(values)
            
            skeleton_data = torch.tensor(data, dtype=torch.float32)
            
            # Validate shape - should be [frames, 75]
            if skeleton_data.size(1) == 74:
                # Pad with zeros to 75
                skeleton_data = torch.cat([skeleton_data, torch.zeros(skeleton_data.size(0), 1)], dim=1)
            elif skeleton_data.size(1) > 75:
                # Trim to 75
                skeleton_data = skeleton_data[:, :75]
            elif skeleton_data.size(1) < 74:
                raise ValueError(f"Expected at least 74 features, got {skeleton_data.size(1)}")
            
            return skeleton_data
        except Exception as e:
            raise ValueError(f"Error loading {txt_path}: {e}")
    
    def classify(self, skeleton_data: torch.Tensor):
        """Classify gesture from skeleton data"""
        # Create sequences
        sequences = []
        for i in range(skeleton_data.size(0) - self.sequence_length + 1):
            sequence = skeleton_data[i:i + self.sequence_length]
            sequences.append(sequence)
        
        if len(sequences) == 0:
            # Pad if too short
            if skeleton_data.size(0) < self.sequence_length:
                repeats = (self.sequence_length // skeleton_data.size(0)) + 1
                sequence = skeleton_data.repeat(repeats, 1)[:self.sequence_length]
            else:
                sequence = skeleton_data[:self.sequence_length]
            sequences = [sequence]
        
        # Stack and flatten
        sequences_tensor = torch.stack(sequences)
        batch_size = sequences_tensor.size(0)
        flattened = sequences_tensor.view(batch_size, -1).to(self.device)
        
        # Normalize
        normalized = tensor_scaler_transform(flattened, self.scaler_params)
        
        # Reshape
        sequences_normalized = normalized.view(batch_size, self.sequence_length, -1)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(sequences_normalized)
            probs = torch.softmax(outputs, dim=1)
            avg_probs = probs.mean(dim=0)
            pred_idx = avg_probs.argmax().item()
            confidence = avg_probs[pred_idx].item()
        
        gesture_name = self.gesture_names[pred_idx]
        gesture_desc = self.gesture_descriptions[pred_idx]
        
        return gesture_name, gesture_desc, confidence, avg_probs

def main():
    parser = argparse.ArgumentParser(description='Test model on .txt skeleton file')
    parser.add_argument('--txt', required=True, help='Path to skeleton .txt file')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--true-label', help='True gesture name (optional, for verification)')
    args = parser.parse_args()
    
    print("=== Testing Gesture Classification ===")
    print(f"File: {args.txt}")
    print(f"Model: {args.model}")
    
    # Load classifier
    classifier = GestureClassifier(args.model)
    
    # Load skeleton data
    print(f"\n=== Loading Skeleton Data ===")
    skeleton_data = classifier.load_skeleton_file(args.txt)
    print(f"Skeleton shape: {skeleton_data.shape}")
    print(f"Value range: [{skeleton_data.min():.3f}, {skeleton_data.max():.3f}]")
    
    # Classify
    print(f"\n=== Classification ===")
    gesture_name, gesture_desc, confidence, all_probs = classifier.classify(skeleton_data)
    
    print(f"\nPredicted: {gesture_name}")
    print(f"Description: {gesture_desc}")
    print(f"Confidence: {confidence:.2%}")
    
    if args.true_label:
        print(f"True label: {args.true_label}")
        print(f"Correct: {gesture_name == args.true_label}")
    
    print(f"\n=== All Class Probabilities ===")
    for i, (name, prob) in enumerate(zip(classifier.gesture_names, all_probs)):
        print(f"{i}. {name}: {prob:.2%}")

if __name__ == '__main__':
    main()

