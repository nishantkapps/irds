"""
Demo script for gesture recognition
Shows how to use the trained model to predict gestures from skeleton data
"""

import torch
import pandas as pd
import joblib
from simple_gesture_model import SimpleGestureModel, predict_gesture_simple
from irds_eda import load_irds_data, load_gesture_labels
import matplotlib.pyplot as plt


def load_trained_model(model_path: str = 'simple_gesture_model.pth',
                      scaler_path: str = 'simple_gesture_scaler.pkl'):
    """Load the trained model and scaler"""
    
    # Load gesture names
    gesture_labels = load_gesture_labels()
    gesture_names = [gesture_labels.get(str(i), f"Gesture {i}") for i in range(9)]
    
    # Load scaler
    scaler = joblib.load(scaler_path)
    
    # Create and load model
    # We need to know the input dimension - let's get it from a sample
    df = load_irds_data(folder_path="/home/nishant/project/irds/data", max_files=1)
    numeric_cols = df.select_dtypes(include=[torch.number]).columns.tolist()
    metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                    'correct_label', 'position']
    skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
    input_dim = len(skeleton_cols)
    
    model = SimpleGestureModel(input_dim, num_classes=len(gesture_names))
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    return model, scaler, gesture_names


def demo_prediction():
    """Demo prediction on a sample from the dataset"""
    
    print("Loading trained model...")
    model, scaler, gesture_names = load_trained_model()
    
    print("Loading sample data...")
    # Load a small sample
    df = load_irds_data(folder_path="/home/nishant/project/irds/data", max_files=2)
    df = df.dropna(subset=['gesture_label'])
    
    # Get skeleton columns
    numeric_cols = df.select_dtypes(include=[torch.number]).columns.tolist()
    metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                    'correct_label', 'position']
    skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Get a few samples for prediction
    sample_indices = [0, 10, 20, 30, 40]  # Different frames
    
    print(f"\nPredicting gestures for {len(sample_indices)} samples:")
    print("=" * 60)
    
    for i, idx in enumerate(sample_indices):
        if idx < len(df):
            # Get skeleton data
            skeleton_data = df.iloc[idx][skeleton_cols].values
            
            # Get actual gesture
            actual_gesture_label = int(df.iloc[idx]['gesture_label'])
            actual_gesture = gesture_names[actual_gesture_label]
            
            # Predict gesture
            predicted_gesture, confidence = predict_gesture_simple(
                model, scaler, skeleton_data, gesture_names
            )
            
            # Check if prediction is correct
            is_correct = predicted_gesture == actual_gesture
            
            print(f"Sample {i+1}:")
            print(f"  Actual: {actual_gesture}")
            print(f"  Predicted: {predicted_gesture} (confidence: {confidence:.3f})")
            print(f"  Correct: {'✓' if is_correct else '✗'}")
            print()


def batch_prediction_demo():
    """Demo batch prediction on multiple samples"""
    
    print("Loading trained model...")
    model, scaler, gesture_names = load_trained_model()
    
    print("Loading test data...")
    # Load test data
    df = load_irds_data(folder_path="/home/nishant/project/irds/data", max_files=3)
    df = df.dropna(subset=['gesture_label'])
    
    # Get skeleton columns
    numeric_cols = df.select_dtypes(include=[torch.number]).columns.tolist()
    metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                    'correct_label', 'position']
    skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Take first 50 samples for testing
    test_df = df.head(50)
    
    print(f"Testing on {len(test_df)} samples...")
    
    # Batch prediction
    X_test = test_df[skeleton_cols].values
    y_test = test_df['gesture_label'].astype(int).values
    
    # Normalize
    X_test_scaled = scaler.transform(X_test)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    
    # Predict
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1).numpy()
        confidences = torch.max(probabilities, dim=1)[0].numpy()
    
    # Calculate accuracy
    accuracy = torch.mean((torch.tensor(predictions) == y_test).float()).item()
    
    print(f"Batch Accuracy: {accuracy:.3f}")
    print(f"Average Confidence: {torch.mean(torch.tensor(confidences)).item():.3f}")
    
    # Show some results
    print("\nSample Results:")
    print("=" * 60)
    for i in range(min(10, len(test_df))):
        actual = gesture_names[y_test[i]]
        predicted = gesture_names[predictions[i]]
        confidence = confidences[i]
        is_correct = predictions[i] == y_test[i]
        
        print(f"Sample {i+1}: {actual} -> {predicted} ({confidence:.3f}) {'✓' if is_correct else '✗'}")


def visualize_predictions():
    """Visualize prediction confidence across different gestures"""
    
    print("Loading trained model...")
    model, scaler, gesture_names = load_trained_model()
    
    print("Loading data for visualization...")
    # Load data
    df = load_irds_data(folder_path="/home/nishant/project/irds/data", max_files=5)
    df = df.dropna(subset=['gesture_label'])
    
    # Get skeleton columns
    numeric_cols = df.select_dtypes(include=[torch.number]).columns.tolist()
    metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                    'correct_label', 'position']
    skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    # Group by gesture and get samples
    gesture_samples = {}
    for gesture in df['gesture_label'].unique():
        gesture_df = df[df['gesture_label'] == gesture].head(10)  # 10 samples per gesture
        gesture_samples[gesture] = gesture_df[skeleton_cols].values
    
    # Create visualization
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (gesture, samples) in enumerate(gesture_samples.items()):
        if i >= 9:  # Only show first 9 gestures
            break
            
        # Predict on samples
        X_scaled = scaler.transform(samples)
        X_tensor = torch.FloatTensor(X_scaled)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0].numpy()
        
        # Plot confidence distribution
        axes[i].hist(confidences, bins=10, alpha=0.7, edgecolor='black')
        axes[i].set_title(f'{gesture_names[gesture]} (n={len(samples)})')
        axes[i].set_xlabel('Prediction Confidence')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/gesture_confidence_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    print("Gesture Recognition Demo")
    print("=" * 40)
    
    try:
        # Demo 1: Single predictions
        print("\n1. Single Sample Predictions:")
        demo_prediction()
        
        # Demo 2: Batch predictions
        print("\n2. Batch Predictions:")
        batch_prediction_demo()
        
        # Demo 3: Visualization
        print("\n3. Confidence Visualization:")
        visualize_predictions()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please train the model first by running: python simple_gesture_model.py")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model files exist and the data is accessible.")

