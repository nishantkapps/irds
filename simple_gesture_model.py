"""
Simple Gesture Recognition Model for IRDS Dataset
A lightweight version for quick testing and prototyping
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from typing import List, Tuple
import joblib

# Import our existing functions
import sys
sys.path.append('/home/nishant/project/irds')
from irds_eda import load_irds_data, load_gesture_labels


class SimpleGestureDataset(Dataset):
    """Simple dataset for gesture recognition"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray):
        self.data = torch.FloatTensor(data)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SimpleGestureModel(nn.Module):
    """Simple neural network for gesture recognition"""
    
    def __init__(self, input_dim: int, num_classes: int, hidden_dim: int = 128):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)


def prepare_simple_data(folder_path: str = "/home/nishant/project/irds/data",
                       max_files: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Prepare data for simple gesture recognition"""
    
    print("Loading IRDS data...")
    df = load_irds_data(folder_path=folder_path, max_files=max_files)
    
    # Load gesture labels
    gesture_labels = load_gesture_labels()
    
    # Filter out rows with missing gesture labels
    df = df.dropna(subset=['gesture_label'])
    df = df[df['gesture_label'].notna()]
    
    print(f"Loaded {len(df)} rows from {max_files} files")
    
    # Get numeric columns (skeleton data)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove metadata columns
    metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                    'correct_label', 'position']
    skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    print(f"Using {len(skeleton_cols)} skeleton columns")
    
    # Get skeleton data and labels
    X = df[skeleton_cols].values
    y = df['gesture_label'].astype(int).values
    
    # Get unique gesture names
    unique_labels = sorted(df['gesture_label'].unique())
    gesture_names = [gesture_labels.get(str(label), f"Gesture {label}") for label in unique_labels]
    
    print(f"Data shape: {X.shape}")
    print(f"Gesture classes: {gesture_names}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, gesture_names


def train_simple_model(X: np.ndarray, y: np.ndarray, gesture_names: List[str],
                      test_size: float = 0.2, num_epochs: int = 20,
                      learning_rate: float = 0.001, batch_size: int = 32):
    """Train the simple gesture model"""
    
    print("Preparing data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Create datasets
    train_dataset = SimpleGestureDataset(X_train_scaled, y_train)
    test_dataset = SimpleGestureDataset(X_test_scaled, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    input_dim = X.shape[1]
    num_classes = len(gesture_names)
    model = SimpleGestureModel(input_dim, num_classes)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Training...")
    
    # Training loop
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    for epoch in range(num_epochs):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for data, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Test
        model.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in test_loader:
                outputs = model(data)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.numpy())
                all_labels.extend(labels.numpy())
        
        test_loss /= len(test_loader)
        test_acc = 100. * test_correct / test_total
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch:2d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Final results
    print(f"\nFinal Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_predictions, target_names=gesture_names))
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Test Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Test Accuracy')
    
    plt.tight_layout()
    plt.savefig('simple_training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save model and scaler
    torch.save(model.state_dict(), 'simple_gesture_model.pth')
    joblib.dump(scaler, 'simple_gesture_scaler.pkl')
    
    print(f"\nModel saved as 'simple_gesture_model.pth'")
    print(f"Scaler saved as 'simple_gesture_scaler.pkl'")
    
    return model, scaler, gesture_names


def predict_gesture_simple(model, scaler, skeleton_data: np.ndarray, gesture_names: List[str]) -> Tuple[str, float]:
    """Predict gesture from skeleton data"""
    model.eval()
    
    # Normalize data
    skeleton_scaled = scaler.transform(skeleton_data.reshape(1, -1))
    skeleton_tensor = torch.FloatTensor(skeleton_scaled)
    
    with torch.no_grad():
        outputs = model(skeleton_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    predicted_gesture = gesture_names[predicted_class]
    
    return predicted_gesture, confidence


if __name__ == "__main__":
    print("Simple Gesture Recognition Model")
    print("=" * 40)
    
    # Prepare data
    X, y, gesture_names = prepare_simple_data(
        folder_path="/home/nishant/project/irds/data",
        max_files=5  # Start with just 5 files for quick testing
    )
    
    # Train model
    model, scaler, gesture_names = train_simple_model(
        X, y, gesture_names,
        test_size=0.2,
        num_epochs=20,
        learning_rate=0.001,
        batch_size=32
    )
    
    print("\nTraining completed!")
    print("You can now use the model to predict gestures from skeleton data.")

