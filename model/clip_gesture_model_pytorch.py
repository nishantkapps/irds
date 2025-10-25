#!/usr/bin/env python3
"""
CLIP-style Gesture Recognition Model - PyTorch Only Version
Replaces NumPy operations with PyTorch to avoid GPU conflicts
"""

import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import joblib
import json

# Import project utilities
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import get_data_path, get_logger
import pandas as pd

# PyTorch-only data loading functions
def load_gesture_labels(labels_path: str = "../../data/labels.csv") -> dict:
    """Load gesture labels from CSV file."""
    logger = get_logger()
    try:
        labels_df = pd.read_csv(labels_path)
        return dict(zip(labels_df['GestureIndex'].astype(str), labels_df['GestureName']))
    except FileNotFoundError:
        logger.info(f"Warning: Labels file not found at {labels_path}")
        return {}
    except Exception as e:
        logger.info(f"Warning: Could not load labels: {e}")
        return {}

def load_irds_data(folder_path: Optional[str] = None,
                   file_pattern: str = "*.txt",
                   has_header: bool = False,
                   add_metadata: bool = True,
                   columns: Optional[List[str]] = None,
                   include_source_file: bool = True,
                   max_files: Optional[int] = None) -> pd.DataFrame:
    """Load IRDS dataset files into a combined pandas DataFrame."""
    logger = get_logger()
    if folder_path is None:
        folder_path = str(get_data_path())
    
    search_path = os.path.join(folder_path, file_pattern)
    all_files = glob.glob(search_path)
    if len(all_files) == 0:
        raise FileNotFoundError(f"No files found at {search_path}")
    
    # Limit number of files for faster loading
    if max_files is not None and len(all_files) > max_files:
        logger.info(f"Loading only first {max_files} files out of {len(all_files)} for faster startup")
        all_files = all_files[:max_files]

    header = 0 if has_header else None
    list_of_dfs: List[pd.DataFrame] = []
    
    for file_path in all_files:
        df = pd.read_csv(file_path, header=header)

        # Assign column names if provided and no header present
        if not has_header and columns is not None:
            if len(columns) != df.shape[1]:
                raise ValueError(
                    f"Provided columns length {len(columns)} does not match file columns {df.shape[1]} for {file_path}"
                )
            df.columns = columns

        if add_metadata:
            filename = os.path.basename(file_path)
            name, _ext = os.path.splitext(filename)
            parts = name.split("_")
            # Expected: subject_id, date_id, gesture_label, rep_number, correct_label, position
            if len(parts) >= 6:
                subject_id, date_id, gesture_label, rep_number, correct_label, position = parts[:6]
            else:
                # Fallback: pad missing parts with None
                subject_id = parts[0] if len(parts) > 0 else None
                date_id = parts[1] if len(parts) > 1 else None
                gesture_label = parts[2] if len(parts) > 2 else None
                rep_number = parts[3] if len(parts) > 3 else None
                correct_label = parts[4] if len(parts) > 4 else None
                position = parts[5] if len(parts) > 5 else None

            df["subject_id"] = subject_id
            df["date_id"] = date_id
            df["gesture_label"] = gesture_label
            df["rep_number"] = rep_number
            df["correct_label"] = correct_label
            df["position"] = position

        if include_source_file:
            df["source_file"] = os.path.basename(file_path)

        list_of_dfs.append(df)

    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    return combined_df

def prepare_clip_gesture_data_pytorch(folder_path: str = "../../data",
                                     max_files: int = 50,
                                     sequence_length: int = 10) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str]]:
    """
    Prepare data for CLIP-style gesture recognition using PyTorch only
    """
    logger = get_logger()
    logger.debug("=== ENTERING prepare_clip_gesture_data_pytorch ===")
    logger.debug(f"Parameters: folder_path={folder_path}, max_files={max_files}, sequence_length={sequence_length}")
    
    import time
    start_time = time.time()
    logger.info("Loading IRDS data for CLIP model...")
    logger.debug(f"Loading from folder: {folder_path}, max_files: {max_files}")
    
    df = load_irds_data(folder_path=folder_path, max_files=max_files)
    logger.info(f"Initial data loaded: {len(df)} rows")
    
    # Load gesture labels
    gesture_labels = load_gesture_labels()
    logger.info(f"Loaded {len(gesture_labels)} gesture labels")
    
    # Filter out rows with missing gesture labels
    logger.debug("Filtering data...")
    df = df.dropna(subset=['gesture_label'])
    df = df[df['gesture_label'].notna()]
    
    logger.info(f"After filtering: {len(df)} rows from {max_files} files")
    
    # Get numeric columns (skeleton data)
    logger.info("Extracting skeleton columns...")
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Remove metadata columns
    metadata_cols = ['subject_id', 'date_id', 'gesture_label', 'rep_number', 
                    'correct_label', 'position']
    skeleton_cols = [col for col in numeric_cols if col not in metadata_cols]
    
    logger.info(f"Using {len(skeleton_cols)} skeleton columns")
    
    # Convert to PyTorch tensors immediately
    logger.info("Converting to PyTorch tensors...")
    skeleton_data = torch.FloatTensor(df[skeleton_cols].values)
    logger.info(f"Skeleton data shape: {skeleton_data.shape}")
    
    # Reshape skeleton data to (samples, joints, coords)
    num_joints = 25
    coords_per_joint = 3
    skeleton_data = skeleton_data.view(-1, num_joints, coords_per_joint)
    logger.info(f"Reshaped skeleton data: {skeleton_data.shape}")
    
    # Get gesture labels
    logger.info("Processing gesture labels...")
    gesture_labels_numeric = torch.LongTensor(df['gesture_label'].astype(int).values)
    logger.info(f"Gesture labels shape: {gesture_labels_numeric.shape}")
    
    # Create sequences using PyTorch operations
    logger.info(f"Creating sequences with length {sequence_length}...")
    sequences = []
    sequence_labels = []
    
    total_sequences = len(skeleton_data) - sequence_length + 1
    logger.info(f"Will create {total_sequences} sequences")
    
    for i in range(total_sequences):
        if i % 1000 == 0:  # Progress indicator
            logger.info(f"Processing sequence {i}/{total_sequences}")
        sequence = skeleton_data[i:i + sequence_length]
        label = gesture_labels_numeric[i + sequence_length - 1]
        sequences.append(sequence)
        sequence_labels.append(label)
    
    logger.info("Stacking sequences...")
    X = torch.stack(sequences)
    y = torch.stack(sequence_labels)
    logger.info(f"Final X shape: {X.shape}, y shape: {y.shape}")
    
    # Get unique gesture names and descriptions
    unique_labels = sorted(df['gesture_label'].unique())
    gesture_names = [gesture_labels.get(str(label), f"Gesture {label}") for label in unique_labels]
    
    # Create descriptions for each gesture
    gesture_descriptions = [
        "Elbow flexion and extension movement of the left arm",
        "Elbow flexion and extension movement of the right arm", 
        "Shoulder abduction and adduction of the left arm",
        "Shoulder abduction and adduction of the right arm",
        "Shoulder flexion and extension of the left arm",
        "Shoulder flexion and extension of the right arm",
        "Hip flexion and extension of the left leg",
        "Hip flexion and extension of the right leg",
        "Knee flexion and extension of the left leg"
    ]
    
    # Ensure we have descriptions for all gestures
    while len(gesture_descriptions) < len(gesture_names):
        gesture_descriptions.append(f"Gesture movement pattern {len(gesture_descriptions) + 1}")
    
    gesture_descriptions = gesture_descriptions[:len(gesture_names)]
    
    logger.info(f"Created {len(X)} sequences")
    logger.info(f"Gesture classes: {gesture_names}")
    logger.info(f"X tensor shape: {X.shape}")
    logger.info(f"y tensor shape: {y.shape}")
    
    elapsed_time = time.time() - start_time
    logger.info(f"Data preparation completed in {elapsed_time:.2f} seconds")
    
    return X, y, gesture_names, gesture_descriptions

def train_clip_gesture_model_pytorch(X: torch.Tensor, y: torch.Tensor, gesture_names: List[str],
                                   gesture_descriptions: List[str], sequence_length: int = 10,
                                   batch_size: int = 32, num_epochs: int = 50,
                                   learning_rate: float = 0.001, test_size: float = 0.2,
                                   device: str = 'cpu'):
    """
    Train the CLIP-style gesture recognition model using PyTorch only
    """
    logger = get_logger()
    logger.info("=== ENTERING train_clip_gesture_model_pytorch ===")
    logger.info(f"Parameters: X.shape={X.shape}, y.shape={y.shape}, device={device}")
    logger.info(f"Training params: batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
    
    logger.info("Preparing CLIP gesture data...")
    
    # Convert to numpy for sklearn split (minimal numpy usage)
    logger.info("Converting to numpy...")
    X_np = X.numpy()
    y_np = y.numpy()
    logger.info("Numpy conversion completed")
    
    # Split data
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_np, y_np, test_size=test_size, random_state=42, stratify=y_np
    )
    logger.info("Data split completed")
    
    # Convert back to PyTorch tensors
    logger.info("Converting back to PyTorch tensors...")
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)
    logger.info("PyTorch tensor conversion completed")
    
    # Scale data using PyTorch
    logger.info("Scaling data...")
    X_train_flat = X_train.view(X_train.size(0), -1)
    X_test_flat = X_test.view(X_test.size(0), -1)
    
    # Calculate mean and std
    mean = X_train_flat.mean(dim=0)
    std = X_train_flat.std(dim=0)
    
    # Normalize
    X_train_flat = (X_train_flat - mean) / (std + 1e-8)
    X_test_flat = (X_test_flat - mean) / (std + 1e-8)
    
    # Reshape back
    X_train = X_train_flat.view(X_train.shape)
    X_test = X_test_flat.view(X_test.shape)
    logger.info("Data scaling completed")
    
    logger.info(f"Training data: {X_train.shape}, {y_train.shape}")
    logger.info(f"Test data: {X_test.shape}, {y_test.shape}")
    
    # Move to device - force GPU usage
    logger.info(f"Moving data to device: {device}")
    
    # Test GPU with a small tensor first
    # print("Testing GPU with small tensor...")
    # test_tensor = torch.randn(10, 10).to(device)
    # print(f"✓ GPU test passed: {test_tensor.device}")
    
    # Now move the actual data
    if device.startswith('cuda'):
        logger.info("Moving training data to GPU...")
        logger.info(f"X_train dtype: {X_train.dtype}")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_train device: {X_train.device}")
        logger.info(f"X_train is contiguous: {X_train.is_contiguous()}")
        logger.info(f"X_train memory usage: {X_train.element_size() * X_train.nelement() / 1e9:.2f} GB")
        X_train = X_train.to(device)
        logger.info("+ X_train moved to GPU")
        
        X_test = X_test.to(device)
        logger.info("+ X_test moved to GPU")
        
        y_train = y_train.to(device)
        logger.info("+ y_train moved to GPU")
    else:
        logger.info("Using CPU - no device transfer needed")
        logger.info(f"X_train dtype: {X_train.dtype}")
        logger.info(f"X_train shape: {X_train.shape}")
        logger.info(f"X_train device: {X_train.device}")
        logger.info(f"X_train memory usage: {X_train.element_size() * X_train.nelement() / 1e9:.2f} GB")
    
    if device.startswith('cuda'):
        y_test = y_test.to(device)
        logger.info("+ y_test moved to GPU")
    else:
        logger.info("+ y_test using CPU")
    
    logger.info(f"Data moved to device: {device}")
    logger.info(f"X_train device: {X_train.device}")
    logger.info(f"y_train device: {y_train.device}")
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("Data loaders created")
    
    # Create model (placeholder - you'll need to implement the actual CLIP model)
    logger.info("Creating model...")
    class SimpleCLIPModel(nn.Module):
        def __init__(self, input_size, num_classes):
            super().__init__()
            self.encoder = nn.LSTM(input_size, 256, 2, batch_first=True)
            self.classifier = nn.Linear(256, num_classes)
        
        def forward(self, x):
            # x: (batch, sequence, features)
            x = x.view(x.size(0), x.size(1), -1)  # Flatten joints and coords
            lstm_out, _ = self.encoder(x)
            # Use last output
            output = self.classifier(lstm_out[:, -1, :])
            return output
    
    model = SimpleCLIPModel(75, len(gesture_names)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    logger.info(f"Model created and moved to {device}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Test model with a small batch
    test_input = torch.randn(2, 10, 75).to(device)
    test_output = model(test_input)
    logger.info(f"Model test output device: {test_output.device}")
    logger.info(f"Model test output shape: {test_output.shape}")
    
    # Check GPU memory if using GPU
    if device.startswith('cuda'):
        logger.info(f"GPU Memory Allocated: {torch.cuda.memory_allocated(device) / 1024**3:.2f} GB")
        logger.info(f"GPU Memory Cached: {torch.cuda.memory_reserved(device) / 1024**3:.2f} GB")
    
    # Training loop
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Debug: Check device for first batch
            if batch_idx == 0:
                logger.info(f"Batch X device: {batch_X.device}")
                logger.info(f"Batch y device: {batch_y.device}")
            
            # Forward pass
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    
    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_y.size(0)
            test_correct += (predicted == batch_y).sum().item()
    
    test_accuracy = 100 * test_correct / test_total
    logger.info(f"Final test accuracy: {test_accuracy:.2f}%")
    
    # Save model
    torch.save(model.state_dict(), 'outputs/clip_gesture_model_pytorch.pth')
    
    # Save scaler info (mean and std)
    scaler_info = {
        'mean': mean.tolist(),
        'std': std.tolist()
    }
    with open('outputs/clip_gesture_scaler_pytorch.json', 'w') as f:
        json.dump(scaler_info, f)
    
    # Save gesture info
    gesture_info = {
        'gesture_names': gesture_names,
        'gesture_descriptions': gesture_descriptions,
        'sequence_length': sequence_length
    }
    with open('outputs/clip_gesture_info_pytorch.json', 'w') as f:
        json.dump(gesture_info, f)
    
    logger.info("Model saved successfully!")
    
    return model, scaler_info, gesture_names, gesture_descriptions

def check_rocm_availability():
    """Check if ROCm is available and working"""
    try:
        if torch.cuda.is_available():
            # Try to create a tensor on GPU to verify it works
            test_tensor = torch.tensor([1.0]).cuda()
            return True
        return False
    except Exception as e:
        logger.info(f"ROCm test failed: {e}")
        return False

if __name__ == "__main__":
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Prepare data with PyTorch only
    X, y, gesture_names, gesture_descriptions = prepare_clip_gesture_data_pytorch(
        folder_path="data",
        max_files=100,
        sequence_length=15
    )
    
    # Train model
    model, scaler, gesture_names, gesture_descriptions = train_clip_gesture_model_pytorch(
        X, y, gesture_names, gesture_descriptions,
        sequence_length=15,
        batch_size=16,
        num_epochs=100,
        learning_rate=0.0005,
        device=device
    )
    
    logger.info("PyTorch-only training completed!")
