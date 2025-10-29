#!/usr/bin/env python3
"""
CLIP-style Gesture Recognition Model - PyTorch Only Version
Replaces NumPy operations with PyTorch to avoid GPU conflicts
"""

import os, sys, glob, json, time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional, Dict

path_to_irds = Path(__file__).parent.parent
sys.path.insert(0, str(path_to_irds))

# Add project root to path for imports
from tensor_utils import tensor_train_test_split, tensor_scaler_fit_transform
from utils import get_data_path, get_logger
from utils.benchmark import GPUBenchmark
from model.model_architectures import get_model, count_parameters, MODEL_INFO

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
    skipped_files = 0
    
    for file_path in all_files:
        try:
            df = pd.read_csv(file_path, header=header)

            # Assign column names if provided and no header present
            if not has_header and columns is not None:
                if len(columns) != df.shape[1]:
                    logger.info(f"Skipping {file_path}: column mismatch")
                    skipped_files += 1
                    continue
                df.columns = columns
        except Exception as e:
            logger.info(f"Skipping {file_path}: {str(e)[:100]}")
            skipped_files += 1
            continue

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

    if skipped_files > 0:
        logger.info(f"Skipped {skipped_files} malformed files out of {len(all_files)} total files")
    
    logger.info(f"Successfully loaded {len(list_of_dfs)} files")
    combined_df = pd.concat(list_of_dfs, ignore_index=True)
    return combined_df

def prepare_clip_gesture_data_pytorch(folder_path: str = "../../data",
                                     max_files: Optional[int] = None,
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
    
    # Validate skeleton columns
    num_joints = 25
    coords_per_joint = 3
    expected_cols = num_joints * coords_per_joint
    
    # Convert to PyTorch tensors immediately
    logger.info("Converting to PyTorch tensors...")
    skeleton_data = torch.FloatTensor(df[skeleton_cols].values)
    logger.info(f"Skeleton data shape: {skeleton_data.shape}")
    
    # Handle incorrect column count
    if skeleton_data.shape[1] != expected_cols:
        logger.info(f"Warning: Expected {expected_cols} skeleton columns but found {skeleton_data.shape[1]}")
        if skeleton_data.shape[1] < expected_cols:
            # Pad with zeros
            padding = torch.zeros(skeleton_data.shape[0], expected_cols - skeleton_data.shape[1])
            skeleton_data = torch.cat([skeleton_data, padding], dim=1)
            logger.info(f"Padded to {skeleton_data.shape}")
        else:
            # Trim to expected size
            skeleton_data = skeleton_data[:, :expected_cols]
            logger.info(f"Trimmed to {skeleton_data.shape}")
    
    # Reshape skeleton data to (samples, joints, coords)
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
                                   device: str = 'cpu', model_architecture: str = 'medium',
                                   benchmark: GPUBenchmark = None):
    """
    Train the CLIP-style gesture recognition model using PyTorch only
    """
    logger = get_logger()
    logger.info("=== ENTERING train_clip_gesture_model_pytorch ===")
    logger.info(f"Parameters: X.shape={X.shape}, y.shape={y.shape}, device={device}")
    logger.info(f"Training params: batch_size={batch_size}, epochs={num_epochs}, lr={learning_rate}")
    
    # Start timing for data preparation
    if benchmark:
        benchmark.start_timer('data_preparation')
    
    logger.info("Preparing CLIP gesture data...")
    
    # Split data using tensor-based function
    logger.info("Splitting data...")
    X_train, X_test, y_train, y_test = tensor_train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    logger.info("Data split completed")
    
    # Scale data using tensor utils
    logger.info("Scaling data...")
    X_train_flat = X_train.view(X_train.size(0), -1)
    X_test_flat = X_test.view(X_test.size(0), -1)
    
    X_train_scaled, X_test_scaled, scaler_params = tensor_scaler_fit_transform(X_train_flat, X_test_flat)
    
    # Reshape back
    X_train = X_train_scaled.view(X_train.shape)
    X_test = X_test_scaled.view(X_test.shape)
    logger.info("Data scaling completed")
    
    logger.info(f"Training data: {X_train.shape}, {y_train.shape}")
    logger.info(f"Test data: {X_test.shape}, {y_test.shape}")
    
    # Record data metrics
    if benchmark:
        benchmark.record_data_metrics(
            train_samples=len(X_train),
            test_samples=len(X_test),
            num_classes=len(gesture_names),
            sequence_length=sequence_length
        )
        benchmark.stop_timer('data_preparation')
        benchmark.start_timer('gpu_transfer')
    
    # Move to device - force GPU usage
    logger.debug(f"Moving data to device: {device}")
    logger.debug(f"Initial tensor device: {X_train.device}")
    
    # Move data to device directly
    if device.startswith('cuda'):
        logger.debug("Moving data to GPU...")
        X_train = X_train.to(device)
        X_test = X_test.to(device)
        y_train = y_train.to(device)
        y_test = y_test.to(device)
        logger.debug("✓ All data moved to GPU")
    else:
        logger.debug("Using CPU - no device transfer needed")
    
    logger.info(f"Data moved to device: {device}")
    logger.debug(f"X_train device: {X_train.device}")
    logger.debug(f"y_train device: {y_train.device}")
    
    if benchmark:
        benchmark.stop_timer('gpu_transfer')
        benchmark.record_memory('after_data_transfer', device)
        benchmark.start_timer('model_creation')
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    logger.info("Data loaders created")
    
    # Create model using model architectures
    logger.info(f"Creating {model_architecture} model...")
    
    # Print model info
    if model_architecture in MODEL_INFO:
        info = MODEL_INFO[model_architecture]
        logger.info(f"Model: {model_architecture}")
        logger.info(f"Expected parameters: {info['params']}")
        logger.info(f"Description: {info['description']}")
    
    model = get_model(model_architecture, 75, len(gesture_names), device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Count and log actual parameters
    param_count = count_parameters(model)
    logger.info(f"Model created and moved to {device}")
    logger.info(f"Model device: {next(model.parameters()).device}")
    logger.info(f"Actual model parameters: {param_count:,}")
    
    if benchmark:
        benchmark.record_model_metrics(model_architecture, param_count)
        benchmark.stop_timer('model_creation')
        benchmark.record_memory('after_model_creation', device)
    
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
    
    if benchmark:
        benchmark.start_timer('total_training')
    
    epoch_start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_iter_start = time.time()
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
            optimizer.zero_grad()
            
            # Debug: Check device for first batch
            if batch_idx == 0:
                logger.debug(f"Batch X device: {batch_X.device}")
                logger.debug(f"Batch y device: {batch_y.device}")
            
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
        
        # Record epoch metrics
        if benchmark:
            benchmark.record_training_metrics(epoch, avg_loss, accuracy, learning_rate)
        
        epoch_time = time.time() - epoch_iter_start
        
        if epoch % 2 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f}s")
    
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
    
    if benchmark:
        benchmark.stop_timer('total_training')
        benchmark.record_test_metrics(test_accuracy)
        benchmark.record_memory('after_training', device)
        
        # Calculate throughput
        total_train_time = time.time() - epoch_start_time
        total_samples = len(X_train) * num_epochs
        samples_per_sec = total_samples / total_train_time
        benchmark.record_throughput(samples_per_sec, 'training')
    
    # Save complete checkpoint for inference
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'gesture_names': gesture_names,
        'gesture_descriptions': gesture_descriptions,
        'sequence_length': sequence_length,
        'scaler_params': scaler_params,
        'architecture': model_architecture,
        'test_accuracy': test_accuracy
    }
    torch.save(checkpoint, 'outputs/clip_gesture_model_pytorch.pth')
    
    # Also save legacy separate files for backward compatibility
    scaler_info = {
        'mean': scaler_params['mean'].tolist(),
        'std': scaler_params['std'].tolist()
    }
    with open('outputs/clip_gesture_scaler_pytorch.json', 'w') as f:
        json.dump(scaler_info, f)
    
    gesture_info = {
        'gesture_names': gesture_names,
        'gesture_descriptions': gesture_descriptions,
        'sequence_length': sequence_length
    }
    with open('outputs/clip_gesture_info_pytorch.json', 'w') as f:
        json.dump(gesture_info, f)
    
    logger.info("Model saved successfully!")
    
    return model, scaler_params, gesture_names, gesture_descriptions

def check_rocm_availability():
    """Check if ROCm is available and working"""
    logger = get_logger()
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
    logger = get_logger()
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    # Prepare data with PyTorch only - train on all files
    X, y, gesture_names, gesture_descriptions = prepare_clip_gesture_data_pytorch(
        folder_path="data",
        max_files=None,  # Use all available files
        sequence_length=15
    )
    
    # Train model
    model, scaler_params, gesture_names, gesture_descriptions = train_clip_gesture_model_pytorch(
        X, y, gesture_names, gesture_descriptions,
        sequence_length=15,
        batch_size=16,
        num_epochs=100,
        learning_rate=0.0005,
        device=device
    )
    
    logger.info("PyTorch-only training completed!")
