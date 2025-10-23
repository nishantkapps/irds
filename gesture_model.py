"""
Gesture Recognition Model for IRDS Dataset
Similar to CLIP approach but for skeleton-based gesture recognition
"""

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import joblib

# Import our existing data loading functions
import sys
sys.path.append('/home/nishant/project/irds')

# Copy the necessary functions directly to avoid import issues
def load_gesture_labels(labels_path: str = "/home/nishant/project/irds/labels.csv") -> dict:
    """Load gesture labels from CSV file."""
    try:
        labels_df = pd.read_csv(labels_path)
        return dict(zip(labels_df['GestureIndex'].astype(str), labels_df['GestureName']))
    except FileNotFoundError:
        print(f"Warning: Labels file not found at {labels_path}")
        return {}
    except Exception as e:
        print(f"Warning: Could not load labels: {e}")
        return {}

def load_irds_data(folder_path: str = "/home/nishant/project/irds/data",
                   file_pattern: str = "*.txt",
                   has_header: bool = False,
                   add_metadata: bool = True,
                   columns: Optional[List[str]] = None,
                   include_source_file: bool = True,
                   max_files: Optional[int] = None) -> pd.DataFrame:
    """Load IRDS dataset files into a combined pandas DataFrame."""
    search_path = os.path.join(folder_path, file_pattern)
    all_files = glob.glob(search_path)
    if len(all_files) == 0:
        raise FileNotFoundError(f"No files found at {search_path}")
    
    # Limit number of files for faster loading
    if max_files is not None and len(all_files) > max_files:
        print(f"Loading only first {max_files} files out of {len(all_files)} for faster startup")
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


class GestureDataset(Dataset):
    """PyTorch Dataset for gesture recognition"""
    
    def __init__(self, data: np.ndarray, labels: np.ndarray, sequence_length: int = 10):
        """
        Args:
            data: Skeleton data of shape (samples, joints, coordinates)
            labels: Gesture labels
            sequence_length: Number of frames to use for each sample
        """
        self.data = data
        self.labels = labels
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence of frames
        sequence = self.data[idx:idx + self.sequence_length]
        label = self.labels[idx + self.sequence_length - 1]  # Use last frame's label
        
        return torch.FloatTensor(sequence), torch.LongTensor([label])


class GestureRecognitionModel(nn.Module):
    """Neural network for gesture recognition from skeleton data"""
    
    def __init__(self, num_joints: int = 25, num_coords: int = 3, 
                 num_classes: int = 9, sequence_length: int = 10,
                 hidden_dim: int = 128):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.num_classes = num_classes
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # Input: (batch, sequence, joints, coords) -> (batch, sequence, joints*coords)
        self.input_dim = num_joints * num_coords
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=75,  # 25 joints * 3 coordinates = 75
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (batch, sequence, joints, coords)
        if len(x.shape) == 4:
            batch_size, seq_len, joints, coords = x.shape
            # Reshape to (batch, sequence, joints*coords)
            x = x.view(batch_size, seq_len, joints * coords)
        else:
            # Handle case where input is already flattened
            batch_size, seq_len = x.shape[:2]
            # Ensure we have 75 features per timestep
            if x.shape[-1] != 75:
                x = x.view(batch_size, seq_len, -1)
                # Take first 75 features if more than 75
                if x.shape[-1] > 75:
                    x = x[:, :, :75]
                # Pad if less than 75
                elif x.shape[-1] < 75:
                    padding = torch.zeros(batch_size, seq_len, 75 - x.shape[-1])
                    x = torch.cat([x, padding], dim=-1)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output


class GestureTrainer:
    """Trainer class for gesture recognition model"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def train_epoch(self, dataloader, optimizer, criterion):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, labels) in enumerate(dataloader):
            data, labels = data.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(data)
            loss = criterion(outputs, labels.squeeze())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()
            
        return total_loss / len(dataloader), 100. * correct / total
    
    def evaluate(self, dataloader, criterion):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for data, labels in dataloader:
                data, labels = data.to(self.device), labels.to(self.device)
                outputs = self.model(data)
                loss = criterion(outputs, labels.squeeze())
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())
        
        return total_loss / len(dataloader), 100. * correct / total, all_predictions, all_labels


def prepare_gesture_data(folder_path: str = "/home/nishant/project/irds/data",
                        max_files: int = 50,
                        sequence_length: int = 10,
                        test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for gesture recognition training
    
    Returns:
        X: Skeleton sequences (samples, sequence_length, joints, coords)
        y: Gesture labels
        gesture_names: List of gesture names
    """
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
    
    # Reshape skeleton data to (samples, joints, coords)
    # Assuming 25 joints with 3 coordinates each
    num_joints = 25
    coords_per_joint = 3
    
    skeleton_data = df[skeleton_cols].values
    skeleton_data = skeleton_data.reshape(-1, num_joints, coords_per_joint)
    
    # Get gesture labels
    gesture_labels_numeric = df['gesture_label'].astype(int).values
    
    # Create sequences
    sequences = []
    sequence_labels = []
    
    for i in range(len(skeleton_data) - sequence_length + 1):
        sequence = skeleton_data[i:i + sequence_length]
        label = gesture_labels_numeric[i + sequence_length - 1]
        sequences.append(sequence)
        sequence_labels.append(label)
    
    X = np.array(sequences)
    y = np.array(sequence_labels)
    
    # Get unique gesture names
    unique_labels = sorted(df['gesture_label'].unique())
    gesture_names = [gesture_labels.get(str(label), f"Gesture {label}") for label in unique_labels]
    
    print(f"Created {len(X)} sequences")
    print(f"Gesture classes: {gesture_names}")
    
    return X, y, gesture_names


def train_gesture_model(X: np.ndarray, y: np.ndarray, gesture_names: List[str],
                       sequence_length: int = 10, batch_size: int = 32,
                       num_epochs: int = 50, learning_rate: float = 0.001,
                       test_size: float = 0.2, device: str = 'cpu'):
    """
    Train the gesture recognition model
    """
    print("Preparing data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = GestureDataset(X_train_scaled, y_train, sequence_length)
    test_dataset = GestureDataset(X_test_scaled, y_test, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_classes = len(gesture_names)
    model = GestureRecognitionModel(
        num_joints=25,
        num_coords=3,
        num_classes=num_classes,
        sequence_length=sequence_length
    )
    
    # Training setup
    trainer = GestureTrainer(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    print(f"Training on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion)
        
        # Evaluate
        test_loss, test_acc, predictions, labels = trainer.evaluate(test_loader, criterion)
        
        # Update learning rate
        scheduler.step(test_loss)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")
    
    # Final evaluation
    print("\nFinal Results:")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=gesture_names))
    
    # Confusion matrix
    cm = confusion_matrix(labels, predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=gesture_names, yticklabels=gesture_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
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
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    # Save model and scaler
    torch.save(model.state_dict(), 'gesture_model.pth')
    joblib.dump(scaler, 'gesture_scaler.pkl')
    
    print(f"\nModel saved as 'gesture_model.pth'")
    print(f"Scaler saved as 'gesture_scaler.pkl'")
    
    return model, scaler, gesture_names


def predict_gesture(model, scaler, sequence_data: np.ndarray, gesture_names: List[str]) -> str:
    """
    Predict gesture from a sequence of skeleton data
    """
    model.eval()
    
    # Normalize data
    sequence_scaled = scaler.transform(sequence_data.reshape(-1, sequence_data.shape[-1])).reshape(sequence_data.shape)
    
    # Add batch dimension
    sequence_tensor = torch.FloatTensor(sequence_scaled).unsqueeze(0)
    
    with torch.no_grad():
        outputs = model(sequence_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, predicted_class].item()
    
    predicted_gesture = gesture_names[predicted_class]
    
    return predicted_gesture, confidence


if __name__ == "__main__":
    # Set device with better GPU detection
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("CUDA not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Prepare data with more files for better training
    X, y, gesture_names = prepare_gesture_data(
        folder_path="/home/nishant/project/irds/data",
        max_files=100,  # Increased for better training
        sequence_length=10
    )
    
    # Train model with GPU-optimized parameters
    model, scaler, gesture_names = train_gesture_model(
        X, y, gesture_names,
        sequence_length=10,
        batch_size=32 if device == 'cuda' else 16,  # Larger batch size for GPU
        num_epochs=50,  # More epochs for better training
        learning_rate=0.001,
        device=device
    )
    
    print("Training completed!")
    print(f"Model saved as 'gesture_model.pth'")
    print(f"Scaler saved as 'gesture_scaler.pkl'")
    print(f"Charts saved as 'confusion_matrix.png' and 'training_curves.png'")

