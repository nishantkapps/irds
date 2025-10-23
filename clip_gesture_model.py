"""
CLIP-style Gesture Recognition Model for IRDS Dataset
Combines skeleton data with text descriptions for gesture recognition
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
import json

# Import our existing data loading functions
import sys
import os
# Get the current directory where the script is located
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# Copy the necessary functions directly to avoid import issues
def load_gesture_labels(labels_path: str = "labels.csv") -> dict:
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

def load_irds_data(folder_path: str = "data",
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


class CLIPGestureDataset(Dataset):
    """PyTorch Dataset for CLIP-style gesture recognition"""
    
    def __init__(self, skeleton_data: np.ndarray, gesture_labels: np.ndarray, 
                 gesture_descriptions: List[str], sequence_length: int = 10):
        """
        Args:
            skeleton_data: Skeleton data of shape (samples, joints, coordinates)
            gesture_labels: Gesture labels
            gesture_descriptions: Text descriptions for each gesture
            sequence_length: Number of frames to use for each sample
        """
        self.skeleton_data = skeleton_data
        self.gesture_labels = gesture_labels
        self.gesture_descriptions = gesture_descriptions
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.skeleton_data) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Get sequence of frames
        sequence = self.skeleton_data[idx:idx + self.sequence_length]
        label = self.gesture_labels[idx + self.sequence_length - 1]
        description = self.gesture_descriptions[label]
        
        return torch.FloatTensor(sequence), torch.LongTensor([label]), description


class SkeletonEncoder(nn.Module):
    """Encoder for skeleton data (similar to CLIP's image encoder)"""
    
    def __init__(self, num_joints: int = 25, num_coords: int = 3, 
                 sequence_length: int = 10, embed_dim: int = 512):
        super().__init__()
        
        self.num_joints = num_joints
        self.num_coords = num_coords
        self.sequence_length = sequence_length
        self.embed_dim = embed_dim
        
        # Input: (batch, sequence, joints, coords) -> (batch, sequence, joints*coords)
        self.input_dim = num_joints * num_coords
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=75,  # 25 joints * 3 coordinates = 75
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=256,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Projection to embedding dimension
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
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
        lstm_out, _ = self.lstm(x)
        
        # Apply attention
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        
        # Global average pooling over sequence
        pooled = torch.mean(attn_out, dim=1)
        
        # Project to embedding dimension
        embedding = self.projection(pooled)
        
        return embedding


class TextEncoder(nn.Module):
    """Encoder for text descriptions (similar to CLIP's text encoder)"""
    
    def __init__(self, vocab_size: int = 10000, embed_dim: int = 512, 
                 max_length: int = 128):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Simple embedding + LSTM for text
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Projection layer
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, text_tokens):
        # text_tokens: (batch, sequence_length)
        embedded = self.embedding(text_tokens)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # Use the last hidden state
        text_embedding = self.projection(hidden[-1])
        
        return text_embedding


class CLIPGestureModel(nn.Module):
    """CLIP-style model for gesture recognition"""
    
    def __init__(self, num_joints: int = 25, num_coords: int = 3,
                 sequence_length: int = 10, embed_dim: int = 512,
                 vocab_size: int = 10000, num_classes: int = 9):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        
        # Encoders
        self.skeleton_encoder = SkeletonEncoder(
            num_joints=num_joints,
            num_coords=num_coords,
            sequence_length=sequence_length,
            embed_dim=embed_dim
        )
        
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim
        )
        
        # Temperature parameter for contrastive learning
        self.temperature = nn.Parameter(torch.ones([]) * 0.07)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embed_dim // 2, num_classes)
        )
        
    def forward(self, skeleton_data, text_tokens=None):
        # Encode skeleton data
        # Ensure skeleton_data has the right shape: (batch, sequence, joints, coords)
        if len(skeleton_data.shape) == 3:
            # Add joint dimension if missing
            batch_size, seq_len, features = skeleton_data.shape
            joints = 25
            coords = features // joints
            skeleton_data = skeleton_data.view(batch_size, seq_len, joints, coords)
        
        skeleton_embedding = self.skeleton_encoder(skeleton_data)
        
        if text_tokens is not None:
            # Encode text
            text_embedding = self.text_encoder(text_tokens)
            
            # Normalize embeddings
            skeleton_embedding = torch.nn.functional.normalize(skeleton_embedding, dim=1)
            text_embedding = torch.nn.functional.normalize(text_embedding, dim=1)
            
            # Compute similarity matrix
            logits = torch.matmul(skeleton_embedding, text_embedding.T) * self.temperature
            
            return logits, skeleton_embedding, text_embedding
        
        # For classification
        output = self.classifier(skeleton_embedding)
        return output


class CLIPGestureTrainer:
    """Trainer class for CLIP-style gesture recognition"""
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.model.to(device)
        
    def contrastive_loss(self, logits, labels):
        """Contrastive loss for CLIP training"""
        # logits: (batch_size, batch_size)
        batch_size = logits.shape[0]
        
        # Create labels for contrastive learning
        labels = torch.arange(batch_size).to(self.device)
        
        # Symmetric loss
        loss_skeleton_to_text = nn.CrossEntropyLoss()(logits, labels)
        loss_text_to_skeleton = nn.CrossEntropyLoss()(logits.T, labels)
        
        return (loss_skeleton_to_text + loss_text_to_skeleton) / 2
    
    def train_epoch(self, dataloader, optimizer, criterion, use_contrastive=True):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (skeleton_data, labels, descriptions) in enumerate(dataloader):
            skeleton_data, labels = skeleton_data.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            
            if use_contrastive and batch_idx % 2 == 0:  # Use contrastive loss every other batch
                # Create text tokens (simplified - in practice you'd tokenize descriptions)
                text_tokens = torch.randint(0, 1000, (skeleton_data.shape[0], 10)).to(self.device)
                
                logits, _, _ = self.model(skeleton_data, text_tokens)
                loss = self.contrastive_loss(logits, labels)
            else:
                # Use classification loss
                outputs = self.model(skeleton_data)
                loss = criterion(outputs, labels.squeeze())
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if not use_contrastive or batch_idx % 2 == 1:
                with torch.no_grad():
                    outputs = self.model(skeleton_data)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels.squeeze()).sum().item()
            
        return total_loss / len(dataloader), 100. * correct / total if total > 0 else 0
    
    def evaluate(self, dataloader, criterion):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for skeleton_data, labels, descriptions in dataloader:
                skeleton_data, labels = skeleton_data.to(self.device), labels.to(self.device)
                outputs = self.model(skeleton_data)
                loss = criterion(outputs, labels.squeeze())
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.squeeze()).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.squeeze().cpu().numpy())
        
        return total_loss / len(dataloader), 100. * correct / total, all_predictions, all_labels


def prepare_clip_gesture_data(folder_path: str = "/home/nishant/project/irds/data",
                             max_files: int = 50,
                             sequence_length: int = 10) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """
    Prepare data for CLIP-style gesture recognition
    
    Returns:
        X: Skeleton sequences (samples, sequence_length, joints, coords)
        y: Gesture labels
        gesture_names: List of gesture names
        gesture_descriptions: List of gesture descriptions
    """
    print("Loading IRDS data for CLIP model...")
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
    
    print(f"Created {len(X)} sequences")
    print(f"Gesture classes: {gesture_names}")
    print(f"Gesture descriptions: {gesture_descriptions}")
    
    return X, y, gesture_names, gesture_descriptions


def train_clip_gesture_model(X: np.ndarray, y: np.ndarray, gesture_names: List[str],
                           gesture_descriptions: List[str], sequence_length: int = 10,
                           batch_size: int = 32, num_epochs: int = 50,
                           learning_rate: float = 0.001, test_size: float = 0.2,
                           device: str = 'cpu'):
    """
    Train the CLIP-style gesture recognition model
    """
    print("Preparing CLIP gesture data...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )
    
    # Normalize data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
    
    # Create datasets
    train_dataset = CLIPGestureDataset(X_train_scaled, y_train, gesture_descriptions, sequence_length)
    test_dataset = CLIPGestureDataset(X_test_scaled, y_test, gesture_descriptions, sequence_length)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model
    num_classes = len(gesture_names)
    model = CLIPGestureModel(
        num_joints=25,
        num_coords=3,
        sequence_length=sequence_length,
        embed_dim=512,
        vocab_size=10000,
        num_classes=num_classes
    )
    
    # Training setup
    trainer = CLIPGestureTrainer(model, device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    print(f"Training CLIP model on {device}...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = trainer.train_epoch(train_loader, optimizer, criterion, use_contrastive=True)
        
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
    plt.title('CLIP Gesture Model - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('clip_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    # Plot training curves
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('CLIP Model - Training and Test Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('CLIP Model - Training and Test Accuracy')
    
    plt.tight_layout()
    plt.savefig('clip_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close instead of show
    
    # Save model and scaler
    torch.save(model.state_dict(), 'clip_gesture_model.pth')
    joblib.dump(scaler, 'clip_gesture_scaler.pkl')
    
    # Save gesture info
    gesture_info = {
        'names': gesture_names,
        'descriptions': gesture_descriptions
    }
    with open('clip_gesture_info.json', 'w') as f:
        json.dump(gesture_info, f)
    
    print(f"\nCLIP model saved as 'clip_gesture_model.pth'")
    print(f"Scaler saved as 'clip_gesture_scaler.pkl'")
    print(f"Gesture info saved as 'clip_gesture_info.json'")
    
    return model, scaler, gesture_names, gesture_descriptions


def predict_clip_gesture(model, scaler, sequence_data: np.ndarray, 
                        gesture_names: List[str], gesture_descriptions: List[str]) -> str:
    """
    Predict gesture from a sequence of skeleton data using CLIP model
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
    predicted_description = gesture_descriptions[predicted_class]
    
    return predicted_gesture, predicted_description, confidence


def check_rocm_availability():
    """Check if ROCm is available and working"""
    try:
        # Check if PyTorch can access GPU
        if torch.cuda.is_available():
            # Try to create a tensor on GPU to verify it works
            test_tensor = torch.tensor([1.0]).cuda()
            return True
        return False
    except Exception as e:
        print(f"ROCm test failed: {e}")
        return False

if __name__ == "__main__":
    # Set device with proper ROCm/AMD GPU detection
    if check_rocm_availability():
        device = 'cuda'
        print(f"Using AMD GPU with ROCm: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        device = 'cpu'
        print("ROCm not available, using CPU")
    
    print(f"Using device: {device}")
    
    # Prepare data with more files for better training
    X, y, gesture_names, gesture_descriptions = prepare_clip_gesture_data(
        folder_path="data",  # Use relative path
        max_files=200,  # More data for better accuracy
        sequence_length=15  # Longer sequences for better context
    )
    
    # Train CLIP model with optimized parameters for higher accuracy
    model, scaler, gesture_names, gesture_descriptions = train_clip_gesture_model(
        X, y, gesture_names, gesture_descriptions,
        sequence_length=15,  # Match the data preparation
        batch_size=16 if device == 'cuda' else 8,  # Smaller batch size for better learning
        num_epochs=100,  # More epochs for better convergence
        learning_rate=0.0005,  # Lower learning rate for better stability
        device=device
    )
    
    print("CLIP gesture model training completed!")
    print(f"CLIP model saved as 'clip_gesture_model.pth'")
    print(f"Scaler saved as 'clip_gesture_scaler.pkl'")
    print(f"Gesture info saved as 'clip_gesture_info.json'")
    print(f"Charts saved as 'clip_confusion_matrix.png' and 'clip_training_curves.png'")
