"""
Gesture Recognition Model Architectures
Multiple model sizes for experimentation and comparison
"""

import torch
import torch.nn as nn
from typing import Dict, Any


class TinyGestureModel(nn.Module):
    """
    Tiny model: ~500K parameters
    Fast training, good for quick experiments
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 128)
        self.encoder = nn.LSTM(128, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.relu(self.input_proj(x))
        lstm_out, _ = self.encoder(x)
        x = lstm_out[:, -1, :]
        x = self.dropout(self.relu(self.fc(x)))
        return self.classifier(x)


class SmallGestureModel(nn.Module):
    """
    Small model: ~1.5M parameters
    Good balance for simple datasets
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 256)
        self.input_norm = nn.LayerNorm(256)
        
        self.encoder = nn.LSTM(256, 256, num_layers=2, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.3)
        
        self.fc = nn.Linear(256, 128)
        self.bn = nn.BatchNorm1d(128)
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.relu(x)
        
        lstm_out, _ = self.encoder(x)
        lstm_out = self.lstm_dropout(lstm_out)
        x = lstm_out[:, -1, :]
        
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.dropout(x)
        return self.classifier(x)


class MediumGestureModel(nn.Module):
    """
    Medium model: ~4M parameters
    Good for most datasets, balanced capacity
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 384)
        self.input_norm = nn.LayerNorm(384)
        
        self.encoder = nn.LSTM(384, 384, num_layers=2, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(384, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.relu(x)
        
        lstm_out, _ = self.encoder(x)
        lstm_out = self.lstm_dropout(lstm_out)
        x = lstm_out[:, -1, :]
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        return self.classifier(x)


class LargeGestureModel(nn.Module):
    """
    Large model: ~11M parameters
    High capacity for complex patterns
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 512)
        self.input_norm = nn.LayerNorm(512)
        
        self.encoder = nn.LSTM(512, 768, num_layers=3, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.3)
        
        self.fc1 = nn.Linear(768, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.3)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)
        
        self.classifier = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.relu(x)
        
        lstm_out, _ = self.encoder(x)
        lstm_out = self.lstm_dropout(lstm_out)
        x = lstm_out[:, -1, :]
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        return self.classifier(x)


class XLargeGestureModel(nn.Module):
    """
    XLarge model: ~20M parameters
    Maximum capacity, may overfit on small datasets
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 640)
        self.input_norm = nn.LayerNorm(640)
        
        self.encoder = nn.LSTM(640, 1024, num_layers=3, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.3)
        
        self.classifier = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.relu(x)
        
        lstm_out, _ = self.encoder(x)
        lstm_out = self.lstm_dropout(lstm_out)
        x = lstm_out[:, -1, :]
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        return self.classifier(x)


class XXLargeGestureModel(nn.Module):
    """
    XXLarge model: ~35M parameters
    Extreme capacity for very large datasets
    """
    def __init__(self, input_size: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_size, 768)
        self.input_norm = nn.LayerNorm(768)
        
        self.encoder = nn.LSTM(768, 1280, num_layers=4, batch_first=True)
        self.lstm_dropout = nn.Dropout(0.4)
        
        self.fc1 = nn.Linear(1280, 640)
        self.bn1 = nn.BatchNorm1d(640)
        self.dropout1 = nn.Dropout(0.4)
        
        self.fc2 = nn.Linear(640, 320)
        self.bn2 = nn.BatchNorm1d(320)
        self.dropout2 = nn.Dropout(0.3)
        
        self.fc3 = nn.Linear(320, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(0.2)
        
        self.classifier = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = x.view(x.size(0), x.size(1), -1)
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.relu(x)
        
        lstm_out, _ = self.encoder(x)
        lstm_out = self.lstm_dropout(lstm_out)
        x = lstm_out[:, -1, :]
        
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        
        return self.classifier(x)


# Model registry for easy selection
MODEL_REGISTRY = {
    'tiny': TinyGestureModel,
    'small': SmallGestureModel,
    'medium': MediumGestureModel,
    'large': LargeGestureModel,
    'xlarge': XLargeGestureModel,
    'xxlarge': XXLargeGestureModel,
}

# Model metadata
MODEL_INFO = {
    'tiny': {
        'params': '~500K',
        'description': 'Fast training, good for quick experiments',
        'recommended_for': 'Small datasets, rapid prototyping'
    },
    'small': {
        'params': '~1.5M',
        'description': 'Good balance for simple datasets',
        'recommended_for': 'Simple gesture recognition tasks'
    },
    'medium': {
        'params': '~4M',
        'description': 'Balanced capacity and speed',
        'recommended_for': 'Most gesture recognition tasks'
    },
    'large': {
        'params': '~11M',
        'description': 'High capacity for complex patterns',
        'recommended_for': 'Complex gestures, large datasets'
    },
    'xlarge': {
        'params': '~20M',
        'description': 'Very high capacity, may overfit',
        'recommended_for': 'Very large datasets with complex patterns'
    },
    'xxlarge': {
        'params': '~35M',
        'description': 'Extreme capacity for very large datasets',
        'recommended_for': 'Massive datasets only'
    },
}


def get_model(model_name: str, input_size: int, num_classes: int, device: str = 'cpu') -> nn.Module:
    """
    Factory function to create and return a model
    
    Args:
        model_name: Name of the model ('tiny', 'small', 'medium', 'large', 'xlarge', 'xxlarge')
        input_size: Input feature size
        num_classes: Number of output classes
        device: Device to move model to
    
    Returns:
        Model instance moved to the specified device
    """
    if model_name not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")
    
    model_class = MODEL_REGISTRY[model_name]
    model = model_class(input_size, num_classes).to(device)
    
    return model


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model_name: str):
    """Print information about a model"""
    if model_name not in MODEL_INFO:
        print(f"Unknown model: {model_name}")
        return
    
    info = MODEL_INFO[model_name]
    print(f"\nModel: {model_name.upper()}")
    print(f"Parameters: {info['params']}")
    print(f"Description: {info['description']}")
    print(f"Recommended for: {info['recommended_for']}")


if __name__ == "__main__":
    # Test all models
    print("=" * 60)
    print("GESTURE RECOGNITION MODEL ARCHITECTURES")
    print("=" * 60)
    
    for model_name in MODEL_REGISTRY.keys():
        print_model_info(model_name)
        
        # Create model and count parameters
        model = get_model(model_name, 75, 10)  # 75 features, 10 classes
        params = count_parameters(model)
        print(f"Actual parameters: {params:,}")
        print("-" * 60)

