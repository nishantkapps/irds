"""
Tensor utilities for PyTorch-only operations
"""
import torch
import logging

logger = logging.getLogger(__name__)

def tensor_train_test_split(X, y, test_size=0.2, random_state=42, stratify=None):
    """
    Custom train-test split using PyTorch tensors only
    
    Args:
        X: Input tensor
        y: Target tensor
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
        stratify: Whether to stratify the split (requires integer labels)
    
    Returns:
        X_train, X_test, y_train, y_test: Split tensors
    """
    logger.info("Performing tensor-based train-test split...")
    
    # Set random seed
    torch.manual_seed(random_state)
    
    n_samples = X.size(0)
    n_test = int(n_samples * test_size)
    n_train = n_samples - n_test
    
    if stratify is not None:
        # Stratified split
        logger.info("Performing stratified split...")
        
        # Get unique classes and their counts
        unique_classes, class_counts = torch.unique(y, return_counts=True)
        
        train_indices = []
        test_indices = []
        
        for class_label, count in zip(unique_classes, class_counts):
            # Get indices for this class
            class_indices = torch.where(y == class_label)[0]
            
            # Shuffle indices for this class
            shuffled_indices = class_indices[torch.randperm(len(class_indices))]
            
            # Calculate split for this class
            n_test_class = int(count * test_size)
            n_train_class = count - n_test_class
            
            # Split indices
            train_indices.append(shuffled_indices[:n_train_class])
            test_indices.append(shuffled_indices[n_train_class:n_train_class + n_test_class])
        
        # Concatenate all indices
        train_indices = torch.cat(train_indices)
        test_indices = torch.cat(test_indices)
        
        # Shuffle the final indices
        train_indices = train_indices[torch.randperm(len(train_indices))]
        test_indices = test_indices[torch.randperm(len(test_indices))]
        
    else:
        # Random split
        logger.info("Performing random split...")
        
        # Create random permutation of indices
        indices = torch.randperm(n_samples)
        
        # Split indices
        train_indices = indices[:n_train]
        test_indices = indices[n_train:]
    
    # Create train/test splits
    X_train = X[train_indices]
    X_test = X[test_indices]
    y_train = y[train_indices]
    y_test = y[test_indices]
    
    logger.info(f"Split complete: Train={len(X_train)}, Test={len(X_test)}")
    
    return X_train, X_test, y_train, y_test

def tensor_scaler_fit_transform(X_train, X_test=None):
    """
    Custom scaling using PyTorch tensors only
    
    Args:
        X_train: Training data tensor
        X_test: Test data tensor (optional)
    
    Returns:
        X_train_scaled: Scaled training data
        X_test_scaled: Scaled test data (if provided)
        scaler_params: Dictionary with mean and std for inverse transform
    """
    logger.info("Performing tensor-based scaling...")
    
    # Calculate mean and std for each feature
    mean = torch.mean(X_train, dim=0)
    std = torch.std(X_train, dim=0)
    
    # Avoid division by zero
    std = torch.where(std == 0, torch.ones_like(std), std)
    
    # Scale training data
    X_train_scaled = (X_train - mean) / std
    
    scaler_params = {
        'mean': mean,
        'std': std
    }
    
    if X_test is not None:
        # Scale test data using training statistics
        X_test_scaled = (X_test - mean) / std
        return X_train_scaled, X_test_scaled, scaler_params
    
    return X_train_scaled, scaler_params

def tensor_scaler_transform(X, scaler_params):
    """
    Transform data using pre-computed scaler parameters
    
    Args:
        X: Data tensor to transform
        scaler_params: Dictionary with 'mean' and 'std' keys
    
    Returns:
        X_scaled: Scaled data tensor
    """
    mean = scaler_params['mean']
    std = scaler_params['std']
    
    return (X - mean) / std

def tensor_scaler_inverse_transform(X_scaled, scaler_params):
    """
    Inverse transform scaled data back to original scale
    
    Args:
        X_scaled: Scaled data tensor
        scaler_params: Dictionary with 'mean' and 'std' keys
    
    Returns:
        X: Original scale data tensor
    """
    mean = scaler_params['mean']
    std = scaler_params['std']
    
    return X_scaled * std + mean

