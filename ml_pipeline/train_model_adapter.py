import torch

def train_model_with_callback(
    model, 
    train_loader, 
    val_loader, 
    optimizer, 
    criterion, 
    num_epochs, 
    writer, 
    trial_num=0,
    progress_callback=None
):
    """
    Extended version of the train_model function that supports a progress callback
    to report training progress in real-time.
    
    Parameters:
    - model: PyTorch model
    - train_loader: Training data loader
    - val_loader: Validation data loader
    - optimizer: PyTorch optimizer
    - criterion: Loss function
    - num_epochs: Number of training epochs
    - writer: TensorBoard writer
    - trial_num: Trial number (for Optuna)
    - progress_callback: Callback function for reporting progress
                         signature: callback(epoch, epochs, train_loss, val_loss, train_acc, val_acc)
    
    Returns:
    - Validation accuracy
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Import needed functions
    import sys
    import os
    
    # Add ml_pipeline to path if not already there
    ml_pipeline_path = os.path.dirname(os.path.abspath(__file__))
    if ml_pipeline_path not in sys.path:
        sys.path.append(ml_pipeline_path)
    
    # Import evaluate_model from test2.py
    from test2 import evaluate_model
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward + backward + optimize
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        # Calculate training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total
        
        # Evaluate on validation set
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        # Append to lists for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Log to tensorboard
        writer.add_scalar(f'Loss/train_trial_{trial_num}', train_loss, epoch)
        writer.add_scalar(f'Loss/val_trial_{trial_num}', val_loss, epoch)
        writer.add_scalar(f'Accuracy/train_trial_{trial_num}', train_acc, epoch)
        writer.add_scalar(f'Accuracy/val_trial_{trial_num}', val_acc, epoch)
        
        # Call progress callback if provided
        if progress_callback:
            progress_callback(epoch + 1, num_epochs, train_loss, val_loss, train_acc, val_acc)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= 3:  # Using patience value of 3
            print(f"Early stopping triggered after {epoch + 1} epochs")
            break
            
    return val_acc

# Create an adapter function that matches the original signature
def adapter_for_train_model(original_train_model_func):
    """
    Create an adapter for the original train_model function that adds support for callbacks
    """
    def wrapped_train_model(*args, **kwargs):
        # Extract the progress_callback if it exists
        progress_callback = kwargs.pop('progress_callback', None)
        
        if progress_callback:
            # Use our version with callback support
            return train_model_with_callback(*args, progress_callback=progress_callback, **kwargs)
        else:
            # Use the original function
            return original_train_model_func(*args, **kwargs)
            
    return wrapped_train_model 