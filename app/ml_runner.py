import os
import sys
import json
import logging
import shutil
import threading
from datetime import datetime

# Add the ml_pipeline directory to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ml_pipeline'))

import torch
import mlflow
from flask import jsonify
from flask_socketio import emit

from app import app, socketio

# Import our adapter
from ml_pipeline.train_model_adapter import adapter_for_train_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Active ML processes tracking
active_processes = {}

def run_ml_pipeline(session_id, config):
    """
    Run the ML pipeline on the uploaded dataset
    """
    try:
        # Update status via Socket.IO
        socketio.emit('ml_status', {
            'status': 'starting',
            'message': 'Initializing ML pipeline...',
            'progress': 5,
            'details': {'config': config}
        }, room=session_id)
        
        # Get session directory
        session_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
        processed_dir = os.path.join(session_dir, "processed")
        
        # Check if processed directory exists
        if not os.path.exists(processed_dir):
            raise FileNotFoundError(f"Processed directory not found: {processed_dir}")
        
        # Setup ML directories
        ml_data_dir = os.path.join(session_dir, "ml_data")
        os.makedirs(ml_data_dir, exist_ok=True)
        
        socketio.emit('ml_status', {
            'status': 'processing',
            'message': 'Created ML data directory',
            'progress': 7,
            'details': {'ml_data_dir': ml_data_dir}
        }, room=session_id)
        
        # Create a directory structure for the ML pipeline
        images_dir = os.path.join(ml_data_dir, "Images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Get metadata for the session
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
        
        # Update status
        socketio.emit('ml_status', {
            'status': 'processing',
            'message': 'Preparing dataset for training...',
            'progress': 10
        }, room=session_id)
        
        # Organize processed images into class folders
        class_dirs = {}
        log_entries = []
        log_entries.append(f"Processing images from {processed_dir}")
        
        # First analyze directory structure to determine the actual class folders
        potential_class_dirs = []
        for item in os.listdir(processed_dir):
            item_path = os.path.join(processed_dir, item)
            if os.path.isdir(item_path) and item != "Images":
                potential_class_dirs.append(item)
        
        log_entries.append(f"Detected potential class directories: {potential_class_dirs}")
        
        # If no class dirs found, check if images are directly in processed_dir
        if not potential_class_dirs:
            log_entries.append("No class directories found, assuming flat structure")
            # Use image names or parent directory as pseudo-class
            for file_info in metadata.get("files", []):
                if file_info.get("type") == "image":
                    class_name = "default_class"
                    
                    # Create class directory if it doesn't exist
                    if class_name not in class_dirs:
                        class_dir = os.path.join(images_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)
                        class_dirs[class_name] = class_dir
                    
                    # Copy the image
                    src_path = os.path.join(processed_dir, file_info["path"])
                    filename = os.path.basename(file_info["path"])
                    dst_path = os.path.join(images_dir, class_name, filename)
                    
                    # Ensure directory exists
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    try:
                        shutil.copy2(src_path, dst_path)
                        log_entries.append(f"Copied {filename} to {class_name} class")
                    except Exception as e:
                        log_entries.append(f"Error copying {filename}: {str(e)}")
                        raise RuntimeError(f"Cannot copy file {filename}: {str(e)}")
        else:
            # Use detected class directories
            for file_info in metadata.get("files", []):
                if file_info.get("type") == "image":
                    # Get class from path components
                    file_path = file_info["path"]
                    path_components = os.path.normpath(file_path).split(os.sep)
                    
                    # Find the first path component that matches a class directory
                    class_name = None
                    for component in path_components:
                        if component in potential_class_dirs:
                            class_name = component
                            break
                    
                    # If no class found, use first directory component or default
                    if not class_name and len(path_components) > 1:
                        class_name = path_components[0]
                    elif not class_name:
                        class_name = "default_class"
                    
                    # Create class directory if it doesn't exist
                    if class_name not in class_dirs:
                        class_dir = os.path.join(images_dir, class_name)
                        os.makedirs(class_dir, exist_ok=True)
                        class_dirs[class_name] = class_dir
                        log_entries.append(f"Created class directory for: {class_name}")
                    
                    # Copy the processed image to the class directory
                    src_path = os.path.join(processed_dir, file_path)
                    filename = os.path.basename(file_path)
                    dst_path = os.path.join(images_dir, class_name, filename)
                    
                    # Ensure destination directory exists
                    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                    
                    # Copy file with explicit error handling
                    try:
                        shutil.copy2(src_path, dst_path)
                        log_entries.append(f"Copied {filename} to {class_name} class")
                    except (PermissionError, FileNotFoundError) as e:
                        log_entries.append(f"Error copying {filename}: {str(e)}")
                        raise RuntimeError(f"Cannot copy file {filename}: {str(e)}")
        
        # Update number of classes in metadata
        num_classes = len(class_dirs)
        log_entries.append(f"Final class count: {num_classes}, Classes: {list(class_dirs.keys())}")
        
        metadata["ml_config"] = {
            "num_classes": num_classes,
            "class_dirs": list(class_dirs.keys()),
            "ml_data_dir": ml_data_dir,
            "ml_logs": log_entries,
            **config
        }
        
        # Save updated metadata
        with open(os.path.join(session_dir, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update status
        socketio.emit('ml_status', {
            'status': 'processing',
            'message': f'Dataset prepared with {num_classes} classes. Starting ML training...',
            'progress': 20
        }, room=session_id)
        
        # Import the test2.py (after dataset is prepared)
        from test2 import (
            create_label_csv, 
            CustomImageDataset, 
            train_model, 
            create_model, 
            predict_and_evaluate
        )
        
        # Apply our adapter to the train_model function
        train_model = adapter_for_train_model(train_model)
        
        # No need to override global variables that don't exist
        # Just use the num_classes we already have
        
        # Create label CSVs
        csv_dir = os.path.join(ml_data_dir, "csv")
        os.makedirs(csv_dir, exist_ok=True)
        
        socketio.emit('ml_status', {
            'status': 'processing',
            'message': 'Creating label CSVs...',
            'progress': 25,
            'details': {
                'num_classes': num_classes,
                'classes': list(class_dirs.keys()),
                'csv_dir': csv_dir
            }
        }, room=session_id)
        
        # Create label CSVs from the organized images
        create_label_csv(images_dir, csv_dir, train_ratio=0.8)
        
        # Setup MLflow tracking
        mlflow_dir = os.path.join(ml_data_dir, "mlruns")
        os.makedirs(mlflow_dir, exist_ok=True)
        mlflow.set_tracking_uri(f"file:{mlflow_dir}")
        
        experiment_name = f"session_{session_id}"
        # Create experiment or get existing one
        try:
            experiment = mlflow.get_experiment_by_name(experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
            else:
                experiment_id = experiment.experiment_id
        except Exception as e:
            logger.error(f"MLflow experiment error: {str(e)}")
            experiment_id = mlflow.create_experiment(experiment_name)
        
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        with mlflow.start_run(experiment_id=experiment_id) as run:
            run_id = run.info.run_id
            
            # Log parameters
            mlflow.log_params({
                "session_id": session_id,
                "num_classes": num_classes,
                "train_ratio": 0.8,
                **config
            })
            
            # Setup model parameters from config
            model_type = config.get("model_type", "simple_cnn")
            dropout_rate = float(config.get("dropout_rate", 0.3))
            learning_rate = float(config.get("learning_rate", 0.001))
            batch_size = int(config.get("batch_size", 32))
            num_epochs = int(config.get("num_epochs", 10))
            optimizer_name = config.get("optimizer", "Adam")
            
            # Setup data loaders
            import torch
            import torchvision.transforms as transforms
            from torch.utils.data import DataLoader
            
            # Data transformation
            transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(10),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            val_transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ]
            )

            # Create datasets and dataloaders
            socketio.emit('ml_status', {
                'status': 'processing',
                'message': 'Creating data loaders...',
                'progress': 30, 
                'details': {
                    'train_csv': os.path.join(csv_dir, "train_labels.csv"),
                    'val_csv': os.path.join(csv_dir, "val_labels.csv")
                }
            }, room=session_id)
            
            try:
                train_dataset = CustomImageDataset(
                    images_dir, 
                    os.path.join(csv_dir, "train_labels.csv"), 
                    transform
                )
                
                val_dataset = CustomImageDataset(
                    images_dir, 
                    os.path.join(csv_dir, "val_labels.csv"), 
                    val_transform
                )
                
                train_loader = DataLoader(
                    train_dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    num_workers=2,
                )
                
                val_loader = DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=2,
                )
                
                socketio.emit('ml_status', {
                    'status': 'processing',
                    'message': f'Created data loaders with {len(train_dataset)} training and {len(val_dataset)} validation samples',
                    'progress': 33,
                    'details': {
                        'train_samples': len(train_dataset),
                        'val_samples': len(val_dataset)
                    }
                }, room=session_id)
                
                # Create model
                socketio.emit('ml_status', {
                    'status': 'processing',
                    'message': f'Creating {model_type} model with {num_classes} classes...',
                    'progress': 35,
                    'details': {
                        'model': model_type,
                        'num_classes': num_classes,
                        'dropout': dropout_rate
                    }
                }, room=session_id)
                
                model = create_model(model_type, num_classes, dropout_rate)
                
                # Define loss function and optimizer
                criterion = torch.nn.CrossEntropyLoss()
                if optimizer_name == "Adam":
                    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
                else:
                    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
                
                socketio.emit('ml_status', {
                    'status': 'processing',
                    'message': f'Configured {optimizer_name} optimizer with learning rate {learning_rate}',
                    'progress': 37,
                    'details': {
                        'optimizer': optimizer_name,
                        'lr': learning_rate,
                        'device': str(device)
                    }
                }, room=session_id)
                
                # Initialize tensorboard writer
                from torch.utils.tensorboard import SummaryWriter
                tb_logdir = os.path.join(ml_data_dir, "runs")
                writer = SummaryWriter(tb_logdir)
                
                # Train model
                socketio.emit('ml_status', {
                    'status': 'processing',
                    'message': 'Training model...',
                    'progress': 40
                }, room=session_id)
                
                # Custom progress callback for Socket.IO updates
                def progress_callback(epoch, epochs, train_loss, val_loss, train_acc, val_acc):
                    progress = int(40 + (epoch / epochs) * 50)
                    socketio.emit('ml_status', {
                        'status': 'processing',
                        'message': f'Training model: Epoch {epoch}/{epochs}',
                        'progress': progress,
                        'metrics': {
                            'train_loss': train_loss,
                            'val_loss': val_loss,
                            'train_acc': train_acc,
                            'val_acc': val_acc
                        }
                    }, room=session_id)
                    
                    # Log metrics to MLflow
                    mlflow.log_metrics({
                        f"train_loss_epoch_{epoch}": train_loss,
                        f"val_loss_epoch_{epoch}": val_loss,
                        f"train_acc_epoch_{epoch}": train_acc,
                        f"val_acc_epoch_{epoch}": val_acc
                    }, step=epoch)
                
                # Add a custom callback to train_model function
                val_accuracy = train_model(
                    model,
                    train_loader,
                    val_loader,
                    optimizer,
                    criterion,
                    num_epochs,
                    writer,
                    0,  # trial number
                    progress_callback=progress_callback  # Custom callback
                )
                
                # Evaluate model
                socketio.emit('ml_status', {
                    'status': 'processing',
                    'message': 'Evaluating model...',
                    'progress': 90
                }, room=session_id)
                
                metrics = predict_and_evaluate(model, val_loader)
                
                # Log metrics
                mlflow.log_metrics({
                    "final_accuracy": metrics["accuracy"],
                    "final_f1_score": metrics["f1_score"]
                })
                
                # Log artifacts
                mlflow.log_artifact(metrics["confusion_matrix_path"])
                
                # Save model
                model_path = os.path.join(ml_data_dir, "model.pth")
                torch.save(model.state_dict(), model_path)
                mlflow.log_artifact(model_path)
                
                # Final update
                socketio.emit('ml_status', {
                    'status': 'complete',
                    'message': 'ML training complete',
                    'progress': 100,
                    'metrics': {
                        'accuracy': float(metrics["accuracy"]),
                        'f1_score': float(metrics["f1_score"])
                    },
                    'mlflow_run_id': run_id
                }, room=session_id)
                
                # Update metadata
                metadata["ml_status"] = "complete"
                metadata["ml_completion_time"] = datetime.now().isoformat()
                metadata["ml_metrics"] = {
                    'accuracy': float(metrics["accuracy"]),
                    'f1_score': float(metrics["f1_score"])
                }
                metadata["mlflow_run_id"] = run_id
                
                with open(os.path.join(session_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Clean up
                writer.close()
                
            except Exception as e:
                logger.error(f"Error in ML pipeline: {str(e)}")
                socketio.emit('ml_status', {
                    'status': 'failed',
                    'message': f'ML pipeline failed: {str(e)}',
                    'progress': 0
                }, room=session_id)
                
                # Update metadata with error
                metadata["ml_status"] = "failed"
                metadata["ml_error"] = str(e)
                
                with open(os.path.join(session_dir, "metadata.json"), "w") as f:
                    json.dump(metadata, f, indent=2)
                
                # Log failure in MLflow
                mlflow.log_param("error", str(e))
                raise e
            
    except Exception as e:
        logger.error(f"ML runner error: {str(e)}")
        socketio.emit('ml_status', {
            'status': 'failed',
            'message': f'ML pipeline failed: {str(e)}',
            'progress': 0
        }, room=session_id)
    finally:
        # Remove from active processes
        if session_id in active_processes:
            del active_processes[session_id]


def start_ml_process(session_id, config):
    """Start ML process in a separate thread"""
    if session_id in active_processes:
        return False, "ML process already running for this session"
    
    # Create thread
    ml_thread = threading.Thread(
        target=run_ml_pipeline,
        args=(session_id, config),
        daemon=True
    )
    
    # Store thread
    active_processes[session_id] = {
        "thread": ml_thread,
        "start_time": datetime.now().isoformat(),
        "config": config
    }
    
    # Start thread
    ml_thread.start()
    
    return True, "ML process started" 