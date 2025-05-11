import os
import uuid
from datetime import datetime
import json
import traceback
import time

from flask import jsonify, render_template, request, send_from_directory, redirect, url_for
from werkzeug.utils import secure_filename
from flask_socketio import join_room

from app import app, executor, socketio

from .utils.file_handling import (
    allowed_file, 
    validate_file_type, 
    extract_archive, 
    analyze_uploaded_dataset, 
    create_processing_session
)
from .preprocessing.data_preprocessor import process_images
from .ml_runner import start_ml_process, active_processes

# Global variable to store extraction status
extraction_status = {}

@app.route("/")
def index():
    return render_template("upload.html")

@app.route("/upload", methods=["POST"])
def handle_upload():
    if "dataset" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["dataset"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if file and allowed_file(file.filename):
        try:
            # Generate session ID
            session_id = str(uuid.uuid4())
            filename = secure_filename(file.filename)
            
            # Create session directory
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
            os.makedirs(save_path, exist_ok=True)
            
            # Save uploaded file
            file_path = os.path.join(save_path, filename)
            file.save(file_path)
            
            # Validate file type
            if not validate_file_type(file_path):
                return jsonify({"error": "Invalid file type or corrupted file"}), 400

            # Initialize session metadata
            initial_metadata = {
                "original_filename": filename,
                "upload_time": datetime.now().isoformat(),
                "status": "uploaded",
                "extraction_needed": filename.lower().endswith((".zip", ".tar", ".gz", ".tgz"))
            }
            
            # Create initial session
            create_processing_session(session_id, initial_metadata)
            
            # Start extraction process in background if needed
            if initial_metadata["extraction_needed"]:
                extraction_status[session_id] = "pending"
                executor.submit(
                    handle_extraction, 
                    session_id, 
                    file_path, 
                    save_path, 
                    filename
                )
                return jsonify({
                    "session_id": session_id, 
                    "redirect": f"/extraction_status/{session_id}",
                    "message": "File uploaded, extraction starting"
                })
            else:
                # Not an archive, go straight to dataset analysis
                metadata = analyze_uploaded_dataset(save_path)
                metadata.update(initial_metadata)
                metadata["status"] = "ready_for_preprocessing"
                create_processing_session(session_id, metadata)
                
                return jsonify({
                    "session_id": session_id, 
                    "redirect": f"/preprocessing/{session_id}",
                    "message": "Upload successful, ready for preprocessing"
                })
            
        except Exception as e:
            # Log the full exception for debugging
            print(f"Upload error: {str(e)}")
            traceback.print_exc()
            return jsonify({"error": f"Upload failed: {str(e)}"}), 500

    return jsonify({"error": "Unsupported file type"}), 400

def handle_extraction(session_id, file_path, save_path, filename):
    """Background task to handle extraction and analysis"""
    try:
        # Create extraction log in metadata
        metadata_path = os.path.join(save_path, "metadata.json")
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        
        # Update status
        metadata["status"] = "extracting"
        metadata["extraction_start_time"] = datetime.now().isoformat()
        metadata["extraction_log"] = []  # Add log container
        create_processing_session(session_id, metadata)
        
        # Define progress callback function
        def update_extraction_progress(status, message, progress, details=None):
            # Add to log
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "status": status,
                "message": message,
                "progress": progress
            }
            if details:
                log_entry["details"] = details
                
            # Update metadata with latest log entry
            with open(metadata_path, "r") as f:
                current_metadata = json.load(f)
            
            # Keep log from growing too large (retain last 50 entries)
            if "extraction_log" not in current_metadata:
                current_metadata["extraction_log"] = []
                
            current_metadata["extraction_log"].append(log_entry)
            if len(current_metadata["extraction_log"]) > 50:
                current_metadata["extraction_log"] = current_metadata["extraction_log"][-50:]
            
            create_processing_session(session_id, current_metadata)
            
            # Emit status update via Socket.IO
            socketio.emit('extraction_update', {
                'status': status,
                'message': message,
                'progress': progress,
                'details': details
            }, room=session_id)
        
        # Emit initial status update
        update_extraction_progress('extracting', f'Extracting {filename}...', 10)
        
        # Extract the archive with progress updates
        success = extract_archive(file_path, save_path, update_extraction_progress)
        
        if not success:
            metadata["status"] = "extraction_failed"
            metadata["extraction_error"] = "Failed to extract archive"
            create_processing_session(session_id, metadata)
            
            # Final status update is handled by extract_archive on failure
            return
        
        # Update status for analysis phase
        update_extraction_progress('analyzing', 'Analyzing extracted files...', 50)
        
        # Analyze the extracted data
        analysis_start = time.time()
        analysis = analyze_uploaded_dataset(save_path)
        analysis_duration = time.time() - analysis_start
        
        update_extraction_progress('analyzing', f'Analysis complete. Found {analysis["image_count"]} images in {len(analysis["classes"])} classes.', 80, {
            'image_count': analysis["image_count"],
            'classes': len(analysis["classes"]),
            'duration_seconds': round(analysis_duration, 2)
        })
        
        # Update metadata with analysis results
        metadata.update(analysis)
        metadata["status"] = "ready_for_preprocessing"
        metadata["extraction_end_time"] = datetime.now().isoformat()
        create_processing_session(session_id, metadata)
        
        # Final update
        update_extraction_progress('complete', 'Extraction and analysis complete', 100, {
            'image_count': analysis["image_count"],
            'classes': len(analysis["classes"]),
            'redirect': f"/preprocessing/{session_id}"
        })
        
        # Update global status
        extraction_status[session_id] = "complete"
        
    except Exception as e:
        print(f"Extraction error: {str(e)}")
        error_details = traceback.format_exc()
        
        # Update metadata with error
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            
            metadata["status"] = "extraction_failed"
            metadata["extraction_error"] = str(e)
            metadata["extraction_error_details"] = error_details
            create_processing_session(session_id, metadata)
        except Exception as inner_e:
            print(f"Error updating metadata: {str(inner_e)}")
        
        # Emit error update
        socketio.emit('extraction_update', {
            'status': 'failed',
            'message': f'Extraction failed: {str(e)}',
            'progress': 0,
            'details': {
                'error': str(e),
                'traceback': error_details
            }
        }, room=session_id)
        
        # Update global status
        extraction_status[session_id] = "failed"

@app.route("/extraction_status/<session_id>")
def show_extraction_status(session_id):
    # Get session directory
    session_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    
    # Check if session exists
    if not os.path.exists(session_dir):
        return redirect(url_for('index'))
    
    # Get metadata
    try:
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
    except Exception:
        return redirect(url_for('index'))
    
    # Check if we should redirect to preprocessing
    if metadata.get("status") == "ready_for_preprocessing":
        return redirect(url_for('preprocessing', session_id=session_id))
    
    return render_template(
        "extraction.html",
        session_id=session_id,
        filename=metadata.get("original_filename", "Unknown"),
        status=metadata.get("status", "unknown")
    )

@app.route("/preprocessing/<session_id>")
def preprocessing(session_id):
    # Get session directory
    session_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    
    # Check if session exists
    if not os.path.exists(session_dir):
        return redirect(url_for('index'))
    
    # Get metadata
    try:
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
    except Exception:
        return redirect(url_for('index'))
    
    # Check if extraction is still needed
    if metadata.get("status") not in ["ready_for_preprocessing", "preprocessing", "preprocessing_complete"]:
        if metadata.get("extraction_needed", False) and metadata.get("status") != "extraction_failed":
            return redirect(url_for('show_extraction_status', session_id=session_id))
    
    return render_template(
        "preprocessing.html", 
        session_id=session_id,
        metadata=metadata
    )

@app.route("/api/preprocess/<session_id>", methods=["POST"])
def start_preprocessing(session_id):
    # Get preprocessing configuration
    config = request.json
    
    # Validate session exists
    session_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    if not os.path.exists(session_path):
        return jsonify({"error": "Invalid session ID"}), 404
    
    # Load session metadata
    try:
        with open(os.path.join(session_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
    except Exception:
        return jsonify({"error": "Failed to load session metadata"}), 500
    
    # Validate session is ready for preprocessing
    if metadata.get("status") != "ready_for_preprocessing":
        return jsonify({"error": "Session is not ready for preprocessing"}), 400
    
    # Update status
    metadata["status"] = "preprocessing"
    metadata["preprocessing_start_time"] = datetime.now().isoformat()
    metadata["preprocessing_config"] = config
    create_processing_session(session_id, metadata)
    
    # Start preprocessing in background
    executor.submit(preprocess_dataset, session_id, config, metadata)
    
    return jsonify({
        "message": "Preprocessing started", 
        "session_id": session_id
    })

def preprocess_dataset(session_id, config, metadata):
    """Background task to preprocess the dataset"""
    session_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    
    try:
        # Update progress to connected clients
        socketio.emit('preprocessing_update', {
            'status': 'processing',
            'message': 'Starting preprocessing...',
            'progress': 5
        }, room=session_id)
        
        # Get image size from config
        image_width = int(config.get("image_width", 224))
        image_height = int(config.get("image_height", 224))
        
        # Process images
        total_images = metadata.get("image_count", 0)
        processed = 0
        
        # Create output directories
        processed_dir = os.path.join(session_path, "processed")
        os.makedirs(processed_dir, exist_ok=True)
        
        for file_info in metadata.get("files", []):
            if file_info.get("type") == "image":
                # Get image path
                img_path = os.path.join(session_path, file_info["path"])
                rel_path = file_info["path"]
                
                # Get output path - maintain directory structure
                output_dir = os.path.dirname(os.path.join(processed_dir, rel_path))
                os.makedirs(output_dir, exist_ok=True)
                
                # Process the image
                process_images(
                    img_path, 
                    os.path.join(processed_dir, rel_path),
                    size=(image_width, image_height),
                    normalize=True,
                    random_flip=config.get("random_hflip", True),
                    random_rotate=config.get("random_rotate", True)
                )
                
                # Update progress
                processed += 1
                percentage = min(90, int(5 + (processed / total_images * 85)))
                
                if processed % 10 == 0 or processed == total_images:
                    socketio.emit('preprocessing_update', {
                        'status': 'processing',
                        'message': f'Processing images: {processed}/{total_images}',
                        'progress': percentage
                    }, room=session_id)
        
        # Finalize processing
        socketio.emit('preprocessing_update', {
            'status': 'complete',
            'message': 'Preprocessing complete',
            'progress': 100,
            'redirect': f'/training/{session_id}'
        }, room=session_id)
        
        # Update metadata with preprocessing info
        metadata["status"] = "preprocessing_complete"
        metadata["preprocessing_end_time"] = datetime.now().isoformat()
        metadata["processed_images"] = processed
        metadata["processed_dir"] = "processed"
        
        create_processing_session(session_id, metadata)
            
    except Exception as e:
        print(f"Preprocessing error: {str(e)}")
        traceback.print_exc()
        
        # Update metadata with error
        try:
            with open(os.path.join(session_path, "metadata.json"), "r") as f:
                metadata = json.load(f)
            
            metadata["status"] = "preprocessing_failed"
            metadata["preprocessing_error"] = str(e)
            create_processing_session(session_id, metadata)
        except Exception:
            pass
        
        # Emit error update
        socketio.emit('preprocessing_update', {
            'status': 'failed',
            'message': f'Preprocessing failed: {str(e)}',
            'progress': 0
        }, room=session_id)

@socketio.on('join')
def on_join(data):
    session_id = data.get('session_id')
    if session_id:
        print(f"Client joined room: {session_id}")
        join_room(session_id)

@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)

@app.route("/mlflow-dashboard")
def mlflow_dashboard():
    """
    Renders the MLflow dashboard page with embedded iframe
    """
    return render_template("mlflow_dashboard.html")

@app.route("/training/<session_id>")
def training(session_id):
    """
    Renders the ML training page for a session
    """
    # Get session directory
    session_dir = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    
    # Check if session exists
    if not os.path.exists(session_dir):
        return redirect(url_for('index'))
    
    # Get metadata
    try:
        with open(os.path.join(session_dir, "metadata.json"), "r") as f:
            metadata = json.load(f)
    except Exception:
        return redirect(url_for('index'))
    
    # Check if preprocessing is complete
    if metadata.get("status") != "preprocessing_complete":
        return redirect(url_for('preprocessing', session_id=session_id))
    
    return render_template(
        "training.html", 
        session_id=session_id,
        metadata=metadata
    )

@app.route("/api/train/<session_id>", methods=["POST"])
def start_training(session_id):
    """
    API endpoint to start ML training on a preprocessed dataset
    """
    # Get training configuration from request
    config = request.json
    
    # Validate session exists
    session_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    if not os.path.exists(session_path):
        return jsonify({"error": "Invalid session ID"}), 404
    
    # Load session metadata
    try:
        with open(os.path.join(session_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
    except Exception:
        return jsonify({"error": "Failed to load session metadata"}), 500
    
    # Validate session is ready for training
    if metadata.get("status") != "preprocessing_complete":
        return jsonify({"error": "Session is not ready for training"}), 400
    
    # Check if ML process is already running
    if session_id in active_processes:
        return jsonify({
            "error": "ML process already running", 
            "start_time": active_processes[session_id]["start_time"]
        }), 409
    
    # Start ML process in background
    success, message = start_ml_process(session_id, config)
    
    if success:
        # Update metadata
        metadata["ml_status"] = "training"
        metadata["ml_start_time"] = datetime.now().isoformat()
        metadata["ml_config"] = config
        
        # Save metadata
        with open(os.path.join(session_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        return jsonify({
            "message": "Training started", 
            "session_id": session_id
        })
    else:
        return jsonify({"error": message}), 500

@app.route("/api/training-status/<session_id>")
def training_status(session_id):
    """
    API endpoint to get the current status of ML training
    """
    # Validate session exists
    session_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    if not os.path.exists(session_path):
        return jsonify({"error": "Invalid session ID"}), 404
    
    # Load session metadata
    try:
        with open(os.path.join(session_path, "metadata.json"), "r") as f:
            metadata = json.load(f)
    except Exception:
        return jsonify({"error": "Failed to load session metadata"}), 500
    
    # Check if ML process is running
    is_active = session_id in active_processes
    
    # Get status from metadata
    status = metadata.get("ml_status", "not_started")
    
    return jsonify({
        "session_id": session_id,
        "status": status,
        "is_active": is_active,
        "ml_config": metadata.get("ml_config", {}),
        "ml_metrics": metadata.get("ml_metrics", {}),
        "ml_start_time": metadata.get("ml_start_time"),
        "ml_completion_time": metadata.get("ml_completion_time"),
        "mlflow_run_id": metadata.get("mlflow_run_id")
    })

@socketio.on('ml_connect')
def on_ml_connect(data):
    """
    Socket.IO event for connecting to ML updates
    """
    session_id = data.get('session_id')
    if session_id:
        print(f"Client connected for ML updates on session {session_id}")
        join_room(session_id)