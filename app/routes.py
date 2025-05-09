import os
import uuid
from datetime import datetime
import json

import mlflow
from flask import Response, jsonify, render_template, request, send_from_directory
from mlflow.server import app as mlflow_app
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

from app import app, socketio

from .utils.file_handling import allowed_file, extract_archive, validate_file_type, analyze_uploaded_dataset, create_processing_session


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

            # Extract if archive
            if filename.lower().endswith((".zip", ".tar", ".gz", ".tgz")):
                success = extract_archive(file_path, save_path)
                if not success:
                    return jsonify({"error": "Failed to extract archive"}), 400
            
            # Analyze dataset
            metadata = analyze_uploaded_dataset(save_path)
            metadata["original_filename"] = filename
            metadata["upload_time"] = datetime.now().isoformat()
            
            # Create session with metadata
            create_processing_session(session_id, metadata)
            
            return jsonify({
                "session_id": session_id, 
                "redirect": f"/preprocessing/{session_id}",
                "message": "Upload successful",
                "file_count": len(metadata["files"]),
                "image_count": metadata["image_count"]
            })
            
        except Exception as e:
            return jsonify({"error": f"Upload failed: {str(e)}"}), 500

    return jsonify({"error": "Unsupported file type"}), 400


@app.route("/preprocessing/<session_id>")
def preprocessing(session_id):
    return render_template("preprocessing.html", session_id=session_id)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)


@socketio.on('connect', namespace='/ws')
def ws_connect():
    print('Client connected to WebSocket')

@socketio.on('disconnect', namespace='/ws')
def ws_disconnect():
    print('Client disconnected from WebSocket')

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
    
    # Start preprocessing in background
    executor.submit(preprocess_dataset, session_id, config, metadata)
    
    return jsonify({"message": "Preprocessing started", "session_id": session_id})

def preprocess_dataset(session_id, config, metadata):
    """Background task to preprocess the dataset"""
    session_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
    
    try:
        # Update progress to connected clients
        socketio.emit('progress', {
            'percentage': 10,
            'message': 'Starting preprocessing...'
        }, namespace='/ws', room=session_id)
        
        # Process images
        total_images = metadata["image_count"]
        processed = 0
        
        for file_info in metadata["files"]:
            if file_info.get("type") == "image":
                # Process each image according to config
                file_path = os.path.join(session_path, file_info["path"])
                # Actual preprocessing would happen here
                
                # Update progress
                processed += 1
                percentage = min(90, int(10 + (processed / total_images * 80)))
                
                socketio.emit('progress', {
                    'percentage': percentage,
                    'message': f'Processing images: {processed}/{total_images}'
                }, namespace='/ws', room=session_id)
        
        # Finalize processing
        socketio.emit('progress', {
            'percentage': 100,
            'message': 'Preprocessing complete'
        }, namespace='/ws', room=session_id)
        
        # Update metadata with preprocessing info
        metadata["preprocessing"] = {
            "completed": True,
            "config": config,
            "processed_images": processed,
            "completed_at": datetime.now().isoformat()
        }
        
        with open(os.path.join(session_path, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)
            
        # Emit completion event
        socketio.emit('completed', {
            'session_id': session_id,
            'redirect': f'/training/{session_id}'
        }, namespace='/ws', room=session_id)
        
    except Exception as e:
        socketio.emit('error', {
            'message': f'Error during preprocessing: {str(e)}'
        }, namespace='/ws', room=session_id)
