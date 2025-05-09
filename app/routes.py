import os
import uuid
from datetime import datetime

import mlflow
from flask import Response, jsonify, render_template, request, send_from_directory
from mlflow.server import app as mlflow_app
from werkzeug.middleware.proxy_fix import ProxyFix
from werkzeug.utils import secure_filename

from app import app, socketio

from .utils.file_handling import allowed_file, extract_archive, validate_file_type


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
        session_id = str(uuid.uuid4())
        filename = secure_filename(file.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], session_id)
        os.makedirs(save_path, exist_ok=True)
        file_path = os.path.join(save_path, filename)
        file.save(file_path)

        # Validate file type
        if not validate_file_type(file_path):
            return jsonify({"error": "Invalid file type"}), 400

        # Extract if archive
        if filename.lower().endswith((".zip", ".tar", ".gz")):
            extract_archive(file_path, save_path)

        return jsonify(
            {"session_id": session_id, "redirect": f"/preprocessing/{session_id}"}
        )

    return jsonify({"error": "Invalid file type"}), 400


@app.route("/preprocessing/<session_id>")
def preprocessing(session_id):
    return render_template("preprocessing.html", session_id=session_id)


@app.route("/static/<path:filename>")
def serve_static(filename):
    return send_from_directory(app.config["STATIC_FOLDER"], filename)
