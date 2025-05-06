import os
import magic
import zipfile
import tarfile
from config import ALLOWED_EXTENSIONS
import json

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_type(file_path):
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(file_path)
    return any([
        file_type.startswith('image/'),
        file_type == 'application/zip',
        file_type == 'application/x-tar',
        file_type == 'application/gzip',
        file_type == 'text/plain'
    ])

def extract_archive(file_path, extract_dir):
    if file_path.endswith('.zip'):
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
    elif file_path.endswith(('.tar', '.tar.gz', '.tgz')):
        with tarfile.open(file_path) as tar_ref:
            tar_ref.extractall(extract_dir)

def create_processing_session(session_id, metadata):
    session_dir = os.path.join(current_app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_dir, exist_ok=True)
    with open(os.path.join(session_dir, 'metadata.json'), 'w') as f:
        json.dump(metadata, f)