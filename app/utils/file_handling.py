import json
import os
import tarfile
import zipfile
import shutil
from flask import current_app
import magic

def allowed_file(filename):
    allowed_extensions = current_app.config["ALLOWED_EXTENSIONS"]
    return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed_extensions

def validate_file_type(file_path):
    try:
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(file_path)
        
        valid_types = [
            "image/",
            "application/zip",
            "application/x-tar",
            "application/gzip",
            "text/plain",
            "application/x-gzip"
        ]
        
        return any(file_type.startswith(t) for t in valid_types)
    except Exception as e:
        print(f"Error validating file type: {str(e)}")
        return False

def extract_archive(file_path, extract_dir):
    try:
        # Create a temporary extraction directory
        temp_dir = os.path.join(extract_dir, "temp_extract")
        os.makedirs(temp_dir, exist_ok=True)
        
        if file_path.endswith(".zip"):
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                zip_ref.extractall(temp_dir)
        elif file_path.endswith((".tar", ".tar.gz", ".tgz", ".gz")):
            with tarfile.open(file_path) as tar_ref:
                tar_ref.extractall(temp_dir)
        
        # Move files from potential nested folders to main extract directory
        organize_extracted_files(temp_dir, extract_dir)
        
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return True
    except Exception as e:
        print(f"Error extracting archive: {str(e)}")
        return False

def organize_extracted_files(source_dir, target_dir):
    """Move files from potentially nested folders to the target directory"""
    
    # First identify if there's a single directory that contains all files
    contents = os.listdir(source_dir)
    
    if len(contents) == 1 and os.path.isdir(os.path.join(source_dir, contents[0])):
        # If there's only one directory, use its contents
        nested_dir = os.path.join(source_dir, contents[0])
        for item in os.listdir(nested_dir):
            source_path = os.path.join(nested_dir, item)
            target_path = os.path.join(target_dir, item)
            
            if os.path.exists(target_path):
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                else:
                    os.remove(target_path)
                    
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)
    else:
        # Otherwise move all files to the target directory
        for item in contents:
            source_path = os.path.join(source_dir, item)
            target_path = os.path.join(target_dir, item)
            
            if os.path.exists(target_path):
                if os.path.isdir(target_path):
                    shutil.rmtree(target_path)
                else:
                    os.remove(target_path)
                    
            if os.path.isdir(source_path):
                shutil.copytree(source_path, target_path)
            else:
                shutil.copy2(source_path, target_path)

def analyze_uploaded_dataset(upload_dir):
    """Analyze the uploaded dataset and return metadata"""
    metadata = {
        "files": [],
        "directories": [],
        "image_count": 0,
        "classes": []
    }
    
    for root, dirs, files in os.walk(upload_dir):
        rel_path = os.path.relpath(root, upload_dir)
        if rel_path != '.':
            metadata["directories"].append(rel_path)
        
        for file in files:
            file_path = os.path.join(root, file)
            rel_file_path = os.path.join(rel_path, file) if rel_path != '.' else file
            
            # Skip metadata file itself
            if file == "metadata.json":
                continue
                
            file_info = {
                "name": file,
                "path": rel_file_path,
                "size": os.path.getsize(file_path)
            }
            
            # Check if it's an image
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)
            
            if file_type.startswith("image/"):
                file_info["type"] = "image"
                metadata["image_count"] += 1
            else:
                file_info["type"] = file_type
                
            metadata["files"].append(file_info)
    
    # Try to detect class structure
    for directory in metadata["directories"]:
        if "/" not in directory and directory not in [".temp", "temp_extract"]:
            # Top-level directories are potential classes
            metadata["classes"].append(directory)
    
    return metadata

def create_processing_session(session_id, metadata):
    """Create a processing session with metadata"""
    session_dir = os.path.join(current_app.config["UPLOAD_FOLDER"], session_id)
    os.makedirs(session_dir, exist_ok=True)
    
    metadata_path = os.path.join(session_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    return metadata_path
