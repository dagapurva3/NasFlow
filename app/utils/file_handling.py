import json
import os
import tarfile
import zipfile
import shutil
from flask import current_app
import magic
import time
import traceback

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

def extract_archive(file_path, extract_dir, update_progress=None):
    """
    Extract an archive file to the specified directory with progress updates
    
    Args:
        file_path: Path to the archive file
        extract_dir: Directory to extract to
        update_progress: Callback function for progress updates (status, message, progress, details)
    
    Returns:
        bool: Success or failure
    """
    try:
        if update_progress:
            update_progress('extracting', 'Preparing for extraction...', 5, {
                'file_path': file_path,
                'size': f"{os.path.getsize(file_path) / (1024*1024):.2f} MB"
            })
            
        # Create a temporary extraction directory
        temp_dir = os.path.join(extract_dir, "temp_extract")
        os.makedirs(temp_dir, exist_ok=True)
        
        if update_progress:
            update_progress('extracting', 'Analyzing archive...', 10, {
                'type': os.path.splitext(file_path)[1],
                'temp_dir': temp_dir
            })
            
        # Extract based on file type
        if file_path.endswith(".zip"):
            total_files = 0
            extracted_files = 0
            
            # First count files
            with zipfile.ZipFile(file_path, "r") as zip_ref:
                total_files = len(zip_ref.namelist())
                
                if update_progress:
                    update_progress('extracting', f'Extracting ZIP archive with {total_files} files...', 15, {
                        'total_files': total_files
                    })
                
                # Now extract with progress updates
                for i, file in enumerate(zip_ref.namelist()):
                    zip_ref.extract(file, temp_dir)
                    extracted_files += 1
                    
                    # Update progress every 10 files or for the last file
                    if update_progress and (i % 10 == 0 or i == total_files - 1):
                        progress = 15 + int((extracted_files / total_files) * 30)
                        update_progress('extracting', f'Extracting files ({extracted_files}/{total_files})...', 
                                       progress, {
                                           'current_file': file,
                                           'extracted': extracted_files,
                                           'total': total_files
                                       })
                        # Small delay to avoid overwhelming the frontend
                        time.sleep(0.01)
                        
        elif file_path.endswith((".tar", ".tar.gz", ".tgz", ".gz")):
            if update_progress:
                update_progress('extracting', 'Extracting TAR/GZ archive...', 15, {
                    'archive_type': os.path.splitext(file_path)[1]
                })
                
            with tarfile.open(file_path) as tar_ref:
                members = tar_ref.getmembers()
                total_files = len(members)
                
                if update_progress:
                    update_progress('extracting', f'Found {total_files} files in archive...', 20, {
                        'total_files': total_files
                    })
                
                # Extract with progress updates
                for i, member in enumerate(members):
                    tar_ref.extract(member, temp_dir)
                    
                    # Update progress every 10 files or for the last file
                    if update_progress and (i % 10 == 0 or i == total_files - 1):
                        progress = 20 + int((i / total_files) * 30)
                        update_progress('extracting', f'Extracting files ({i+1}/{total_files})...', 
                                       progress, {
                                           'current_file': member.name,
                                           'extracted': i+1,
                                           'total': total_files
                                       })
                        # Small delay to avoid overwhelming the frontend
                        time.sleep(0.01)
        
        if update_progress:
            update_progress('extracting', 'Organizing extracted files...', 50, {
                'source_dir': temp_dir,
                'target_dir': extract_dir
            })
            
        # Move files from potential nested folders to main extract directory
        organize_extracted_files(temp_dir, extract_dir, update_progress)
        
        if update_progress:
            update_progress('extracting', 'Cleaning up temporary files...', 90, {
                'temp_dir': temp_dir
            })
            
        # Clean up temporary directory
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        if update_progress:
            update_progress('extracting', 'Extraction completed successfully.', 95, {
                'extracted_to': extract_dir,
                'status': 'success'
            })
            
        return True
    except Exception as e:
        error_details = traceback.format_exc()
        print(f"Error extracting archive: {str(e)}\n{error_details}")
        
        if update_progress:
            update_progress('failed', f'Extraction failed: {str(e)}', 0, {
                'error': str(e),
                'details': error_details
            })
            
        return False

def organize_extracted_files(source_dir, target_dir, update_progress=None):
    """Move files from potentially nested folders to the target directory with progress updates"""
    
    # First identify if there's a single directory that contains all files
    contents = os.listdir(source_dir)
    
    if update_progress:
        update_progress('extracting', 'Analyzing extracted content structure...', 55, {
            'contents_count': len(contents)
        })
    
    if len(contents) == 1 and os.path.isdir(os.path.join(source_dir, contents[0])):
        # If there's only one directory, use its contents
        nested_dir = os.path.join(source_dir, contents[0])
        nested_contents = os.listdir(nested_dir)
        
        if update_progress:
            update_progress('extracting', f'Found single nested directory with {len(nested_contents)} items...', 60, {
                'nested_dir': contents[0],
                'items_count': len(nested_contents)
            })
        
        for i, item in enumerate(nested_contents):
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
                
            # Update progress occasionally
            if update_progress and (i % 5 == 0 or i == len(nested_contents) - 1):
                progress = 60 + int((i / len(nested_contents)) * 25)
                update_progress('extracting', f'Organizing files ({i+1}/{len(nested_contents)})...', 
                               progress, {
                                   'current_item': item,
                                   'processed': i+1,
                                   'total': len(nested_contents)
                               })
    else:
        # Otherwise move all files to the target directory
        if update_progress:
            update_progress('extracting', f'Organizing {len(contents)} items...', 60, {
                'items_count': len(contents)
            })
            
        for i, item in enumerate(contents):
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
                
            # Update progress occasionally
            if update_progress and (i % 5 == 0 or i == len(contents) - 1):
                progress = 60 + int((i / len(contents)) * 25)
                update_progress('extracting', f'Organizing files ({i+1}/{len(contents)})...', 
                               progress, {
                                   'current_item': item,
                                   'processed': i+1,
                                   'total': len(contents)
                               })

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
