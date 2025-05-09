import os
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

def process_images(input_path, output_path, size=(224, 224), normalize=True, 
                  random_flip=True, random_rotate=True, normalize_mean=None, 
                  normalize_std=None):
    """Process an image for machine learning"""
    
    if normalize_mean is None:
        normalize_mean = [0.485, 0.456, 0.406]  # ImageNet mean
    
    if normalize_std is None:
        normalize_std = [0.229, 0.224, 0.225]  # ImageNet std
    
    # Create transformation pipeline
    transform_list = [
        transforms.Resize(size),
    ]
    
    # Add augmentations if requested
    if random_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
        
    if random_rotate:
        transform_list.append(transforms.RandomRotation(10))
    
    # Add normalization if requested
    if normalize:
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=normalize_mean, std=normalize_std)
        ])
    else:
        transform_list.append(transforms.ToTensor())
    
    # Create the transform
    transform = transforms.Compose(transform_list)
    
    try:
        # Open the image
        img = Image.open(input_path).convert('RGB')
        
        # Apply transformations
        img_tensor = transform(img)
        
        # Create the output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the processed image (convert back to PIL Image)
        if normalize:
            # De-normalize if we normalized
            inv_normalize = transforms.Normalize(
                mean=[-m/s for m, s in zip(normalize_mean, normalize_std)],
                std=[1/s for s in normalize_std]
            )
            img_tensor = inv_normalize(img_tensor)
        
        # Convert from tensor to PIL image
        processed_img = transforms.ToPILImage()(img_tensor)
        
        # Save the processed image
        processed_img.save(output_path)
        
        return True
    except Exception as e:
        print(f"Error processing image {input_path}: {str(e)}")
        return False

def process_directory(input_dir, output_dir, **kwargs):
    """Process all images in a directory"""
    processed_count = 0
    total_count = 0
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Process all files in the directory
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
                total_count += 1
                input_path = os.path.join(root, file)
                
                # Create relative path to maintain directory structure
                rel_path = os.path.relpath(input_path, input_dir)
                output_path = os.path.join(output_dir, rel_path)
                
                # Ensure output directory exists
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                
                # Process the image
                if process_images(input_path, output_path, **kwargs):
                    processed_count += 1
    
    return {
        "processed": processed_count,
        "total": total_count
    }
