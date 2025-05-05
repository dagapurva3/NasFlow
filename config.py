import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
STATIC_FOLDER = os.path.join(BASE_DIR, 'app/static')
ALLOWED_EXTENSIONS = {'zip', 'tar', 'gz', 'csv', 'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 2 * 1024 * 1024 * 1024  # 2GB

# Image preprocessing defaults
DEFAULT_IMAGE_SIZE = (224, 224)
NORMALIZATION_MEAN = [0.485, 0.456, 0.406]
NORMALIZATION_STD = [0.229, 0.224, 0.225]

# Secret key for session management
SECRET_KEY = 'your-secret-key-here'