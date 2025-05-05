import os
import logging
from PIL import Image
from torchvision import transforms
from config import DEFAULT_IMAGE_SIZE, NORMALIZATION_MEAN, NORMALIZATION_STD

logger = logging.getLogger(__name__)

class ImagePreprocessor:
    def __init__(self, session_id, config):
        self.session_id = session_id
        self.config = config
        self.transform = self._build_transform_pipeline()
        self.class_mapping = {}

    def _build_transform_pipeline(self):
        pipeline = [
            transforms.Resize(self.config.get('image_size', DEFAULT_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=self.config.get('normalization_mean', NORMALIZATION_MEAN),
                std=self.config.get('normalization_std', NORMALIZATION_STD)
            )
        ]
        
        if self.config.get('random_hflip', False):
            pipeline.insert(1, transforms.RandomHorizontalFlip())
        if self.config.get('random_rotate', False):
            pipeline.insert(1, transforms.RandomRotation(10))
        
        return transforms.Compose(pipeline)

    def process_image(self, image_path, output_dir):
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                processed = self.transform(img)
                output_path = os.path.join(output_dir, os.path.basename(image_path))
                processed.save(output_path)
                return output_path
        except Exception as e:
            logger.error(f"Error processing {image_path}: {str(e)}")
            return None