"""
Models Loader.

Централизованная загрузка и управление всеми inference моделями.
"""

import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.inference_primary import PrimaryClassifier
from src.inference_segmentation import RetinaSegmentationModel
from src.inference_detection import BloodCellDetector
from src.inference_cifar10 import CIFAR10Classifier
from bot.config import MODEL_PATHS, MODELS_DIR

logger = logging.getLogger(__name__)


class ModelsManager:
    """
    Singleton manager for all inference models.
    
    Загружает модели один раз при инициализации и предоставляет
    единый интерфейс для inference.
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize models (only once)."""
        if self._initialized:
            return
        
        logger.info("Initializing models...")
        
        try:
            # Load primary classifier
            logger.info("Loading primary classifier...")
            self.primary_classifier = PrimaryClassifier(
                model_path=MODEL_PATHS['primary']
            )
            
            # Load U-Net segmentation
            logger.info("Loading U-Net segmentation model...")
            self.segmentation_model = RetinaSegmentationModel(
                model_path=MODEL_PATHS['unet']
            )
            
            # Load YOLO detector
            logger.info("Loading YOLO blood cell detector...")
            self.detection_model = BloodCellDetector(
                model_path=MODEL_PATHS['yolo']
            )
            
            # Load CIFAR-10 classifier
            logger.info("Loading CIFAR-10 classifier...")
            self.cifar10_classifier = CIFAR10Classifier(
                model_path=MODEL_PATHS['cifar10']
            )
            
            self._initialized = True
            logger.info("✓ All models loaded successfully!")
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
    
    def classify_image_type(self, image):
        """
        Classify image into retina/blood/scene.
        
        Args:
            image: PIL Image or path
            
        Returns:
            (image_type, confidence, all_probs)
        """
        return self.primary_classifier.predict(image, return_probs=True)
    
    def segment_retina(self, image, threshold=0.5):
        """
        Segment blood vessels in retina image.
        
        Args:
            image: PIL Image or path
            threshold: Segmentation threshold
            
        Returns:
            (mask, overlay_image)
        """
        return self.segmentation_model.predict_with_overlay(
            image, threshold=threshold, alpha=0.4
        )
    
    def detect_blood_cells(self, image):
        """
        Detect blood cells in microscopy image.
        
        Args:
            image: PIL Image or path
            
        Returns:
            (detections, counts, annotated_image)
        """
        detections, counts = self.detection_model.predict_with_counts(image)
        annotated = self.detection_model.visualize(image, detections)
        
        return detections, counts, annotated
    
    def classify_scene(self, image, top_k=3):
        """
        Classify general scene image.
        
        Args:
            image: PIL Image or path
            top_k: Number of top predictions
            
        Returns:
            (predicted_class, confidence, top_predictions)
        """
        return self.cifar10_classifier.predict(
            image, return_probs=True, top_k=top_k
        )
    
    def get_models_info(self):
        """Get information about all loaded models."""
        return {
            'primary': self.primary_classifier.get_model_info(),
            'segmentation': self.segmentation_model.get_model_info(),
            'detection': self.detection_model.get_model_info(),
            'cifar10': self.cifar10_classifier.get_model_info()
        }


def get_models_manager() -> ModelsManager:
    """Get singleton instance of ModelsManager."""
    return ModelsManager()
