"""
Primary Classification Inference Module.

Classifies images into 3 categories:
- retina: Retinal fundus images
- blood: Blood cell microscopy images
- scene: General scene images (CIFAR-10 like)

This module routes images to appropriate auxiliary models.
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import DEVICE
from src.utils import setup_logger

logger = setup_logger(__name__)


class PrimaryClassifier:
    """
    Primary classifier for routing images.
    
    Uses the best-performing model from Stage 3 training
    (ResNet-18, Baseline CNN, or ViT).
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize primary classifier.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device or DEVICE
        self.class_names = ['retina', 'blood', 'scene']
        self.num_classes = len(self.class_names)
        
        # Default model path (ResNet-18 - 100% accuracy)
        if model_path is None:
            model_path = project_root / "models" / "resnet18_primary" / "best_model.pth"
        
        self.model_path = Path(model_path)
        
        # Image transforms (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"Primary classifier loaded from: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Classes: {self.class_names}")
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint (weights_only=False for compatibility with saved checkpoints)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Create model architecture (ResNet-18)
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, self.num_classes)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess(self, image):
        """
        Preprocess image for inference.
        
        Args:
            image: PIL Image or path to image file
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image, (str, Path)):
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif not isinstance(image, Image.Image):
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def predict(self, image, return_probs=False):
        """
        Predict class for a single image.
        
        Args:
            image: PIL Image, path to image, or numpy array
            return_probs: If True, return class probabilities
            
        Returns:
            If return_probs=False: (predicted_class, confidence)
            If return_probs=True: (predicted_class, confidence, all_probabilities)
        """
        # Preprocess
        tensor = self.preprocess(image)
        tensor = tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, predicted = probs.max(1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_value = confidence.item()
        
        if return_probs:
            all_probs = {
                self.class_names[i]: probs[0, i].item()
                for i in range(self.num_classes)
            }
            return predicted_class, confidence_value, all_probs
        
        return predicted_class, confidence_value
    
    def predict_batch(self, images, batch_size=32):
        """
        Predict classes for multiple images.
        
        Args:
            images: List of images (PIL, paths, or numpy arrays)
            batch_size: Batch size for inference
            
        Returns:
            List of (predicted_class, confidence) tuples
        """
        results = []
        
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            
            # Preprocess batch
            batch_tensors = []
            for img in batch_images:
                tensor = self.preprocess(img)
                batch_tensors.append(tensor)
            
            batch_tensor = torch.cat(batch_tensors, dim=0)
            batch_tensor = batch_tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.model(batch_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidences, predicted = probs.max(1)
            
            # Collect results
            for pred_idx, conf in zip(predicted, confidences):
                predicted_class = self.class_names[pred_idx.item()]
                confidence_value = conf.item()
                results.append((predicted_class, confidence_value))
        
        return results
    
    def get_model_info(self):
        """Get model information."""
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        info = {
            'model_path': str(self.model_path),
            'classes': self.class_names,
            'num_classes': self.num_classes,
            'device': str(self.device),
            'accuracy': checkpoint.get('val_acc', 'N/A')
        }
        
        if 'epoch' in checkpoint:
            info['trained_epochs'] = checkpoint['epoch']
        
        return info


def main():
    """Test primary classifier."""
    logger.info("="*60)
    logger.info("Testing Primary Classifier")
    logger.info("="*60)
    
    # Initialize classifier
    classifier = PrimaryClassifier()
    
    # Display model info
    info = classifier.get_model_info()
    logger.info("\nModel Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Test with sample images
    test_data_dir = project_root / "data" / "primary"
    
    if test_data_dir.exists():
        logger.info("\nTesting on sample images...")
        
        for class_name in classifier.class_names:
            class_dir = test_data_dir / "val" / class_name
            
            if class_dir.exists():
                images = list(class_dir.glob("*.jpg")) + list(class_dir.glob("*.png"))
                
                if images:
                    # Test first image from each class
                    test_image = images[0]
                    pred_class, confidence, probs = classifier.predict(
                        test_image, return_probs=True
                    )
                    
                    logger.info(f"\nTrue class: {class_name}")
                    logger.info(f"Predicted: {pred_class} (confidence: {confidence:.4f})")
                    logger.info("All probabilities:")
                    for cls, prob in probs.items():
                        logger.info(f"  {cls}: {prob:.4f}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ Primary classifier ready for inference!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
