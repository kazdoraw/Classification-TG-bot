"""
CIFAR-10 Scene Classification Inference Module.

Classifies general scene images into 10 categories:
- airplane, automobile, bird, cat, deer
- dog, frog, horse, ship, truck
"""

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import DEVICE
from src.utils import setup_logger

logger = setup_logger(__name__)


class CIFAR10Classifier:
    """
    ResNet-18 model for CIFAR-10 scene classification.
    
    Classifies images into 10 natural scene categories.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize CIFAR-10 classifier.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device or DEVICE
        
        # CIFAR-10 class names
        self.class_names = [
            'airplane', 'automobile', 'bird', 'cat', 'deer',
            'dog', 'frog', 'horse', 'ship', 'truck'
        ]
        self.num_classes = len(self.class_names)
        
        # Default model path
        if model_path is None:
            model_path = project_root / "models" / "resnet18_cifar10" / "best_model.pth"
        
        self.model_path = Path(model_path)
        
        # Image transforms (must match training - no augmentation for inference)
        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),  # CIFAR-10 native size
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 statistics
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"CIFAR-10 classifier loaded from: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Classes: {self.num_classes}")
    
    def _load_model(self):
        """Load trained ResNet-18 model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint (weights_only=False for compatibility)
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
    
    def predict(self, image, return_probs=False, top_k=3):
        """
        Predict class for a single image.
        
        Args:
            image: PIL Image, path to image, or numpy array
            return_probs: If True, return class probabilities
            top_k: Number of top predictions to return
            
        Returns:
            If return_probs=False: (predicted_class, confidence)
            If return_probs=True: (predicted_class, confidence, top_k_predictions)
        """
        # Preprocess
        tensor = self.preprocess(image)
        tensor = tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Top prediction
            confidence, predicted = probs.max(1)
            predicted_class = self.class_names[predicted.item()]
            confidence_value = confidence.item()
            
            if return_probs:
                # Get top-k predictions
                top_probs, top_indices = probs.topk(min(top_k, self.num_classes), dim=1)
                
                top_predictions = []
                for i in range(top_probs.size(1)):
                    class_idx = top_indices[0, i].item()
                    class_prob = top_probs[0, i].item()
                    top_predictions.append({
                        'class': self.class_names[class_idx],
                        'probability': class_prob
                    })
                
                return predicted_class, confidence_value, top_predictions
        
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
    
    def visualize_prediction(self, image, save_path=None):
        """
        Create visualization of prediction with top-k classes.
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
            
        Returns:
            Figure object
        """
        # Predict
        pred_class, confidence, top_predictions = self.predict(
            image, return_probs=True, top_k=5
        )
        
        # Load original image
        if isinstance(image, (str, Path)):
            original = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            original = Image.fromarray(image)
        else:
            original = image
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        axes[0].imshow(original)
        axes[0].set_title(f'Prediction: {pred_class}\nConfidence: {confidence:.4f}')
        axes[0].axis('off')
        
        # Show top-k predictions as bar chart
        classes = [p['class'] for p in top_predictions]
        probs = [p['probability'] for p in top_predictions]
        
        y_pos = np.arange(len(classes))
        axes[1].barh(y_pos, probs, color='skyblue')
        axes[1].set_yticks(y_pos)
        axes[1].set_yticklabels(classes)
        axes[1].invert_yaxis()
        axes[1].set_xlabel('Probability')
        axes[1].set_title('Top 5 Predictions')
        axes[1].set_xlim([0, 1])
        
        # Add probability values on bars
        for i, prob in enumerate(probs):
            axes[1].text(prob + 0.01, i, f'{prob:.3f}', va='center')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        return fig
    
    def get_class_distribution(self, predictions):
        """
        Calculate class distribution from multiple predictions.
        
        Args:
            predictions: List of predicted class names
            
        Returns:
            Dictionary with class counts and percentages
        """
        counts = {cls: 0 for cls in self.class_names}
        
        for pred in predictions:
            if pred in counts:
                counts[pred] += 1
        
        total = len(predictions)
        distribution = {}
        
        for cls, count in counts.items():
            distribution[cls] = {
                'count': count,
                'percentage': (count / total * 100) if total > 0 else 0
            }
        
        return distribution
    
    def get_model_info(self):
        """Get model information."""
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        info = {
            'model_path': str(self.model_path),
            'architecture': 'ResNet-18',
            'classes': self.class_names,
            'num_classes': self.num_classes,
            'device': str(self.device),
            'val_accuracy': checkpoint.get('val_acc', 'N/A'),
            'val_f1': checkpoint.get('val_f1', 'N/A'),
            'epoch': checkpoint.get('epoch', 'N/A')
        }
        
        return info


def main():
    """Test CIFAR-10 classifier."""
    logger.info("="*60)
    logger.info("Testing CIFAR-10 Classifier")
    logger.info("="*60)
    
    # Initialize classifier
    classifier = CIFAR10Classifier()
    
    # Display model info
    info = classifier.get_model_info()
    logger.info("\nModel Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Test with sample images from CIFAR-10
    test_data_dir = project_root / "data" / "cifar10" / "cifar-10-batches-py"
    
    # Since CIFAR-10 is in pickle format, we'll just test with model info
    logger.info("\nCIFAR-10 test images are in pickle format.")
    logger.info("Use torchvision.datasets.CIFAR10 for testing.")
    
    # Example of how to use with torchvision dataset
    try:
        from torchvision import datasets
        
        test_dataset = datasets.CIFAR10(
            root=str(project_root / "data" / "cifar10"),
            train=False,
            download=False
        )
        
        if len(test_dataset) > 0:
            # Test on first image
            test_image, true_label = test_dataset[0]
            
            pred_class, confidence, top_preds = classifier.predict(
                test_image, return_probs=True
            )
            
            logger.info(f"\nTest prediction:")
            logger.info(f"True label: {classifier.class_names[true_label]}")
            logger.info(f"Predicted: {pred_class} (confidence: {confidence:.4f})")
            logger.info("\nTop 3 predictions:")
            for i, pred in enumerate(top_preds[:3], 1):
                logger.info(f"  {i}. {pred['class']}: {pred['probability']:.4f}")
            
            # Create visualization
            save_path = project_root / "tmp" / "test_cifar10.png"
            save_path.parent.mkdir(exist_ok=True)
            classifier.visualize_prediction(test_image, save_path)
    
    except Exception as e:
        logger.warning(f"Could not test with CIFAR-10 dataset: {e}")
    
    logger.info("\n" + "="*60)
    logger.info("✓ CIFAR-10 classifier ready for inference!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
