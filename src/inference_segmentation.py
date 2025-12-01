"""
U-Net Retina Segmentation Inference Module.

Segments blood vessels in retinal fundus images using U-Net.
"""

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import DEVICE
from src.utils import setup_logger
from src.models.unet import UNet

logger = setup_logger(__name__)


class RetinaSegmentationModel:
    """
    U-Net model for retinal vessel segmentation.
    
    Predicts binary mask of blood vessels in retinal images.
    """
    
    def __init__(self, model_path=None, device=None):
        """
        Initialize segmentation model.
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on
        """
        self.device = device or DEVICE
        self.img_size = 256  # Must match training
        
        # Default model path
        if model_path is None:
            model_path = project_root / "models" / "unet_retina" / "best_model.pth"
        
        self.model_path = Path(model_path)
        
        # Image transforms (must match training)
        self.transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"U-Net segmentation model loaded from: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Image size: {self.img_size}x{self.img_size}")
    
    def _load_model(self):
        """Load trained U-Net model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load checkpoint (weights_only=False for compatibility)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        # Create model (UNet uses n_channels and n_classes parameters)
        model = UNet(n_channels=3, n_classes=1)
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
        
        # Store original size for resizing mask later
        self.original_size = image.size  # (width, height)
        
        # Apply transforms
        tensor = self.transform(image)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        return tensor
    
    def postprocess(self, mask_tensor, threshold=0.5, return_original_size=True):
        """
        Postprocess predicted mask.
        
        Args:
            mask_tensor: Model output tensor
            threshold: Binarization threshold
            return_original_size: If True, resize to original image size
            
        Returns:
            Binary mask as numpy array
        """
        # Remove batch dimension and convert to numpy
        mask = mask_tensor.squeeze().cpu().numpy()
        
        # Apply sigmoid and threshold
        mask = 1 / (1 + np.exp(-mask))  # Sigmoid
        mask = (mask > threshold).astype(np.uint8)
        
        # Resize to original size if requested
        if return_original_size and hasattr(self, 'original_size'):
            mask = cv2.resize(
                mask,
                self.original_size,
                interpolation=cv2.INTER_NEAREST
            )
        
        return mask
    
    def predict(self, image, threshold=0.5, return_original_size=True):
        """
        Predict vessel segmentation mask.
        
        Args:
            image: PIL Image, path to image, or numpy array
            threshold: Binarization threshold (default: 0.5)
            return_original_size: If True, return mask at original image size
            
        Returns:
            Binary mask as numpy array (0 or 1)
        """
        # Preprocess
        tensor = self.preprocess(image)
        tensor = tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            output = self.model(tensor)
        
        # Postprocess
        mask = self.postprocess(output, threshold, return_original_size)
        
        return mask
    
    def predict_with_overlay(self, image, threshold=0.5, alpha=0.5):
        """
        Predict and create overlay visualization.
        
        Args:
            image: PIL Image, path to image, or numpy array
            threshold: Binarization threshold
            alpha: Overlay transparency (0-1)
            
        Returns:
            (mask, overlay_image) - mask and RGB overlay visualization
        """
        # Load original image if path
        if isinstance(image, (str, Path)):
            original_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            original_image = Image.fromarray(image)
        else:
            original_image = image.copy()
        
        # Predict mask
        mask = self.predict(image, threshold, return_original_size=True)
        
        # Create overlay
        img_array = np.array(original_image)
        
        # Create red overlay for vessels
        overlay = img_array.copy()
        overlay[mask == 1] = [255, 0, 0]  # Red color for vessels
        
        # Blend with original
        overlay_image = cv2.addWeighted(img_array, 1-alpha, overlay, alpha, 0)
        overlay_image = Image.fromarray(overlay_image)
        
        return mask, overlay_image
    
    def calculate_metrics(self, predicted_mask, ground_truth_mask):
        """
        Calculate segmentation metrics.
        
        Args:
            predicted_mask: Predicted binary mask
            ground_truth_mask: Ground truth binary mask
            
        Returns:
            Dictionary with metrics (Dice, IoU, Pixel Accuracy)
        """
        pred = predicted_mask.flatten()
        gt = ground_truth_mask.flatten()
        
        # Dice coefficient
        intersection = np.sum(pred * gt)
        dice = (2. * intersection) / (np.sum(pred) + np.sum(gt) + 1e-8)
        
        # IoU (Jaccard)
        union = np.sum(pred) + np.sum(gt) - intersection
        iou = intersection / (union + 1e-8)
        
        # Pixel accuracy
        pixel_acc = np.sum(pred == gt) / len(pred)
        
        return {
            'dice': dice,
            'iou': iou,
            'pixel_accuracy': pixel_acc
        }
    
    def visualize_prediction(self, image, save_path=None):
        """
        Create comprehensive visualization of segmentation.
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
            
        Returns:
            Figure object
        """
        # Predict
        mask, overlay = self.predict_with_overlay(image, alpha=0.4)
        
        # Load original image
        if isinstance(image, (str, Path)):
            original = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            original = Image.fromarray(image)
        else:
            original = image
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Predicted Vessel Mask')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay (Red = Vessels)')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to: {save_path}")
        
        return fig
    
    def get_model_info(self):
        """Get model information."""
        checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
        
        info = {
            'model_path': str(self.model_path),
            'architecture': 'U-Net',
            'input_size': f'{self.img_size}x{self.img_size}',
            'device': str(self.device),
            'best_dice': checkpoint.get('best_dice', 'N/A'),
            'epoch': checkpoint.get('epoch', 'N/A')
        }
        
        return info


def main():
    """Test segmentation model."""
    logger.info("="*60)
    logger.info("Testing Retina Segmentation Model")
    logger.info("="*60)
    
    # Initialize model
    model = RetinaSegmentationModel()
    
    # Display model info
    info = model.get_model_info()
    logger.info("\nModel Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Test with sample images
    test_data_dir = project_root / "data" / "DRIVE" / "training" / "images"
    
    if test_data_dir.exists():
        images = list(test_data_dir.glob("*.tif"))
        
        if images:
            logger.info(f"\nTesting on sample image: {images[0].name}")
            
            # Predict
            mask = model.predict(images[0])
            logger.info(f"Mask shape: {mask.shape}")
            logger.info(f"Vessel pixels: {np.sum(mask)} ({100*np.sum(mask)/mask.size:.2f}%)")
            
            # Create visualization
            save_path = project_root / "tmp" / "test_segmentation.png"
            save_path.parent.mkdir(exist_ok=True)
            model.visualize_prediction(images[0], save_path)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Segmentation model ready for inference!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
