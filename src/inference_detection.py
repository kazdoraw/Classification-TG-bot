"""
YOLOv8 Blood Cell Detection Inference Module.

Detects and classifies blood cells:
- WBC (White Blood Cells)
- RBC (Red Blood Cells)
- Platelets
"""

import torch
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
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

logger = setup_logger(__name__)


class BloodCellDetector:
    """
    YOLOv8 model for blood cell detection and classification.
    
    Detects three types of blood cells: WBC, RBC, Platelets.
    """
    
    def __init__(self, model_path=None, device=None, conf_threshold=0.25, iou_threshold=0.45):
        """
        Initialize blood cell detector.
        
        Args:
            model_path: Path to trained YOLO model
            device: Device to run inference on ('cpu', 'cuda', or 'mps')
            conf_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.device = device or DEVICE
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        
        # Default model path
        if model_path is None:
            model_path = project_root / "models" / "yolov8_bccd" / "weights" / "best.pt"
        
        self.model_path = Path(model_path)
        
        # Class names (must match training)
        self.class_names = ['WBC', 'RBC', 'Platelets']
        self.num_classes = len(self.class_names)
        
        # Colors for visualization (BGR format for OpenCV)
        self.colors = {
            'WBC': (255, 0, 0),      # Blue
            'RBC': (0, 0, 255),      # Red
            'Platelets': (0, 255, 0) # Green
        }
        
        # Load model
        self.model = self._load_model()
        
        logger.info(f"YOLOv8 blood cell detector loaded from: {self.model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Classes: {self.class_names}")
        logger.info(f"Confidence threshold: {self.conf_threshold}")
    
    def _load_model(self):
        """Load trained YOLO model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {self.model_path}")
        
        # Load YOLO model
        model = YOLO(str(self.model_path))
        
        return model
    
    def predict(self, image, conf_threshold=None, iou_threshold=None):
        """
        Detect blood cells in image.
        
        Args:
            image: PIL Image, path to image, or numpy array
            conf_threshold: Override default confidence threshold
            iou_threshold: Override default IoU threshold
            
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - class: Class name
                - confidence: Detection confidence
        """
        conf = conf_threshold or self.conf_threshold
        iou = iou_threshold or self.iou_threshold
        
        # Run inference
        results = self.model(
            image,
            conf=conf,
            iou=iou,
            device=self.device,
            verbose=False
        )
        
        # Parse results
        detections = []
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Extract box data
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                class_name = self.class_names[class_id]
                
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'class': class_name,
                    'confidence': confidence,
                    'class_id': class_id
                })
        
        return detections
    
    def predict_with_counts(self, image):
        """
        Detect cells and return counts by type.
        
        Args:
            image: Input image
            
        Returns:
            (detections, counts) - detections list and count dictionary
        """
        detections = self.predict(image)
        
        # Count by class
        counts = {class_name: 0 for class_name in self.class_names}
        for det in detections:
            counts[det['class']] += 1
        
        return detections, counts
    
    def visualize(self, image, detections, show_conf=True, thickness=2):
        """
        Draw detections on image.
        
        Args:
            image: PIL Image or numpy array
            detections: List of detections from predict()
            show_conf: Show confidence scores
            thickness: Bounding box line thickness
            
        Returns:
            Annotated image as numpy array
        """
        # Convert to numpy if PIL
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image.copy()
        
        # Draw each detection
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class']
            confidence = det['confidence']
            
            # Get color
            color = self.colors[class_name]
            
            # Draw bounding box
            cv2.rectangle(
                img_array,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                color,
                thickness
            )
            
            # Prepare label
            if show_conf:
                label = f"{class_name}: {confidence:.2f}"
            else:
                label = class_name
            
            # Draw label background
            (text_width, text_height), _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                1
            )
            
            cv2.rectangle(
                img_array,
                (int(x1), int(y1) - text_height - 4),
                (int(x1) + text_width, int(y1)),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                img_array,
                label,
                (int(x1), int(y1) - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        return img_array
    
    def create_detection_report(self, image, save_path=None):
        """
        Create comprehensive detection visualization and report.
        
        Args:
            image: Input image
            save_path: Optional path to save visualization
            
        Returns:
            (figure, detections, counts)
        """
        # Detect
        detections, counts = self.predict_with_counts(image)
        
        # Load original image
        if isinstance(image, (str, Path)):
            original = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            original = Image.fromarray(image)
        else:
            original = image
        
        # Create annotated image
        annotated = self.visualize(original, detections)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Original image
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Annotated image with counts
        axes[1].imshow(annotated)
        title = f'Detections: {len(detections)} total\n'
        title += ', '.join([f"{cls}: {count}" for cls, count in counts.items()])
        axes[1].set_title(title)
        axes[1].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Detection report saved to: {save_path}")
        
        return fig, detections, counts
    
    def get_statistics(self, detections):
        """
        Calculate detection statistics.
        
        Args:
            detections: List of detections
            
        Returns:
            Dictionary with statistics
        """
        if not detections:
            return {
                'total': 0,
                'by_class': {cls: 0 for cls in self.class_names},
                'avg_confidence': 0.0,
                'confidence_by_class': {cls: 0.0 for cls in self.class_names}
            }
        
        # Count by class
        counts = {cls: 0 for cls in self.class_names}
        confidences = {cls: [] for cls in self.class_names}
        
        for det in detections:
            class_name = det['class']
            counts[class_name] += 1
            confidences[class_name].append(det['confidence'])
        
        # Average confidences
        avg_conf_by_class = {}
        for cls in self.class_names:
            if confidences[cls]:
                avg_conf_by_class[cls] = np.mean(confidences[cls])
            else:
                avg_conf_by_class[cls] = 0.0
        
        # Overall average
        all_confidences = [det['confidence'] for det in detections]
        avg_confidence = np.mean(all_confidences) if all_confidences else 0.0
        
        return {
            'total': len(detections),
            'by_class': counts,
            'avg_confidence': avg_confidence,
            'confidence_by_class': avg_conf_by_class
        }
    
    def get_model_info(self):
        """Get model information."""
        info = {
            'model_path': str(self.model_path),
            'architecture': 'YOLOv8n',
            'classes': self.class_names,
            'num_classes': self.num_classes,
            'device': str(self.device),
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold
        }
        
        return info


def main():
    """Test blood cell detector."""
    logger.info("="*60)
    logger.info("Testing Blood Cell Detector")
    logger.info("="*60)
    
    # Initialize detector
    detector = BloodCellDetector()
    
    # Display model info
    info = detector.get_model_info()
    logger.info("\nModel Information:")
    for key, value in info.items():
        logger.info(f"  {key}: {value}")
    
    # Test with sample images
    test_data_dir = project_root / "data" / "bccd_yolo" / "val" / "images"
    
    if test_data_dir.exists():
        images = list(test_data_dir.glob("*.jpg"))
        
        if images:
            logger.info(f"\nTesting on sample image: {images[0].name}")
            
            # Detect
            detections, counts = detector.predict_with_counts(images[0])
            
            logger.info(f"Total detections: {len(detections)}")
            logger.info("Counts by class:")
            for class_name, count in counts.items():
                logger.info(f"  {class_name}: {count}")
            
            # Statistics
            stats = detector.get_statistics(detections)
            logger.info(f"Average confidence: {stats['avg_confidence']:.4f}")
            
            # Create visualization
            save_path = project_root / "tmp" / "test_detection.png"
            save_path.parent.mkdir(exist_ok=True)
            detector.create_detection_report(images[0], save_path)
    
    logger.info("\n" + "="*60)
    logger.info("✓ Blood cell detector ready for inference!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
