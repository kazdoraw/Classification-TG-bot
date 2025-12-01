"""
Train YOLOv8 for Blood Cell Detection.

Dataset: BCCD (Blood Cell Count and Detection)
Task: Object detection of blood cells
Classes: WBC (White Blood Cells), RBC (Red Blood Cells), Platelets

Model: YOLOv8n (nano - fast and lightweight)
Training: Fine-tuning from COCO pretrained weights
"""

from pathlib import Path
from ultralytics import YOLO
import yaml
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import DEVICE
from src.utils import setup_logger

# Configuration - OPTIMIZED FOR FAST TRAINING
CONFIG = {
    'model': 'yolov8n.pt',  # Nano model (fastest)
    'epochs': 20,  # Reduced from 50 for speed (4x faster)
    'imgsz': 416,  # Reduced from 640 (2.4x faster, good for small objects)
    'batch': 16,  # Keep same
    'device': 'cpu',  # CPU training (use 0 for GPU)
    'patience': 5,  # Reduced from 10 (faster early stopping)
    'lr0': 0.01,  # Initial learning rate
    'optimizer': 'SGD',
    'augment': True,  # Data augmentation
    'project': str(project_root / 'models'),
    'name': 'yolov8_bccd',
    'exist_ok': True,
    'pretrained': True,
    'verbose': True,
    'plots': True,
    'save': True,
}

logger = setup_logger(__name__)


def check_dataset(yaml_path):
    """Verify dataset configuration and paths."""
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    logger.info("Dataset configuration:")
    logger.info(f"  Path: {data['path']}")
    logger.info(f"  Train: {data['train']}")
    logger.info(f"  Val: {data['val']}")
    logger.info(f"  Classes: {data['names']}")
    
    # Check if paths exist
    dataset_path = Path(data['path'])
    train_path = dataset_path / data['train']
    val_path = dataset_path / data['val']
    
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Train path not found: {train_path}")
    if not val_path.exists():
        raise FileNotFoundError(f"Val path not found: {val_path}")
    
    # Count images
    train_images = list(train_path.glob("*.jpg"))
    val_images = list(val_path.glob("*.jpg"))
    
    logger.info(f"  Train images: {len(train_images)}")
    logger.info(f"  Val images: {len(val_images)}")
    
    if len(train_images) == 0:
        raise ValueError("No training images found!")
    if len(val_images) == 0:
        raise ValueError("No validation images found!")
    
    return data


def train_yolo():
    """Train YOLOv8 model."""
    logger.info("="*60)
    logger.info("Training YOLOv8 for Blood Cell Detection")
    logger.info("="*60)
    
    # Dataset path
    dataset_yaml = project_root / "data" / "bccd_yolo" / "dataset.yaml"
    
    if not dataset_yaml.exists():
        logger.error(f"Dataset YAML not found: {dataset_yaml}")
        logger.error("Run 'python src/prepare_bccd_yolo.py' first!")
        sys.exit(1)
    
    # Check dataset
    logger.info("\nVerifying dataset...")
    dataset_config = check_dataset(dataset_yaml)
    
    # Load pretrained model
    logger.info(f"\nLoading model: {CONFIG['model']}")
    model = YOLO(CONFIG['model'])
    
    logger.info(f"Model: YOLOv8n")
    logger.info(f"Pretrained: COCO weights")
    logger.info(f"Task: Object Detection")
    logger.info(f"Classes: {len(dataset_config['names'])}")
    
    # Training configuration
    logger.info("\nTraining configuration:")
    logger.info(f"  Epochs: {CONFIG['epochs']} (with early stopping, patience={CONFIG['patience']})")
    logger.info(f"  Image size: {CONFIG['imgsz']}x{CONFIG['imgsz']}")
    logger.info(f"  Batch size: {CONFIG['batch']}")
    logger.info(f"  Learning rate: {CONFIG['lr0']}")
    logger.info(f"  Optimizer: {CONFIG['optimizer']}")
    logger.info(f"  Device: {CONFIG['device']}")
    logger.info(f"  Data augmentation: {CONFIG['augment']}")
    
    # Expected training time
    logger.info(f"\nExpected training time: ~8-12 minutes on CPU (optimized)")
    logger.info("")
    
    # Start training
    logger.info("Starting training...")
    logger.info("-"*60)
    
    try:
        results = model.train(
            data=str(dataset_yaml),
            epochs=CONFIG['epochs'],
            imgsz=CONFIG['imgsz'],
            batch=CONFIG['batch'],
            device=CONFIG['device'],
            patience=CONFIG['patience'],
            lr0=CONFIG['lr0'],
            optimizer=CONFIG['optimizer'],
            augment=CONFIG['augment'],
            project=CONFIG['project'],
            name=CONFIG['name'],
            exist_ok=CONFIG['exist_ok'],
            pretrained=CONFIG['pretrained'],
            verbose=CONFIG['verbose'],
            plots=CONFIG['plots'],
            save=CONFIG['save'],
            # Minimal augmentation for speed (optimized)
            hsv_h=0.01,    # Reduced HSV-Hue
            hsv_s=0.5,     # Reduced HSV-Saturation
            hsv_v=0.3,     # Reduced HSV-Value
            degrees=10.0,  # Reduced rotation
            translate=0.05, # Reduced translation
            scale=0.3,     # Reduced scale
            flipud=0.0,    # No flip up-down
            fliplr=0.5,    # Keep horizontal flip (important)
            mosaic=0.5,    # Reduced mosaic (slower augmentation)
            mixup=0.0,     # No mixup
        )
        
        logger.info("-"*60)
        logger.info("✓ Training complete!")
        
        # Print metrics
        logger.info("\nFinal metrics:")
        logger.info(f"  mAP50-95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A'):.4f}")
        logger.info(f"  mAP50: {results.results_dict.get('metrics/mAP50(B)', 'N/A'):.4f}")
        logger.info(f"  Precision: {results.results_dict.get('metrics/precision(B)', 'N/A'):.4f}")
        logger.info(f"  Recall: {results.results_dict.get('metrics/recall(B)', 'N/A'):.4f}")
        
        # Model saved location
        model_dir = Path(CONFIG['project']) / CONFIG['name']
        best_model = model_dir / 'weights' / 'best.pt'
        last_model = model_dir / 'weights' / 'last.pt'
        
        logger.info(f"\nModel saved:")
        logger.info(f"  Best: {best_model}")
        logger.info(f"  Last: {last_model}")
        
        # Validate
        logger.info("\nValidating model...")
        metrics = model.val()
        
        logger.info("\nValidation results:")
        logger.info(f"  mAP50-95: {metrics.box.map:.4f}")
        logger.info(f"  mAP50: {metrics.box.map50:.4f}")
        logger.info(f"  mAP75: {metrics.box.map75:.4f}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ YOLOv8 training complete!")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"✗ Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    train_yolo()
