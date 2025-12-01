"""
Configuration loader for the project.
Loads settings from .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "./data")
MODELS_DIR = PROJECT_ROOT / os.getenv("MODELS_DIR", "./models")
TMP_DIR = PROJECT_ROOT / os.getenv("TMP_DIR", "./tmp")

# Model paths
PRIMARY_MODEL_PATH = PROJECT_ROOT / os.getenv(
    "PRIMARY_MODEL_PATH", "./models/primary_resnet18.pth"
)
SEGMENTATION_MODEL_PATH = PROJECT_ROOT / os.getenv(
    "SEGMENTATION_MODEL_PATH", "./models/retina_unet.pth"
)
DETECTION_MODEL_PATH = PROJECT_ROOT / os.getenv(
    "DETECTION_MODEL_PATH", "./models/blood_yolo.pt"
)
CIFAR10_MODEL_PATH = PROJECT_ROOT / os.getenv(
    "CIFAR10_MODEL_PATH", "./models/cifar10_classifier.pth"
)

# Telegram bot settings
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")

# Inference settings
DEVICE = os.getenv("DEVICE", "cpu")
MAX_IMAGE_SIZE = int(os.getenv("MAX_IMAGE_SIZE", "2048"))
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

# Data paths
PRIMARY_TRAIN_DIR = DATA_DIR / "primary" / "train"
PRIMARY_VAL_DIR = DATA_DIR / "primary" / "val"
RAW_DATA_DIR = DATA_DIR / "raw"

# Class names
PRIMARY_CLASSES = ["retina", "blood", "scene"]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]
BLOOD_CLASSES = ["RBC", "WBC", "Platelets"]


def validate_config():
    """Validate configuration and create necessary directories."""
    # Create directories if they don't exist
    for directory in [DATA_DIR, MODELS_DIR, TMP_DIR]:
        directory.mkdir(parents=True, exist_ok=True)
    
    # Create tmp subdirectories
    (TMP_DIR / "uploads").mkdir(exist_ok=True)
    (TMP_DIR / "segmentations").mkdir(exist_ok=True)
    (TMP_DIR / "detections").mkdir(exist_ok=True)
    
    # Check if bot token is set
    if not TELEGRAM_BOT_TOKEN or TELEGRAM_BOT_TOKEN == "your_bot_token_here":
        print("WARNING: TELEGRAM_BOT_TOKEN not set in .env")
    
    return True


if __name__ == "__main__":
    validate_config()
    print("Configuration loaded successfully")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"DATA_DIR: {DATA_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    print(f"DEVICE: {DEVICE}")
