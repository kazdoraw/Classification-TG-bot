"""
Utility functions for the project.
"""

import logging
import sys
from pathlib import Path
from PIL import Image
import torch
from torchvision import transforms


def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Setup logger with console handler.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger


def load_and_preprocess_image(
    image_path: Path | str,
    target_size: int = 224,
    normalize: bool = True
) -> torch.Tensor:
    """
    Load and preprocess image for model inference.
    
    Args:
        image_path: Path to image
        target_size: Target size for resize
        normalize: Whether to apply ImageNet normalization
        
    Returns:
        Preprocessed image tensor [1, 3, H, W]
    """
    image = Image.open(image_path).convert('RGB')
    
    transform_list = [
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
    ]
    
    if normalize:
        transform_list.append(
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        )
    
    transform = transforms.Compose(transform_list)
    image_tensor = transform(image).unsqueeze(0)
    
    return image_tensor


def resize_image_if_needed(
    image_path: Path | str,
    max_size: int = 2048
) -> Path:
    """
    Resize image if it's too large.
    
    Args:
        image_path: Path to image
        max_size: Maximum dimension size
        
    Returns:
        Path to resized image (or original if no resize needed)
    """
    image_path = Path(image_path)
    image = Image.open(image_path)
    
    # Check if resize needed
    if max(image.size) <= max_size:
        return image_path
    
    # Calculate new size maintaining aspect ratio
    ratio = max_size / max(image.size)
    new_size = tuple(int(dim * ratio) for dim in image.size)
    
    # Resize and save
    resized_image = image.resize(new_size, Image.Resampling.LANCZOS)
    resized_path = image_path.parent / f"resized_{image_path.name}"
    resized_image.save(resized_path)
    
    return resized_path


def validate_image_file(file_path: Path | str) -> bool:
    """
    Validate if file is a valid image.
    
    Args:
        file_path: Path to file
        
    Returns:
        True if valid image, False otherwise
    """
    file_path = Path(file_path)
    
    # Check extension
    valid_extensions = {'.jpg', '.jpeg', '.png'}
    if file_path.suffix.lower() not in valid_extensions:
        return False
    
    # Check if file exists
    if not file_path.exists():
        return False
    
    # Try to open as image
    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False


def cleanup_temp_file(file_path: Path | str):
    """
    Remove temporary file if it exists.
    
    Args:
        file_path: Path to file
    """
    file_path = Path(file_path)
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception as e:
            logging.warning(f"Failed to remove temp file {file_path}: {e}")
