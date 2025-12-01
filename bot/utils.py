"""
Utility functions for the bot.

Вспомогательные функции для загрузки изображений, обработки и форматирования.
"""

import io
import logging
from pathlib import Path
from PIL import Image
from aiogram.types import Message, BufferedInputFile
from aiogram import Bot

logger = logging.getLogger(__name__)


async def download_photo(bot: Bot, message: Message) -> Image.Image:
    """
    Download photo from Telegram message.
    
    Args:
        bot: Bot instance
        message: Message with photo
        
    Returns:
        PIL Image
    """
    # Get highest resolution photo
    photo = message.photo[-1]
    
    # Download file
    file = await bot.get_file(photo.file_id)
    file_bytes = await bot.download_file(file.file_path)
    
    # Convert to PIL Image
    image = Image.open(file_bytes).convert('RGB')
    
    logger.info(f"Downloaded photo: size={image.size}, format={image.format}")
    
    return image


def save_temp_image(image: Image.Image, filename: str, tmp_dir: Path) -> Path:
    """
    Save image to temporary directory.
    
    Args:
        image: PIL Image
        filename: Output filename
        tmp_dir: Temporary directory path
        
    Returns:
        Path to saved file
    """
    filepath = tmp_dir / filename
    image.save(filepath, format='PNG')
    
    logger.info(f"Saved temp image: {filepath}")
    
    return filepath


def image_to_bytes(image: Image.Image, format: str = 'PNG') -> bytes:
    """
    Convert PIL Image to bytes.
    
    Args:
        image: PIL Image
        format: Output format
        
    Returns:
        Image bytes
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    buffer.seek(0)
    return buffer.getvalue()


def create_input_file(image_bytes: bytes, filename: str) -> BufferedInputFile:
    """
    Create Telegram input file from bytes.
    
    Args:
        image_bytes: Image bytes
        filename: Filename for Telegram
        
    Returns:
        BufferedInputFile for sending
    """
    return BufferedInputFile(image_bytes, filename=filename)


def format_confidence(confidence: float) -> str:
    """Format confidence as percentage."""
    return f"{confidence * 100:.1f}%"


def format_detection_stats(counts: dict) -> str:
    """
    Format blood cell detection statistics.
    
    Args:
        counts: Dictionary with cell counts
        
    Returns:
        Formatted string
    """
    total = sum(counts.values())
    
    lines = [f"**Всего клеток:** {total}"]
    lines.append("")
    
    for cell_type, count in counts.items():
        percentage = (count / total * 100) if total > 0 else 0
        lines.append(f"• **{cell_type}:** {count} ({percentage:.1f}%)")
    
    return "\n".join(lines)


def format_top_predictions(predictions: list, top_k: int = 3) -> str:
    """
    Format top-k predictions.
    
    Args:
        predictions: List of {'class': str, 'probability': float}
        top_k: Number of predictions to show
        
    Returns:
        Formatted string
    """
    lines = []
    
    for i, pred in enumerate(predictions[:top_k], 1):
        class_name = pred['class']
        probability = pred['probability'] * 100
        
        # Add emoji for top prediction
        emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉"
        
        lines.append(f"{emoji} **{class_name}** - {probability:.1f}%")
    
    return "\n".join(lines)


def cleanup_temp_files(tmp_dir: Path, pattern: str = "*"):
    """
    Clean up temporary files.
    
    Args:
        tmp_dir: Temporary directory
        pattern: File pattern to match
    """
    try:
        for file in tmp_dir.glob(pattern):
            if file.is_file():
                file.unlink()
                logger.debug(f"Deleted temp file: {file}")
    except Exception as e:
        logger.error(f"Error cleaning up temp files: {e}")
