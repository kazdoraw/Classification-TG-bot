"""
Bot Handlers.

Обработчики команд и фото для Telegram бота.
"""

import logging
from PIL import Image
import numpy as np
from aiogram import Router, F
from aiogram.filters import CommandStart, Command
from aiogram.types import Message, BufferedInputFile
from aiogram.enums import ParseMode

from bot.config import MESSAGES, MAX_FILE_SIZE_MB
from bot.utils import (
    download_photo,
    image_to_bytes,
    create_input_file,
    format_confidence,
    format_detection_stats,
    format_top_predictions
)
from bot.models_loader import get_models_manager

logger = logging.getLogger(__name__)

# Create router
router = Router()

# Get models manager (singleton)
models = get_models_manager()


@router.message(CommandStart())
async def cmd_start(message: Message):
    """Handle /start command."""
    await message.answer(
        MESSAGES['start'],
        parse_mode=ParseMode.MARKDOWN
    )
    logger.info(f"User {message.from_user.id} started the bot")


@router.message(Command("help"))
async def cmd_help(message: Message):
    """Handle /help command."""
    await message.answer(
        MESSAGES['help'],
        parse_mode=ParseMode.MARKDOWN
    )
    logger.info(f"User {message.from_user.id} requested help")


@router.message(F.photo)
async def handle_photo(message: Message):
    """
    Handle photo messages.
    
    Workflow:
    1. Download photo
    2. Classify image type (retina/blood/scene)
    3. Route to appropriate model
    4. Return results
    """
    user_id = message.from_user.id
    logger.info(f"User {user_id} sent a photo")
    
    try:
        # Check file size
        photo = message.photo[-1]
        file_size_mb = photo.file_size / (1024 * 1024)
        
        if file_size_mb > MAX_FILE_SIZE_MB:
            await message.answer(MESSAGES['error_too_large'])
            return
        
        # Send processing message
        status_msg = await message.answer(MESSAGES['processing'])
        
        # Download photo
        image = await download_photo(message.bot, message)
        logger.info(f"Downloaded image: {image.size}")
        
        # Update status
        await status_msg.edit_text(MESSAGES['classifying'])
        
        # Classify image type
        image_type, confidence, probs = models.classify_image_type(image)
        logger.info(f"Image classified as: {image_type} ({confidence:.4f})")
        
        # Update status
        await status_msg.edit_text(MESSAGES['analyzing'])
        
        # Route to appropriate model
        if image_type == 'retina':
            await process_retina(message, image, status_msg)
        elif image_type == 'blood':
            await process_blood(message, image, status_msg)
        elif image_type == 'scene':
            await process_scene(message, image, status_msg)
        else:
            await message.answer(MESSAGES['error_processing'])
        
        # Delete status message
        await status_msg.delete()
        
    except Exception as e:
        logger.error(f"Error processing photo: {e}", exc_info=True)
        await message.answer(MESSAGES['error_processing'])


async def process_retina(message: Message, image: Image.Image, status_msg: Message):
    """
    Process retina fundus image.
    
    Performs vessel segmentation and returns overlay visualization.
    """
    logger.info("Processing retina image")
    
    try:
        # Segment vessels
        mask, overlay = models.segment_retina(image, threshold=0.5)
        
        # Convert overlay to bytes
        overlay_bytes = image_to_bytes(overlay, format='PNG')
        
        # Calculate vessel percentage
        vessel_percentage = (np.sum(mask) / mask.size) * 100
        
        # Prepare caption
        caption = f"""
🔬 **Сегментация сетчатки**

✅ Обработано успешно!

📊 **Результаты:**
• Сосуды выделены красным цветом
• Площадь сосудов: {vessel_percentage:.2f}%

Метод: U-Net (Dice: 0.51)
"""
        
        # Send result
        await message.answer_photo(
            photo=create_input_file(overlay_bytes, "retina_segmentation.png"),
            caption=caption,
            parse_mode=ParseMode.MARKDOWN
        )
        
        logger.info("Retina processing complete")
        
    except Exception as e:
        logger.error(f"Error in retina processing: {e}", exc_info=True)
        raise


async def process_blood(message: Message, image: Image.Image, status_msg: Message):
    """
    Process blood cell microscopy image.
    
    Performs cell detection and returns annotated image with counts.
    """
    logger.info("Processing blood cell image")
    
    try:
        # Detect cells
        detections, counts, annotated = models.detect_blood_cells(image)
        
        # Convert annotated image to bytes
        annotated_pil = Image.fromarray(annotated)
        annotated_bytes = image_to_bytes(annotated_pil, format='PNG')
        
        # Prepare caption
        total_cells = sum(counts.values())
        caption = f"""
🩸 **Детекция клеток крови**

✅ Обнаружено: {len(detections)} объектов

📊 **Подсчёт клеток:**
{format_detection_stats(counts)}

**Легенда:**
🔵 WBC - Белые кровяные тельца
🔴 RBC - Красные кровяные тельца  
🟢 Platelets - Тромбоциты

Модель: YOLOv8n (mAP50: 0.935)
"""
        
        # Send result
        await message.answer_photo(
            photo=create_input_file(annotated_bytes, "blood_detection.png"),
            caption=caption,
            parse_mode=ParseMode.MARKDOWN
        )
        
        logger.info(f"Blood processing complete: {counts}")
        
    except Exception as e:
        logger.error(f"Error in blood processing: {e}", exc_info=True)
        raise


async def process_scene(message: Message, image: Image.Image, status_msg: Message):
    """
    Process general scene image.
    
    Classifies into CIFAR-10 categories.
    """
    logger.info("Processing scene image")
    
    try:
        # Classify scene
        pred_class, confidence, top_predictions = models.classify_scene(image, top_k=3)
        
        # Prepare caption
        caption = f"""
🌅 **Классификация сцены**

✅ **Результат:** {pred_class.upper()}
Уверенность: {format_confidence(confidence)}

📊 **Топ-3 предсказания:**
{format_top_predictions(top_predictions, top_k=3)}

Модель: ResNet-18 (Accuracy: 85%+)
"""
        
        # Send result (just text, no image modification for scene)
        await message.answer(
            caption,
            parse_mode=ParseMode.MARKDOWN
        )
        
        logger.info(f"Scene processing complete: {pred_class}")
        
    except Exception as e:
        logger.error(f"Error in scene processing: {e}", exc_info=True)
        raise


@router.message(F.text)
async def handle_text(message: Message):
    """Handle any text message (fallback)."""
    await message.answer(
        "📸 Пожалуйста, отправьте фото для анализа.\n\n"
        "Используйте /help для подробной информации.",
        parse_mode=ParseMode.MARKDOWN
    )


@router.message()
async def handle_other(message: Message):
    """Handle any other message type."""
    await message.answer(MESSAGES['error_no_photo'])
