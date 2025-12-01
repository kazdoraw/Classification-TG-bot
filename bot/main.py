"""
Main Bot Entry Point.

Запускает Telegram бота с загрузкой всех моделей и обработчиков.
"""

import asyncio
import logging
import sys
from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode

from bot.config import BOT_TOKEN
from bot.handlers import router
from bot.models_loader import get_models_manager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log')
    ]
)

logger = logging.getLogger(__name__)


async def main():
    """Main bot function."""
    logger.info("="*60)
    logger.info("Starting Medical Image Analysis Bot")
    logger.info("="*60)
    
    try:
        # Initialize models
        logger.info("Loading inference models...")
        models_manager = get_models_manager()
        
        # Display models info
        models_info = models_manager.get_models_info()
        logger.info("\nLoaded models:")
        for model_name, info in models_info.items():
            logger.info(f"  • {model_name}: {info.get('architecture', 'N/A')}")
        
        # Initialize bot
        logger.info("\nInitializing bot...")
        bot = Bot(
            token=BOT_TOKEN,
            default=DefaultBotProperties(parse_mode=ParseMode.HTML)
        )
        
        # Initialize dispatcher
        dp = Dispatcher()
        
        # Register router
        dp.include_router(router)
        
        # Delete webhook if exists (for polling to work)
        await bot.delete_webhook(drop_pending_updates=True)
        logger.info("Webhook deleted (if existed)")
        
        # Get bot info
        bot_info = await bot.get_me()
        logger.info(f"Bot username: @{bot_info.username}")
        logger.info(f"Bot ID: {bot_info.id}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ Bot started successfully!")
        logger.info("="*60)
        logger.info("Press Ctrl+C to stop\n")
        
        # Start polling
        await dp.start_polling(bot)
        
    except KeyboardInterrupt:
        logger.info("\nBot stopped by user")
    except Exception as e:
        logger.error(f"Error starting bot: {e}", exc_info=True)
        sys.exit(1)
    finally:
        logger.info("Bot shutdown complete")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
