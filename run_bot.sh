#!/bin/bash

# Run Telegram Bot
# Запускает бота с автоматической загрузкой моделей

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ml-python312

echo "============================================================"
echo "Medical Image Analysis Telegram Bot"
echo "============================================================"
echo ""
echo "Environment: ml-python312"
echo "Python version:"
python --version
echo ""

# Check if .env exists
if [ ! -f .env ]; then
    echo "❌ Error: .env file not found!"
    echo ""
    echo "Please create .env file with:"
    echo "TELEGRAM_BOT_TOKEN=your_token_here"
    echo ""
    exit 1
fi

echo "Loading models and starting bot..."
echo ""
echo "Features:"
echo "  🔬 Retina vessel segmentation (U-Net)"
echo "  🩸 Blood cell detection (YOLOv8)"
echo "  🌅 Scene classification (ResNet-18)"
echo ""
echo "Press Ctrl+C to stop the bot"
echo ""
echo "============================================================"
echo ""

# Run bot
python -m bot.main

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "============================================================"
    echo "✓ Bot stopped gracefully"
    echo "============================================================"
else
    echo "============================================================"
    echo "✗ Bot exited with error code: $exit_code"
    echo "============================================================"
fi
