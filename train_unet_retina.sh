#!/bin/bash

# Training script for U-Net (Retina Vessel Segmentation)
# Uses ml-python312 conda environment

echo "============================================================"
echo "Training U-Net for Retina Vessel Segmentation"
echo "============================================================"
echo ""

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ml-python312

echo "Environment: ml-python312"
echo "Python version:"
python --version
echo ""
echo "Dataset: DRIVE (20 training + 20 test images)"
echo "Model: U-Net with encoder-decoder architecture"
echo "Loss: Combined Dice Loss + BCE Loss"
echo "Metrics: Dice coefficient, IoU"
echo ""
echo "Starting training..."
echo "Epochs: 20 (with early stopping, patience=7)"
echo "Batch size: 4"
echo "Image size: 256x256 (optimized for speed)"
echo "Learning rate: 0.0001 (reduced for small dataset)"
echo "Loss: Dice (0.7) + BCE with pos_weight=10.0 (0.3)"
echo "Expected time: ~7-12 minutes"
echo ""

# Train U-Net
python src/train_retina_segmentation.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ U-Net Training Complete!"
    echo "============================================================"
    echo ""
    echo "Model saved to: models/unet_retina/best_model.pth"
    echo "Training history: models/unet_retina/training_history.png"
    echo "Predictions: models/unet_retina/predictions.png"
    echo ""
    echo "Check the predictions to see vessel segmentation quality!"
    echo ""
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi
