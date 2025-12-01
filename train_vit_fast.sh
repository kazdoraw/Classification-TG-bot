#!/bin/bash

# Quick training script for Vision Transformer (ViT-B/16)
# Uses ml-python312 conda environment

echo "============================================================"
echo "Training Vision Transformer (ViT-B/16)"
echo "============================================================"
echo ""

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ml-python312

echo "Environment: ml-python312"
echo "Python version:"
python --version
echo ""
echo "Starting training..."
echo "Epochs: 10, Batch size: 16"
echo "Expected time: ~15-25 minutes"
echo "Note: ViT is computationally heavier than CNN/ResNet"
echo ""

# Train ViT
python src/train_primary_vit.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ ViT Training Complete!"
    echo "============================================================"
    echo ""
    echo "Model saved to: models/vit/best_model.pth"
    echo "Confusion matrix: models/vit/confusion_matrix.png"
    echo "Training history: models/vit/training_history.png"
    echo ""
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi
