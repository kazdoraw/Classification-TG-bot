#!/bin/bash

# Quick training script for ResNet-18 only (MAIN MODEL)
# Uses ml-python312 conda environment

echo "============================================================"
echo "Quick Training: ResNet-18 (Main Model)"
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
echo "Epochs: 15, Batch size: 32"
echo "Expected time: ~15-30 minutes"
echo ""

# Train ResNet-18
python src/train_primary_resnet18.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ ResNet-18 Training Complete!"
    echo "============================================================"
    echo ""
    echo "Model saved to: models/resnet18/best_model.pth"
    echo "Confusion matrix: models/resnet18/confusion_matrix.png"
    echo "Training history: models/resnet18/training_history.png"
    echo ""
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi
