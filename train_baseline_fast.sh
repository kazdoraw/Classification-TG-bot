#!/bin/bash

# Quick training script for Baseline CNN
# Uses ml-python312 conda environment

echo "============================================================"
echo "Training Baseline CNN (for comparison)"
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
echo "Epochs: 20, Batch size: 32"
echo "Expected time: ~15-25 minutes"
echo ""

# Train Baseline CNN
python src/train_primary_baseline_cnn.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Baseline CNN Training Complete!"
    echo "============================================================"
    echo ""
    echo "Model saved to: models/baseline_cnn/best_model.pth"
    echo "Confusion matrix: models/baseline_cnn/confusion_matrix.png"
    echo "Training history: models/baseline_cnn/training_history.png"
    echo ""
else
    echo ""
    echo "✗ Training failed!"
    exit 1
fi
