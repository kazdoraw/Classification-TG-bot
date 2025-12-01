#!/bin/bash

# Quick model comparison script
# Compares all trained primary classifier models
# Uses ml-python312 conda environment

echo "============================================================"
echo "Comparing All Primary Classifier Models"
echo "============================================================"
echo ""

# Activate conda environment
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ml-python312

echo "Environment: ml-python312"
echo "Python version:"
python --version
echo ""

# Check which models exist
echo "Checking trained models..."
echo ""

if [ -f "models/resnet18/best_model.pth" ]; then
    echo "  ✓ ResNet-18 found"
else
    echo "  ✗ ResNet-18 not found"
fi

if [ -f "models/baseline_cnn/best_model.pth" ]; then
    echo "  ✓ Baseline CNN found"
else
    echo "  ✗ Baseline CNN not found"
fi

if [ -f "models/vit/best_model.pth" ]; then
    echo "  ✓ ViT-B/16 found"
else
    echo "  ✗ ViT-B/16 not found"
fi

echo ""
echo "Running comparison..."
echo ""

# Compare models
python src/compare_primary_models.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ Model Comparison Complete!"
    echo "============================================================"
    echo ""
    echo "Results saved to: models/model_comparison.png"
    echo ""
else
    echo ""
    echo "✗ Comparison failed!"
    exit 1
fi
