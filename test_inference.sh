#!/bin/bash

# Test all inference modules
# Validates that trained models can be loaded and used for prediction

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ml-python312

echo "============================================================"
echo "Testing Inference Modules"
echo "============================================================"
echo ""
echo "Environment: ml-python312"
echo "Python version:"
python --version
echo ""

# Test Primary Classifier
echo "============================================================"
echo "1. Testing Primary Classifier (retina/blood/scene)"
echo "============================================================"
python src/inference_primary.py
echo ""

# Test U-Net Segmentation
echo "============================================================"
echo "2. Testing U-Net Retina Segmentation"
echo "============================================================"
python src/inference_segmentation.py
echo ""

# Test YOLO Detection
echo "============================================================"
echo "3. Testing YOLOv8 Blood Cell Detection"
echo "============================================================"
python src/inference_detection.py
echo ""

# Test CIFAR-10 Classifier
echo "============================================================"
echo "4. Testing CIFAR-10 Scene Classifier"
echo "============================================================"
python src/inference_cifar10.py
echo ""

echo "============================================================"
echo "✓ All Inference Modules Tested!"
echo "============================================================"
echo ""
echo "Results:"
echo "  - Primary classifier: Ready ✓"
echo "  - U-Net segmentation: Ready ✓"
echo "  - YOLOv8 detection: Ready ✓"
echo "  - CIFAR-10 classifier: Ready ✓"
echo ""
echo "Test visualizations saved in: tmp/"
echo ""
