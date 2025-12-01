#!/bin/bash

# Train YOLOv8 for Blood Cell Detection
# Dataset: BCCD (Blood Cell Count and Detection)
# Model: YOLOv8n (nano - lightweight and fast)

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ml-python312

echo "============================================================"
echo "Training YOLOv8 for Blood Cell Detection"
echo "============================================================"
echo ""
echo "Environment: ml-python312"
echo "Python version:"
python --version
echo ""
echo "Dataset: BCCD (Blood Cell Count and Detection)"
echo "Model: YOLOv8n (nano)"
echo "Task: Object Detection"
echo "Classes: WBC, RBC, Platelets"
echo ""
echo "Step 1: Preparing dataset (Pascal VOC → YOLO format)"
echo "-------------------------------------------------------"

# Prepare dataset
python src/prepare_bccd_yolo.py

if [ $? -ne 0 ]; then
    echo ""
    echo "============================================================"
    echo "✗ Dataset preparation failed!"
    echo "============================================================"
    exit 1
fi

echo ""
echo "Step 2: Training YOLOv8"
echo "-------------------------------------------------------"
echo "⚡ OPTIMIZED FOR FAST TRAINING:"
echo "  Epochs: 20 (reduced from 50)"
echo "  Batch size: 16"
echo "  Image size: 416x416 (reduced from 640)"
echo "  Early stopping patience: 5 (reduced from 10)"
echo "  Learning rate: 0.01"
echo "  Optimizer: SGD"
echo "  Augmentation: Minimal (for speed)"
echo "  Expected time: ~8-12 minutes on CPU"
echo ""

# Train YOLO
python src/train_blood_detector.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ YOLOv8 Training Complete!"
    echo "============================================================"
    echo ""
    echo "Model saved in: models/yolov8_bccd/weights/"
    echo "  - best.pt: Best checkpoint (highest mAP)"
    echo "  - last.pt: Last checkpoint"
    echo ""
    echo "Training plots saved in: models/yolov8_bccd/"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "✗ Training failed!"
    echo "============================================================"
    exit 1
fi
