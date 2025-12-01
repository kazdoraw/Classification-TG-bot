#!/bin/bash

# Train ResNet-18 for CIFAR-10 Scene Classification
# Dataset: CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
# Model: ResNet-18 with ImageNet transfer learning

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate ml-python312

echo "============================================================"
echo "Training ResNet-18 for CIFAR-10 Classification"
echo "============================================================"
echo ""
echo "Environment: ml-python312"
echo "Python version:"
python --version
echo ""
echo "Dataset: CIFAR-10 (50,000 train + 10,000 test)"
echo "Model: ResNet-18 (Transfer Learning from ImageNet)"
echo "Task: Multi-class classification (10 classes)"
echo "Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck"
echo ""
echo "⚡ OPTIMIZED FOR FAST TRAINING:"
echo "  Epochs: 15 (with early stopping, patience=5)"
echo "  Batch size: 64"
echo "  Learning rate: 0.001"
echo "  Optimizer: Adam"
echo "  Augmentation: RandomFlip, RandomCrop"
echo "  Expected time: ~5-8 minutes on CPU"
echo ""

# Train ResNet-18
python src/train_cifar10_classifier.py

if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✓ CIFAR-10 Training Complete!"
    echo "============================================================"
    echo ""
    echo "Model saved in: models/resnet18_cifar10/"
    echo "  - best_model.pth: Best checkpoint (highest accuracy)"
    echo "  - training_history.png: Training curves"
    echo "  - confusion_matrix.png: Confusion matrix"
    echo "  - classification_report.txt: Detailed metrics"
    echo ""
else
    echo ""
    echo "============================================================"
    echo "✗ Training failed!"
    echo "============================================================"
    exit 1
fi
