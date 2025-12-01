"""
Show training results from saved checkpoint.

Quick script to display model metrics without re-running evaluation.
"""

import torch
from pathlib import Path

# Path to trained model
model_path = Path("models/resnet18/best_model.pth")

if not model_path.exists():
    print(f"❌ Model not found: {model_path}")
    print("\nPlease train the model first:")
    print("  ./train_resnet_fast.sh")
    exit(1)

# Load checkpoint
print("Loading checkpoint...")
checkpoint = torch.load(model_path, weights_only=False)

# Display results
print("\n" + "="*60)
print("ResNet-18 Training Results")
print("="*60)
print(f"\n✓ Training completed at epoch: {checkpoint['epoch']}")
print(f"✓ Best Validation Accuracy: {checkpoint['val_acc']:.4f} ({checkpoint['val_acc']*100:.2f}%)")
print(f"✓ Best Validation F1-Score: {checkpoint['val_f1']:.4f}")
print(f"\nClasses: {checkpoint['class_names']}")

print("\n" + "="*60)
print("Files created:")
print("="*60)
print(f"  ✓ Model: models/resnet18/best_model.pth")
print(f"  ✓ Confusion Matrix: models/resnet18/confusion_matrix.png")
print(f"  ✓ Training History: models/resnet18/training_history.png")

print("\n" + "="*60)
print("Next Steps:")
print("="*60)
print("1. View confusion_matrix.png to see where model makes errors")
print("2. View training_history.png to see learning curves")
print("3. Use this model for the Telegram bot!")
print("\nModel is ready for production! 🚀")
print("="*60 + "\n")
