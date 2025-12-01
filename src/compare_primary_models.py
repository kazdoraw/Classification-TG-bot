"""
Compare trained primary classifier models.

Compares Baseline CNN, ResNet-18, and ViT-B/16 on validation set.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights, ViT_B_16_Weights
from sklearn.metrics import f1_score, accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import PRIMARY_VAL_DIR, MODELS_DIR, PRIMARY_CLASSES, DEVICE
from src.utils import setup_logger
from src.train_primary_baseline_cnn import BaselineCNN

logger = setup_logger(__name__)


def load_baseline_model(model_path, num_classes, device):
    """Load Baseline CNN model."""
    model = BaselineCNN(num_classes=num_classes)
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def load_resnet_model(model_path, num_classes, device):
    """Load ResNet-18 model."""
    model = models.resnet18(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def load_vit_model(model_path, num_classes, device):
    """Load ViT-B/16 model."""
    model = models.vit_b_16(weights=None)
    num_features = model.heads.head.in_features
    model.heads.head = nn.Linear(num_features, num_classes)
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    return model, checkpoint


def evaluate_model(model, dataloader, device):
    """Evaluate model on validation set."""
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return all_preds, all_labels, all_probs, accuracy, f1


def plot_comparison(results, save_path):
    """Plot comparison of models."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    models = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in models]
    f1_scores = [results[m]['f1'] for m in models]
    
    # Accuracy comparison
    axes[0].bar(models, accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_title('Validation Accuracy')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].grid(axis='y', alpha=0.3)
    for i, v in enumerate(accuracies):
        axes[0].text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    # F1-Score comparison
    axes[1].bar(models, f1_scores, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_title('Validation F1-Score')
    axes[1].set_ylabel('F1-Score')
    axes[1].set_ylim([0, 1])
    axes[1].grid(axis='y', alpha=0.3)
    for i, v in enumerate(f1_scores):
        axes[1].text(i, v + 0.02, f'{v:.4f}', ha='center')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Comparison plot saved to {save_path}")


def print_comparison_table(results):
    """Print comparison table."""
    data = []
    for model_name, metrics in results.items():
        data.append({
            'Model': model_name,
            'Accuracy': f"{metrics['accuracy']:.4f}",
            'F1-Score': f"{metrics['f1']:.4f}",
            'Epochs': metrics['epochs'],
        })
    
    df = pd.DataFrame(data)
    logger.info("\n" + "="*60)
    logger.info("Model Comparison")
    logger.info("="*60)
    logger.info("\n" + df.to_string(index=False))
    logger.info("="*60)


def main():
    """Main comparison function."""
    logger.info("="*60)
    logger.info("Comparing Primary Classifier Models")
    logger.info("="*60)
    
    device = torch.device(DEVICE)
    logger.info(f"Using device: {device}")
    
    # Data transforms
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_dataset = datasets.ImageFolder(PRIMARY_VAL_DIR, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)
    
    logger.info(f"Validation dataset: {len(val_dataset)} images")
    logger.info(f"Classes: {PRIMARY_CLASSES}\n")
    
    # Model paths
    models_info = {
        'Baseline CNN': {
            'path': MODELS_DIR / 'baseline_cnn' / 'best_model.pth',
            'loader': load_baseline_model
        },
        'ResNet-18': {
            'path': MODELS_DIR / 'resnet18' / 'best_model.pth',
            'loader': load_resnet_model
        },
        'ViT-B/16': {
            'path': MODELS_DIR / 'vit' / 'best_model.pth',
            'loader': load_vit_model
        }
    }
    
    results = {}
    
    # Evaluate each model
    for model_name, info in models_info.items():
        model_path = info['path']
        
        if not model_path.exists():
            logger.warning(f"⚠️  {model_name} model not found at {model_path}")
            logger.warning(f"   Please train the model first: python src/train_primary_{model_name.lower().replace('-', '').replace(' ', '_')}.py")
            continue
        
        logger.info(f"\nEvaluating {model_name}...")
        logger.info("-" * 60)
        
        # Load model
        model, checkpoint = info['loader'](model_path, len(PRIMARY_CLASSES), device)
        
        # Evaluate
        preds, labels, probs, accuracy, f1 = evaluate_model(model, val_loader, device)
        
        # Store results
        results[model_name] = {
            'accuracy': accuracy,
            'f1': f1,
            'epochs': checkpoint['epoch'],
            'preds': preds,
            'labels': labels
        }
        
        logger.info(f"Accuracy: {accuracy:.4f}")
        logger.info(f"F1-Score: {f1:.4f}")
        logger.info(f"Trained epochs: {checkpoint['epoch']}")
        
        # Print classification report
        logger.info("\nClassification Report:")
        logger.info("\n" + classification_report(
            labels, preds, target_names=PRIMARY_CLASSES, digits=4
        ))
    
    if not results:
        logger.error("\n✗ No models found. Please train at least one model first.")
        return
    
    # Print comparison table
    print_comparison_table(results)
    
    # Plot comparison
    plot_comparison(results, MODELS_DIR / "model_comparison.png")
    
    # Determine best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    logger.info(f"\n🏆 Best Model: {best_model[0]}")
    logger.info(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
    logger.info(f"   F1-Score: {best_model[1]['f1']:.4f}")
    
    # Recommendation
    logger.info("\n" + "="*60)
    logger.info("Recommendation for Production")
    logger.info("="*60)
    
    if 'ResNet-18' in results:
        resnet_acc = results['ResNet-18']['accuracy']
        logger.info(f"✓ ResNet-18 is recommended for production:")
        logger.info(f"  - Good accuracy: {resnet_acc:.4f}")
        logger.info(f"  - Balanced speed/performance")
        logger.info(f"  - Smaller model size")
        logger.info(f"  - Use: models/resnet18/best_model.pth")
    else:
        logger.info(f"✓ Use {best_model[0]} for production")
    
    logger.info("\n" + "="*60)


if __name__ == "__main__":
    main()
