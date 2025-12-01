"""
Train ResNet-18 for CIFAR-10 Scene Classification.

Dataset: CIFAR-10 (10 classes of natural scenes)
Task: Multi-class classification
Model: ResNet-18 with ImageNet transfer learning

Classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

This is an auxiliary model for the Telegram bot to classify general scene images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report, f1_score
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import DEVICE
from src.utils import setup_logger

# Configuration - OPTIMIZED FOR FAST TRAINING
CONFIG = {
    'batch_size': 64,  # Increased from 32 for speed
    'num_epochs': 15,  # Reduced from 50 for speed
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'patience': 5,  # Early stopping patience
    'device': DEVICE,
    'num_workers': 4,
    'pin_memory': True,
    'num_classes': 10,
}

# CIFAR-10 class names
CLASS_NAMES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

logger = setup_logger(__name__)


def get_transforms(train=True):
    """
    Get image transforms for training or validation.
    
    Args:
        train: If True, apply data augmentation
    
    Returns:
        torchvision transforms
    """
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],  # CIFAR-10 statistics
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train model for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })
    
    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate_epoch(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc='Validation')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })
    
    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = 100. * correct / total
    
    # Calculate F1 score
    f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, f1, all_preds, all_labels


def plot_training_history(history, save_dir):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Accuracy', marker='o')
    axes[1].plot(history['val_acc'], label='Val Accuracy', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main training pipeline."""
    logger.info("="*60)
    logger.info("Training ResNet-18 for CIFAR-10 Classification")
    logger.info("="*60)
    logger.info(f"Device: {CONFIG['device']}")
    logger.info(f"Classes: {len(CLASS_NAMES)}")
    logger.info("")
    
    # Create save directory
    save_dir = project_root / "models" / "resnet18_cifar10"
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Load CIFAR-10 dataset
    logger.info("Loading CIFAR-10 dataset...")
    train_dataset = torchvision.datasets.CIFAR10(
        root=str(project_root / "data" / "cifar10"),
        train=True,
        download=True,
        transform=get_transforms(train=True)
    )
    
    val_dataset = torchvision.datasets.CIFAR10(
        root=str(project_root / "data" / "cifar10"),
        train=False,
        download=True,
        transform=get_transforms(train=False)
    )
    
    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info("")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory']
    )
    
    # Create model
    logger.info("Creating ResNet-18 model...")
    device = torch.device(CONFIG['device'])
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    
    # Modify final layer for CIFAR-10 (10 classes)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, CONFIG['num_classes'])
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: ResNet-18 (Transfer Learning)")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    logger.info("")
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3
    )
    
    # Training configuration
    logger.info("Training configuration:")
    logger.info(f"  Epochs: {CONFIG['num_epochs']}")
    logger.info(f"  Batch size: {CONFIG['batch_size']}")
    logger.info(f"  Learning rate: {CONFIG['learning_rate']}")
    logger.info(f"  Early stopping patience: {CONFIG['patience']}")
    logger.info(f"  Expected time: ~5-8 minutes")
    logger.info("")
    
    # Training loop
    logger.info("Starting training...")
    logger.info("-"*60)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_f1': []
    }
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-"*60)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%, F1: {val_f1:.4f}")
        
        # Learning rate scheduler
        old_lr = optimizer.param_groups[0]['lr']
        scheduler.step(val_acc)
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != old_lr:
            logger.info(f"Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': CLASS_NAMES,
            }, save_dir / 'best_model.pth')
            
            logger.info(f"✓ New best model saved! Val Acc: {val_acc:.2f}%")
            
            # Save confusion matrix and classification report for best model
            plot_confusion_matrix(val_labels, val_preds, CLASS_NAMES, save_dir)
            
            report = classification_report(val_labels, val_preds,
                                          target_names=CLASS_NAMES)
            with open(save_dir / 'classification_report.txt', 'w') as f:
                f.write(f"Best Model - Epoch {epoch+1}\n")
                f.write(f"Validation Accuracy: {val_acc:.2f}%\n")
                f.write(f"Validation F1 Score: {val_f1:.4f}\n\n")
                f.write(report)
        else:
            patience_counter += 1
            logger.info(f"No improvement for {patience_counter} epochs")
            
            if patience_counter >= CONFIG['patience']:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    logger.info("")
    logger.info("-"*60)
    logger.info("✓ Training complete!")
    logger.info(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Save training history
    plot_training_history(history, save_dir)
    
    with open(save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info(f"Results saved to: {save_dir}")
    
    # Load best model and show final classification report
    checkpoint = torch.load(save_dir / 'best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, _, final_preds, final_labels = validate_epoch(
        model, val_loader, criterion, device
    )
    
    logger.info("")
    logger.info("="*60)
    logger.info("Final Classification Report (Best Model)")
    logger.info("="*60)
    logger.info("")
    report = classification_report(final_labels, final_preds,
                                   target_names=CLASS_NAMES)
    logger.info(report)
    
    logger.info("="*60)
    logger.info("✓ CIFAR-10 training complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
