"""
Train Primary Classifier using ResNet-18 with Transfer Learning.

This is the MAIN model for primary classification (3 classes: retina, blood, scene).
Uses transfer learning from ImageNet pretrained weights.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import PRIMARY_TRAIN_DIR, PRIMARY_VAL_DIR, MODELS_DIR, PRIMARY_CLASSES, DEVICE
from src.utils import setup_logger

# Configuration - OPTIMIZED FOR SPEED
CONFIG = {
    'num_classes': 3,
    'batch_size': 32,  # Increased for faster training
    'num_epochs': 15,  # Reduced for speed
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'patience': 5,  # Early stopping patience - reduced
    'device': DEVICE,
    'num_workers': 4,
    'pin_memory': True,
}

logger = setup_logger(__name__)


def get_class_weights(train_dir):
    """Calculate class weights for imbalanced dataset."""
    dataset = datasets.ImageFolder(train_dir)
    class_counts = np.bincount([label for _, label in dataset.samples])
    
    # Inverse frequency weighting
    total = sum(class_counts)
    weights = total / (len(class_counts) * class_counts)
    
    logger.info(f"Class counts: {dict(zip(PRIMARY_CLASSES, class_counts))}")
    logger.info(f"Class weights: {dict(zip(PRIMARY_CLASSES, weights))}")
    
    return torch.FloatTensor(weights)


def get_data_transforms():
    """Get data augmentation transforms for train and validation."""
    # Training transforms with augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Validation transforms (no augmentation)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def create_model(num_classes, device):
    """Create ResNet-18 model with transfer learning."""
    # Load pretrained ResNet-18
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    
    # Freeze early layers (optional - uncomment for faster training)
    # for param in model.parameters():
    #     param.requires_grad = False
    
    # Replace final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)
    
    model = model.to(device)
    
    logger.info(f"Model created: ResNet-18 (pretrained)")
    logger.info(f"Modified FC layer: {num_features} -> {num_classes}")
    
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
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
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    epoch_f1 = f1_score(all_labels, all_preds, average='weighted')
    
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix - ResNet-18')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Confusion matrix saved to {save_path}")


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    # F1-score
    axes[2].plot(history['train_f1'], label='Train')
    axes[2].plot(history['val_f1'], label='Val')
    axes[2].set_title('F1-Score')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1-Score')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    logger.info(f"Training history saved to {save_path}")


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("Training Primary Classifier - ResNet-18")
    logger.info("="*60)
    
    # Create output directory
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = MODELS_DIR / "resnet18"
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device(CONFIG['device'])
    logger.info(f"Using device: {device}")
    
    # Data transforms
    train_transforms, val_transforms = get_data_transforms()
    
    # Load datasets
    train_dataset = datasets.ImageFolder(PRIMARY_TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(PRIMARY_VAL_DIR, transform=val_transforms)
    
    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Val dataset: {len(val_dataset)} images")
    logger.info(f"Classes: {train_dataset.classes}")
    
    # Data loaders
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
    
    # Class weights for imbalanced dataset
    class_weights = get_class_weights(PRIMARY_TRAIN_DIR).to(device)
    
    # Create model
    model = create_model(CONFIG['num_classes'], device)
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=5
    )
    
    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    
    logger.info("\nStarting training...")
    logger.info(f"Total epochs: {CONFIG['num_epochs']}")
    logger.info(f"Early stopping patience: {CONFIG['patience']}")
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            
            # Save model
            model_path = output_dir / "best_model.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': PRIMARY_CLASSES,
                'config': CONFIG,
            }, model_path)
            
            logger.info(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
            
            # Save confusion matrix
            plot_confusion_matrix(
                val_labels, val_preds, PRIMARY_CLASSES,
                output_dir / "confusion_matrix.png"
            )
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
        
        # Early stopping
        if epochs_without_improvement >= CONFIG['patience']:
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    
    # Plot training history
    plot_training_history(history, output_dir / "training_history.png")
    
    # Load best model and print final classification report
    checkpoint = torch.load(output_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    _, _, _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(
        val_labels, val_preds,
        target_names=PRIMARY_CLASSES,
        digits=4
    ))
    
    logger.info(f"\nModel saved to: {output_dir / 'best_model.pth'}")
    logger.info(f"Confusion matrix: {output_dir / 'confusion_matrix.png'}")
    logger.info(f"Training history: {output_dir / 'training_history.png'}")


if __name__ == "__main__":
    main()
