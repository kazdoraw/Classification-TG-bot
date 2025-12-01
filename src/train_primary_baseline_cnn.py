"""
Train Primary Classifier using Baseline CNN.

Simple CNN architecture trained from scratch for comparison with transfer learning models.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
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
    'batch_size': 32,
    'num_epochs': 20,  # Reduced for speed
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'patience': 7,  # Early stopping patience - reduced
    'device': DEVICE,
    'num_workers': 4,
    'pin_memory': True,
}

logger = setup_logger(__name__)


class BaselineCNN(nn.Module):
    """Simple CNN architecture for classification."""
    
    def __init__(self, num_classes=3):
        super(BaselineCNN, self).__init__()
        
        # Convolutional blocks
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """Forward pass."""
        x = self.conv1(x)  # 224 -> 112
        x = self.conv2(x)  # 112 -> 56
        x = self.conv3(x)  # 56 -> 28
        x = self.conv4(x)  # 28 -> 14
        
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        
        return x


def get_class_weights(train_dir):
    """Calculate class weights for imbalanced dataset."""
    dataset = datasets.ImageFolder(train_dir)
    class_counts = np.bincount([label for _, label in dataset.samples])
    
    total = sum(class_counts)
    weights = total / (len(class_counts) * class_counts)
    
    logger.info(f"Class counts: {dict(zip(PRIMARY_CLASSES, class_counts))}")
    logger.info(f"Class weights: {dict(zip(PRIMARY_CLASSES, weights))}")
    
    return torch.FloatTensor(weights)


def get_data_transforms():
    """Get data augmentation transforms."""
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transforms, val_transforms


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
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
    plt.title('Confusion Matrix - Baseline CNN')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_training_history(history, save_path):
    """Plot training history."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_title('Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
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


def main():
    """Main training function."""
    logger.info("="*60)
    logger.info("Training Primary Classifier - Baseline CNN")
    logger.info("="*60)
    
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    output_dir = MODELS_DIR / "baseline_cnn"
    output_dir.mkdir(exist_ok=True)
    
    device = torch.device(CONFIG['device'])
    logger.info(f"Using device: {device}")
    
    # Data
    train_transforms, val_transforms = get_data_transforms()
    train_dataset = datasets.ImageFolder(PRIMARY_TRAIN_DIR, transform=train_transforms)
    val_dataset = datasets.ImageFolder(PRIMARY_VAL_DIR, transform=val_transforms)
    
    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Val dataset: {len(val_dataset)} images")
    
    train_loader = DataLoader(
        train_dataset, batch_size=CONFIG['batch_size'], shuffle=True,
        num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory']
    )
    val_loader = DataLoader(
        val_dataset, batch_size=CONFIG['batch_size'], shuffle=False,
        num_workers=CONFIG['num_workers'], pin_memory=CONFIG['pin_memory']
    )
    
    # Class weights
    class_weights = get_class_weights(PRIMARY_TRAIN_DIR).to(device)
    
    # Model
    model = BaselineCNN(num_classes=CONFIG['num_classes']).to(device)
    logger.info(f"Model created: Baseline CNN")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['learning_rate'],
                          weight_decay=CONFIG['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=7
    )
    
    # Training loop
    best_val_acc = 0.0
    epochs_without_improvement = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'train_f1': [], 'val_f1': []
    }
    
    logger.info(f"\nStarting training for {CONFIG['num_epochs']} epochs...")
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"\nEpoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 60)
        
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device
        )
        
        scheduler.step(val_acc)
        
        logger.info(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_f1,
                'class_names': PRIMARY_CLASSES,
                'config': CONFIG,
            }, output_dir / "best_model.pth")
            
            logger.info(f"✓ New best model saved! Val Acc: {val_acc:.4f}")
            plot_confusion_matrix(val_labels, val_preds, PRIMARY_CLASSES,
                                output_dir / "confusion_matrix.png")
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= CONFIG['patience']:
            logger.info(f"\nEarly stopping after {epoch+1} epochs")
            break
    
    # Final results
    logger.info("\n" + "="*60)
    logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
    plot_training_history(history, output_dir / "training_history.png")
    
    checkpoint = torch.load(output_dir / "best_model.pth", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    _, _, _, val_preds, val_labels = validate(model, val_loader, criterion, device)
    
    logger.info("\nClassification Report:")
    logger.info("\n" + classification_report(
        val_labels, val_preds, target_names=PRIMARY_CLASSES, digits=4
    ))
    
    logger.info(f"\nModel saved to: {output_dir / 'best_model.pth'}")


if __name__ == "__main__":
    main()
