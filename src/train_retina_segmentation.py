"""
Train U-Net for Retina Vessel Segmentation.

Dataset: DRIVE (Digital Retinal Images for Vessel Extraction)
- 20 training images with manual vessel segmentations
- 20 test images with vessel segmentations
- Task: Binary segmentation of blood vessels in ret

ina images

Model: U-Net architecture
Metrics: Dice coefficient, IoU, Pixel Accuracy
Loss: Dice Loss + BCE Loss
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import random
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, jaccard_score
import cv2

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config import DEVICE
from src.utils import setup_logger
from src.models.unet import UNet

# Configuration - OPTIMIZED FOR FAST TRAINING + CLASS IMBALANCE
CONFIG = {
    'img_size': 256,  # Reduced from 512 for faster training (4x speedup)
    'batch_size': 4,  # Small batch due to large images
    'num_epochs': 20,  # Increased slightly for better convergence
    'learning_rate': 0.0001,  # Reduced for small dataset (was 0.001)
    'weight_decay': 1e-4,
    'patience': 7,  # Early stopping patience
    'device': DEVICE,
    'num_workers': 4,
    'pin_memory': True,
}

logger = setup_logger(__name__)


class DRIVEDataset(Dataset):
    """
    DRIVE Dataset for retina vessel segmentation.
    
    Directory structure:
        data/raw/retina/
            images/          - Original retina images (.tif)
            1st_manual/      - Manual vessel segmentations (.gif)
            mask/            - Field of view masks (.gif)
    """
    
    def __init__(self, images_dir, masks_dir, fov_masks_dir, train=True):
        """
        Args:
            images_dir: Path to images directory
            masks_dir: Path to segmentation masks directory
            fov_masks_dir: Path to field of view masks directory
            train: If True, apply data augmentation
        """
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.fov_masks_dir = Path(fov_masks_dir)
        self.train = train
        
        # Get all image files
        self.image_files = sorted(list(self.images_dir.glob("*.tif")))
        
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in {images_dir}")
        
        logger.info(f"Found {len(self.image_files)} images in {images_dir}")
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Get corresponding mask filename
        img_name = img_path.stem
        # DRIVE naming: 21_training.tif -> 21_manual1.gif
        mask_name = img_name.split('_')[0] + '_manual1.gif'
        mask_path = self.masks_dir / mask_name
        
        if not mask_path.exists():
            # Try alternative naming
            mask_path = self.masks_dir / f"{img_name}_manual1.gif"
        
        # Load mask
        mask = Image.open(mask_path).convert('L')  # Grayscale
        
        # Resize both to same size
        resize = transforms.Resize((CONFIG['img_size'], CONFIG['img_size']))
        image = resize(image)
        mask = resize(mask)
        
        # Apply synchronized geometric augmentation for training
        if self.train:
            # Random horizontal flip
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)
            
            # Random vertical flip
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = transforms.functional.rotate(image, angle)
                mask = transforms.functional.rotate(mask, angle)
        
        # Convert to tensors
        to_tensor = transforms.ToTensor()
        mask = to_tensor(mask)
        
        # Apply normalization and color augmentation only to image
        if self.train:
            image = transforms.ColorJitter(brightness=0.2, contrast=0.2)(image)
        
        image = to_tensor(image)
        image = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])(image)
        
        # Binarize mask (threshold at 0.5)
        mask = (mask > 0.5).float()
        
        return image, mask


class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation tasks.
    
    Dice coefficient measures overlap between predicted and ground truth masks.
    Dice Loss = 1 - Dice Coefficient
    """
    
    def __init__(self, smooth=1.0):
        """
        Args:
            smooth: Smoothing factor to avoid division by zero
        """
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        """
        Args:
            pred: Predicted mask (logits or probabilities)
            target: Ground truth mask (binary)
        
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)  # Convert logits to probabilities
        
        # Flatten tensors
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combined loss: Dice Loss + Binary Cross Entropy Loss with pos_weight.
    
    This combination helps with:
    - Dice: Handles class imbalance (global overlap metric)
    - BCE with pos_weight: Pixel-wise accuracy with class balancing
    
    For medical segmentation with ~10-15% positive pixels (vessels),
    we use pos_weight to balance the rare positive class.
    """
    
    def __init__(self, dice_weight=0.7, bce_weight=0.3, pos_weight=10.0):
        """
        Args:
            dice_weight: Weight for Dice loss (higher for imbalanced data)
            bce_weight: Weight for BCE loss  
            pos_weight: Positive class weight for BCE (compensates imbalance)
                       Typically set to (# negative pixels / # positive pixels)
                       For vessels: ~8-10
        """
        super().__init__()
        self.dice_loss = DiceLoss()
        
        # BCE with pos_weight to handle class imbalance
        self.bce_loss = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )
        
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
    
    def forward(self, pred, target):
        dice = self.dice_loss(pred, target)
        bce = self.bce_loss(pred, target)
        return self.dice_weight * dice + self.bce_weight * bce




def calculate_dice_coefficient(pred, target, threshold=0.5):
    """
    Calculate Dice coefficient for binary segmentation.
    
    Args:
        pred: Predicted mask (probabilities)
        target: Ground truth mask (binary)
        threshold: Threshold for binarizing predictions
    
    Returns:
        Dice coefficient
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    
    if union == 0:
        return 1.0
    
    dice = (2.0 * intersection) / union
    return dice.item()


def calculate_iou(pred, target, threshold=0.5):
    """
    Calculate Intersection over Union (IoU) / Jaccard Index.
    
    Args:
        pred: Predicted mask (probabilities)
        target: Ground truth mask (binary)
        threshold: Threshold for binarizing predictions
    
    Returns:
        IoU score
    """
    pred = (pred > threshold).float()
    target = target.float()
    
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    
    if union == 0:
        return 1.0
    
    iou = intersection / union
    return iou.item()


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Returns:
        Average loss, Dice coefficient, IoU
    """
    model.train()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(dataloader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate metrics
        with torch.no_grad():
            probs = torch.sigmoid(outputs)
            dice = calculate_dice_coefficient(probs, masks)
            iou = calculate_iou(probs, masks)
        
        running_loss += loss.item()
        running_dice += dice
        running_iou += iou
        
        pbar.set_postfix({'loss': loss.item(), 'dice': dice, 'iou': iou})
    
    n = len(dataloader)
    return running_loss / n, running_dice / n, running_iou / n


def validate_epoch(model, dataloader, criterion, device):
    """
    Validate for one epoch.
    
    Returns:
        Average loss, Dice coefficient, IoU
    """
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    running_iou = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Calculate metrics
            probs = torch.sigmoid(outputs)
            dice = calculate_dice_coefficient(probs, masks)
            iou = calculate_iou(probs, masks)
            
            running_loss += loss.item()
            running_dice += dice
            running_iou += iou
            
            pbar.set_postfix({'loss': loss.item(), 'dice': dice, 'iou': iou})
    
    n = len(dataloader)
    return running_loss / n, running_dice / n, running_iou / n


def plot_training_history(history, save_path):
    """Plot and save training history."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Dice Coefficient
    axes[1].plot(history['train_dice'], label='Train Dice', marker='o')
    axes[1].plot(history['val_dice'], label='Val Dice', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Coefficient')
    axes[1].set_title('Dice Coefficient')
    axes[1].legend()
    axes[1].grid(True)
    
    # IoU
    axes[2].plot(history['train_iou'], label='Train IoU', marker='o')
    axes[2].plot(history['val_iou'], label='Val IoU', marker='o')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].set_title('Intersection over Union')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training history saved to {save_path}")


def visualize_predictions(model, dataset, device, save_path, num_samples=4):
    """Visualize model predictions on sample images."""
    model.eval()
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            image, mask = dataset[idx]
            image_input = image.unsqueeze(0).to(device)
            
            # Predict
            output = model(image_input)
            pred_mask = torch.sigmoid(output).cpu().squeeze()
            
            # Denormalize image for visualization
            img_display = image.cpu().permute(1, 2, 0).numpy()
            img_display = img_display * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img_display = np.clip(img_display, 0, 1)
            
            # Plot
            axes[i, 0].imshow(img_display)
            axes[i, 0].set_title('Original Image')
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(mask.squeeze(), cmap='gray')
            axes[i, 1].set_title('Ground Truth')
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(pred_mask, cmap='gray')
            axes[i, 2].set_title('Prediction')
            axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Predictions visualization saved to {save_path}")


def main():
    """Main training function."""
    logger.info("=" * 60)
    logger.info("Training U-Net for Retina Vessel Segmentation")
    logger.info("=" * 60)
    logger.info(f"Using device: {CONFIG['device']}")
    
    # Setup paths - DRIVE dataset is in data/DRIVE/
    data_root = project_root / "data" / "DRIVE"
    
    # NOTE: DRIVE test set doesn't have ground truth masks (1st_manual/)
    # We'll use only training data and split it into train/val
    train_images = data_root / "training" / "images"
    train_masks = data_root / "training" / "1st_manual"
    train_fov = data_root / "training" / "mask"
    
    # Create datasets for train and validation
    # We'll split indices manually to have different transforms
    all_files = sorted(list(train_images.glob("*.tif")))
    n_total = len(all_files)
    
    # Set random seed and shuffle indices
    torch.manual_seed(42)
    indices = torch.randperm(n_total).tolist()
    
    # Split 80/20
    train_size = int(0.8 * n_total)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create full datasets with appropriate transforms
    train_dataset_full = DRIVEDataset(train_images, train_masks, train_fov, train=True)
    val_dataset_full = DRIVEDataset(train_images, train_masks, train_fov, train=False)
    
    # Create subsets
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset_full, val_indices)
    
    logger.info(f"Train dataset: {len(train_dataset)} images")
    logger.info(f"Val dataset: {len(val_dataset)} images")
    
    # Create dataloaders
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
    device = torch.device(CONFIG['device'])
    model = UNet(n_channels=3, n_classes=1, bilinear=False).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model created: U-Net")
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Setup training with class imbalance handling
    # Dice weight increased (0.7) as it's better for imbalanced data
    # pos_weight=10 compensates for ~10:1 background:vessel ratio
    criterion = CombinedLoss(dice_weight=0.7, bce_weight=0.3, pos_weight=10.0)
    optimizer = optim.Adam(
        model.parameters(),
        lr=CONFIG['learning_rate'],
        weight_decay=CONFIG['weight_decay']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',  # Maximize Dice coefficient
        factor=0.5,
        patience=5
    )
    
    # Training loop
    best_dice = 0.0
    epochs_without_improvement = 0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_dice': [], 'val_dice': [],
        'train_iou': [], 'val_iou': []
    }
    
    # Output directory
    output_dir = project_root / "models" / "unet_retina"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nStarting training...")
    logger.info(f"Total epochs: {CONFIG['num_epochs']}")
    logger.info(f"Early stopping patience: {CONFIG['patience']}")
    logger.info("")
    
    for epoch in range(CONFIG['num_epochs']):
        logger.info(f"Epoch {epoch+1}/{CONFIG['num_epochs']}")
        logger.info("-" * 60)
        
        # Train
        train_loss, train_dice, train_iou = train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        
        # Validate
        val_loss, val_dice, val_iou = validate_epoch(
            model, val_loader, criterion, device
        )
        
        # Update learning rate
        scheduler.step(val_dice)
        
        # Log metrics
        logger.info(f"Train - Loss: {train_loss:.4f}, Dice: {train_dice:.4f}, IoU: {train_iou:.4f}")
        logger.info(f"Val   - Loss: {val_loss:.4f}, Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
        
        # Save history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        
        # Save best model
        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dice': val_dice,
                'val_iou': val_iou,
                'config': CONFIG
            }
            
            torch.save(checkpoint, output_dir / "best_model.pth")
            logger.info(f"✓ New best model saved! Val Dice: {val_dice:.4f}")
            
            # Visualize predictions
            visualize_predictions(model, val_dataset, device,
                                output_dir / "predictions.png")
        else:
            epochs_without_improvement += 1
            logger.info(f"No improvement for {epochs_without_improvement} epochs")
        
        # Early stopping
        if epochs_without_improvement >= CONFIG['patience']:
            logger.info(f"\nEarly stopping after {epoch+1} epochs")
            break
        
        logger.info("")
    
    # Final results
    logger.info("\n" + "=" * 60)
    logger.info("Training Complete!")
    logger.info("=" * 60)
    logger.info(f"Best validation Dice coefficient: {best_dice:.4f}")
    
    # Plot training history
    plot_training_history(history, output_dir / "training_history.png")
    
    logger.info(f"\nModel saved to: {output_dir / 'best_model.pth'}")
    logger.info(f"Training history: {output_dir / 'training_history.png'}")
    logger.info(f"Predictions: {output_dir / 'predictions.png'}")


if __name__ == "__main__":
    main()
