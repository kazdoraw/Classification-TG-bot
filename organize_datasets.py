"""
Script to organize downloaded datasets and prepare primary classification dataset.

Analyzes the actual structure of downloaded datasets and creates the primary
classification dataset with 3 classes: retina, blood, scene.
"""

import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


def organize_drive_data(source_dir: Path, target_dir: Path):
    """
    Organize DRIVE dataset.
    
    Structure:
        source: DRIVE/training/images/*.tif, DRIVE/test/images/*.tif
        target: raw/drive/images/*.tif
    """
    print("\n" + "="*60)
    print("Organizing DRIVE Dataset (Retina)")
    print("="*60)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Copy training images
    train_imgs = source_dir / "training" / "images"
    if train_imgs.exists():
        for img_path in train_imgs.glob("*.tif"):
            shutil.copy2(img_path, images_dir / img_path.name)
    
    # Copy test images
    test_imgs = source_dir / "test" / "images"
    if test_imgs.exists():
        for img_path in test_imgs.glob("*.tif"):
            shutil.copy2(img_path, images_dir / img_path.name)
    
    count = len(list(images_dir.glob("*.tif")))
    print(f"✓ Organized {count} retina images")
    
    return count


def organize_bccd_data(source_dir: Path, target_dir: Path):
    """
    Organize BCCD dataset.
    
    Structure:
        source: BCCD_Dataset-master/BCCD/JPEGImages/*.jpg
        target: raw/bccd/images/*.jpg
    """
    print("\n" + "="*60)
    print("Organizing BCCD Dataset (Blood)")
    print("="*60)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    images_dir = target_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Find JPEGImages directory
    jpeg_dir = source_dir / "BCCD" / "JPEGImages"
    if not jpeg_dir.exists():
        print(f"✗ JPEGImages not found in {source_dir}")
        return 0
    
    # Copy all images
    for img_path in jpeg_dir.glob("*.jpg"):
        shutil.copy2(img_path, images_dir / img_path.name)
    
    count = len(list(images_dir.glob("*.jpg")))
    print(f"✓ Organized {count} blood images")
    
    return count


def organize_cifar10_data(source_dir: Path, target_dir: Path):
    """
    Organize CIFAR-10 dataset.
    
    Structure:
        source: cifar10/cifar10/train/{class}/*.png
        target: raw/cifar10/{class}/*.png
    """
    print("\n" + "="*60)
    print("Organizing CIFAR-10 Dataset (Scenes)")
    print("="*60)
    
    target_dir.mkdir(parents=True, exist_ok=True)
    
    # Find cifar10 directory (can be nested)
    cifar_train = source_dir / "cifar10" / "train"
    if not cifar_train.exists():
        cifar_train = source_dir / "train"
    
    if not cifar_train.exists():
        print(f"✗ CIFAR-10 train directory not found in {source_dir}")
        return 0
    
    total_count = 0
    
    # Copy each class
    for class_dir in cifar_train.iterdir():
        if class_dir.is_dir():
            target_class_dir = target_dir / class_dir.name
            target_class_dir.mkdir(exist_ok=True)
            
            count = 0
            for img_path in class_dir.glob("*.png"):
                shutil.copy2(img_path, target_class_dir / img_path.name)
                count += 1
            
            print(f"  {class_dir.name}: {count} images")
            total_count += count
    
    print(f"✓ Organized {total_count} CIFAR-10 images")
    
    return total_count


def prepare_primary_dataset(
    raw_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    max_per_class: int = 200
):
    """
    Prepare primary classification dataset (3 classes).
    
    Args:
        raw_dir: Directory with organized raw data
        output_dir: Output directory for primary dataset
        train_ratio: Train/val split ratio
        max_per_class: Maximum images per class
    """
    print("\n" + "="*60)
    print("Preparing Primary Classification Dataset")
    print("="*60)
    
    # Create output directories
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    for split in [train_dir, val_dir]:
        for class_name in ["retina", "blood", "scene"]:
            (split / class_name).mkdir(parents=True, exist_ok=True)
    
    # Process each class
    process_retina_class(raw_dir, train_dir, val_dir, train_ratio, max_per_class)
    process_blood_class(raw_dir, train_dir, val_dir, train_ratio, max_per_class)
    process_scene_class(raw_dir, train_dir, val_dir, train_ratio, max_per_class)
    
    # Print statistics
    print("\n" + "="*60)
    print("Primary Dataset Statistics")
    print("="*60)
    for split_name, split_dir in [("Train", train_dir), ("Val", val_dir)]:
        print(f"\n{split_name}:")
        for class_name in ["retina", "blood", "scene"]:
            count = len(list((split_dir / class_name).glob("*.*")))
            print(f"  {class_name}: {count} images")


def process_retina_class(
    raw_dir: Path,
    train_out: Path,
    val_out: Path,
    train_ratio: float,
    max_images: int
):
    """Process retina images from DRIVE."""
    print("\nProcessing retina class...")
    
    drive_images = raw_dir / "drive" / "images"
    if not drive_images.exists():
        print(f"✗ DRIVE images not found")
        return
    
    # Get all .tif images
    image_paths = list(drive_images.glob("*.tif"))
    
    # Limit and shuffle
    random.shuffle(image_paths)
    image_paths = image_paths[:min(len(image_paths), max_images)]
    
    # Split train/val
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Copy images
    copy_images(train_paths, train_out / "retina", "retina")
    copy_images(val_paths, val_out / "retina", "retina")
    
    print(f"✓ Processed {len(image_paths)} retina images")


def process_blood_class(
    raw_dir: Path,
    train_out: Path,
    val_out: Path,
    train_ratio: float,
    max_images: int
):
    """Process blood images from BCCD."""
    print("\nProcessing blood class...")
    
    bccd_images = raw_dir / "bccd" / "images"
    if not bccd_images.exists():
        print(f"✗ BCCD images not found")
        return
    
    # Get all .jpg images
    image_paths = list(bccd_images.glob("*.jpg"))
    
    # Limit and shuffle
    random.shuffle(image_paths)
    image_paths = image_paths[:min(len(image_paths), max_images)]
    
    # Split train/val
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Copy images
    copy_images(train_paths, train_out / "blood", "blood")
    copy_images(val_paths, val_out / "blood", "blood")
    
    print(f"✓ Processed {len(image_paths)} blood images")


def process_scene_class(
    raw_dir: Path,
    train_out: Path,
    val_out: Path,
    train_ratio: float,
    max_images: int
):
    """Process scene images from CIFAR-10."""
    print("\nProcessing scene class...")
    
    cifar_dir = raw_dir / "cifar10"
    if not cifar_dir.exists():
        print(f"✗ CIFAR-10 directory not found")
        return
    
    # Use selected classes for "scene"
    scene_classes = ["airplane", "automobile", "ship", "truck", "bird"]
    
    image_paths = []
    for class_name in scene_classes:
        class_dir = cifar_dir / class_name
        if class_dir.exists():
            image_paths.extend(list(class_dir.glob("*.png")))
    
    # Limit and shuffle
    random.shuffle(image_paths)
    image_paths = image_paths[:min(len(image_paths), max_images)]
    
    # Split train/val
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Copy images
    copy_images(train_paths, train_out / "scene", "scene")
    copy_images(val_paths, val_out / "scene", "scene")
    
    print(f"✓ Processed {len(image_paths)} scene images")


def copy_images(image_paths: list, output_dir: Path, prefix: str):
    """Copy images to output directory with unique names."""
    for idx, img_path in enumerate(tqdm(image_paths, desc=f"Copying {prefix}")):
        ext = img_path.suffix
        new_name = f"{prefix}_{idx:04d}{ext}"
        output_path = output_dir / new_name
        
        try:
            shutil.copy2(img_path, output_path)
        except Exception as e:
            print(f"Warning: Failed to copy {img_path}: {e}")


def main():
    """Main function."""
    print("="*60)
    print("Dataset Organization Script")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    raw_dir = data_dir / "raw"
    primary_dir = data_dir / "primary"
    
    # Create raw directory
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Source directories (downloaded datasets)
    drive_source = data_dir / "DRIVE"
    bccd_source = data_dir / "BCCD_Dataset-master"
    cifar_source = data_dir / "cifar10"
    
    # Check if datasets exist
    print("\nChecking downloaded datasets...")
    datasets_ok = True
    
    for name, path in [("DRIVE", drive_source), ("BCCD", bccd_source), ("CIFAR-10", cifar_source)]:
        if path.exists():
            print(f"✓ {name} found at {path}")
        else:
            print(f"✗ {name} NOT found at {path}")
            datasets_ok = False
    
    if not datasets_ok:
        print("\n✗ Not all datasets are available. Please download them first.")
        return
    
    # Step 1: Organize datasets into data/raw/
    print("\n" + "="*60)
    print("Step 1: Organizing Raw Datasets")
    print("="*60)
    
    organize_drive_data(drive_source, raw_dir / "drive")
    organize_bccd_data(bccd_source, raw_dir / "bccd")
    organize_cifar10_data(cifar_source, raw_dir / "cifar10")
    
    # Step 2: Prepare primary classification dataset
    print("\n" + "="*60)
    print("Step 2: Creating Primary Classification Dataset")
    print("="*60)
    
    prepare_primary_dataset(
        raw_dir=raw_dir,
        output_dir=primary_dir,
        train_ratio=0.8,
        max_per_class=200
    )
    
    print("\n" + "="*60)
    print("✓ Dataset organization complete!")
    print("="*60)
    print(f"\nRaw datasets location: {raw_dir}")
    print(f"Primary dataset location: {primary_dir}")
    print("\nYou can now proceed to train the primary classifier.")


if __name__ == "__main__":
    main()
