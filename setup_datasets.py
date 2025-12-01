"""
Script to download and prepare all datasets for the project.

Prerequisites:
1. Install kaggle: pip install kaggle
2. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/account
   - Click "Create New API Token"
   - Save kaggle.json to ~/.kaggle/kaggle.json
   - chmod 600 ~/.kaggle/kaggle.json
"""

import os
import shutil
import zipfile
from pathlib import Path
import random
from tqdm import tqdm


def download_dataset(dataset_name: str, output_dir: Path):
    """Download dataset from Kaggle."""
    print(f"\nDownloading {dataset_name}...")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        import kaggle
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_dir),
            unzip=True
        )
        print(f"✓ Downloaded and extracted to {output_dir}")
    except Exception as e:
        print(f"✗ Failed to download {dataset_name}: {e}")
        print("Please download manually from Kaggle and place in the correct folder.")
        return False
    
    return True


def prepare_primary_dataset(
    drive_dir: Path,
    bccd_dir: Path,
    cifar10_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    images_per_class: int = 200
):
    """
    Prepare dataset for primary classification (3 classes).
    
    Args:
        drive_dir: Path to DRIVE dataset
        bccd_dir: Path to BCCD dataset
        cifar10_dir: Path to CIFAR-10 dataset
        output_dir: Output directory for primary dataset
        train_ratio: Ratio of train/val split
        images_per_class: Number of images per class
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
    process_retina_images(drive_dir, train_dir / "retina", val_dir / "retina", 
                          train_ratio, images_per_class)
    process_blood_images(bccd_dir, train_dir / "blood", val_dir / "blood",
                        train_ratio, images_per_class)
    process_scene_images(cifar10_dir, train_dir / "scene", val_dir / "scene",
                        train_ratio, images_per_class)
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    for split_name, split_dir in [("Train", train_dir), ("Val", val_dir)]:
        print(f"\n{split_name}:")
        for class_name in ["retina", "blood", "scene"]:
            count = len(list((split_dir / class_name).glob("*.*")))
            print(f"  {class_name}: {count} images")


def process_retina_images(
    drive_dir: Path,
    train_out: Path,
    val_out: Path,
    train_ratio: float,
    max_images: int
):
    """Process DRIVE retina images."""
    print("\nProcessing retina images (DRIVE)...")
    
    # Find all retina images
    image_paths = []
    
    # DRIVE structure: training/images and test/images
    for subdir in ["training/images", "test/images", "training", "test"]:
        img_dir = drive_dir / subdir
        if img_dir.exists():
            image_paths.extend(list(img_dir.glob("*.tif")))
            image_paths.extend(list(img_dir.glob("*.png")))
            image_paths.extend(list(img_dir.glob("*.jpg")))
    
    if not image_paths:
        print(f"✗ No images found in {drive_dir}")
        return
    
    # Limit number of images
    image_paths = image_paths[:max_images]
    random.shuffle(image_paths)
    
    # Split train/val
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Copy images
    copy_images(train_paths, train_out, "retina")
    copy_images(val_paths, val_out, "retina")
    
    print(f"✓ Processed {len(image_paths)} retina images")


def process_blood_images(
    bccd_dir: Path,
    train_out: Path,
    val_out: Path,
    train_ratio: float,
    max_images: int
):
    """Process BCCD blood images."""
    print("\nProcessing blood images (BCCD)...")
    
    # Find all blood images
    image_paths = []
    
    # BCCD structure can vary, search in multiple locations
    for pattern in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(list(bccd_dir.rglob(pattern)))
    
    # Filter out duplicates
    image_paths = list(set(image_paths))
    
    if not image_paths:
        print(f"✗ No images found in {bccd_dir}")
        return
    
    # Limit number of images
    image_paths = image_paths[:max_images]
    random.shuffle(image_paths)
    
    # Split train/val
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Copy images
    copy_images(train_paths, train_out, "blood")
    copy_images(val_paths, val_out, "blood")
    
    print(f"✓ Processed {len(image_paths)} blood images")


def process_scene_images(
    cifar10_dir: Path,
    train_out: Path,
    val_out: Path,
    train_ratio: float,
    max_images: int
):
    """Process CIFAR-10 scene images."""
    print("\nProcessing scene images (CIFAR-10)...")
    
    # CIFAR-10 classes to use for "scene"
    scene_classes = ["airplane", "automobile", "ship", "truck", "bird"]
    
    image_paths = []
    
    # Find images from selected classes
    for class_name in scene_classes:
        # Try multiple possible structures
        for subdir in ["train", "test", ""]:
            class_dir = cifar10_dir / subdir / class_name
            if class_dir.exists():
                image_paths.extend(list(class_dir.glob("*.png")))
                image_paths.extend(list(class_dir.glob("*.jpg")))
    
    if not image_paths:
        print(f"✗ No images found in {cifar10_dir}")
        return
    
    # Limit and shuffle
    random.shuffle(image_paths)
    image_paths = image_paths[:max_images]
    
    # Split train/val
    split_idx = int(len(image_paths) * train_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    # Copy images
    copy_images(train_paths, train_out, "scene")
    copy_images(val_paths, val_out, "scene")
    
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
    """Main function to download and prepare all datasets."""
    print("="*60)
    print("Dataset Setup Script")
    print("="*60)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Define paths
    project_root = Path(__file__).parent
    raw_data_dir = project_root / "data" / "raw"
    primary_dir = project_root / "data" / "primary"
    
    # Create directories
    raw_data_dir.mkdir(parents=True, exist_ok=True)
    
    # Dataset information
    datasets = {
        "DRIVE": {
            "kaggle_name": "andrewmvd/drive-digital-retinal-images-for-vessel-extraction",
            "output_dir": raw_data_dir / "drive"
        },
        "BCCD": {
            "kaggle_name": "surajiiitm/bccd-dataset",
            "output_dir": raw_data_dir / "bccd"
        },
        "CIFAR-10": {
            "kaggle_name": "swaroopkml/cifar10-pngs-in-folders",
            "output_dir": raw_data_dir / "cifar10"
        }
    }
    
    # Check if kaggle is installed
    try:
        import kaggle
        print("✓ Kaggle API is installed")
    except ImportError:
        print("✗ Kaggle API not installed")
        print("\nPlease install it:")
        print("  pip install kaggle")
        print("\nAnd set up credentials:")
        print("  1. Go to https://www.kaggle.com/account")
        print("  2. Click 'Create New API Token'")
        print("  3. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("  4. chmod 600 ~/.kaggle/kaggle.json")
        return
    
    # Download datasets
    print("\n" + "="*60)
    print("Step 1: Downloading Datasets")
    print("="*60)
    
    for name, info in datasets.items():
        if info["output_dir"].exists() and list(info["output_dir"].iterdir()):
            print(f"\n✓ {name} already exists in {info['output_dir']}")
            continue
        
        success = download_dataset(info["kaggle_name"], info["output_dir"])
        if not success:
            print(f"\nDataset {name} not downloaded.")
            print(f"Please download manually and place in: {info['output_dir']}")
    
    # Check if all datasets are available
    print("\n" + "="*60)
    print("Checking Downloaded Datasets")
    print("="*60)
    
    all_ready = True
    for name, info in datasets.items():
        if info["output_dir"].exists() and list(info["output_dir"].iterdir()):
            print(f"✓ {name}: {info['output_dir']}")
        else:
            print(f"✗ {name}: Not found in {info['output_dir']}")
            all_ready = False
    
    if not all_ready:
        print("\n⚠ Not all datasets are available.")
        print("Please download missing datasets manually.")
        return
    
    # Prepare primary classification dataset
    print("\n" + "="*60)
    print("Step 2: Preparing Primary Classification Dataset")
    print("="*60)
    
    prepare_primary_dataset(
        drive_dir=datasets["DRIVE"]["output_dir"],
        bccd_dir=datasets["BCCD"]["output_dir"],
        cifar10_dir=datasets["CIFAR-10"]["output_dir"],
        output_dir=primary_dir,
        train_ratio=0.8,
        images_per_class=200
    )
    
    print("\n" + "="*60)
    print("✓ Dataset setup complete!")
    print("="*60)
    print(f"\nPrimary dataset location: {primary_dir}")
    print("\nYou can now proceed to train the primary classifier.")


if __name__ == "__main__":
    main()
