"""
Prepare BCCD Dataset for YOLO Training.

Converts Pascal VOC XML annotations to YOLO format and organizes dataset:
- Converts bounding boxes from absolute to normalized coordinates
- Splits data into train/val sets (80/20)
- Creates YOLO-compatible directory structure
- Generates dataset.yaml configuration file

Classes:
- 0: WBC (White Blood Cells)
- 1: RBC (Red Blood Cells)
- 2: Platelets
"""

import xml.etree.ElementTree as ET
from pathlib import Path
import shutil
import random
from tqdm import tqdm
import yaml

# Configuration
BCCD_ROOT = Path(__file__).parent.parent / "data" / "BCCD_Dataset-master" / "BCCD"
OUTPUT_ROOT = Path(__file__).parent.parent / "data" / "bccd_yolo"
TRAIN_SPLIT = 0.8

# Class mapping
CLASS_NAMES = ['WBC', 'RBC', 'Platelets']
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}

print("="*60)
print("BCCD Dataset Preparation for YOLO")
print("="*60)
print(f"Source: {BCCD_ROOT}")
print(f"Output: {OUTPUT_ROOT}")
print(f"Classes: {CLASS_NAMES}")
print(f"Train/Val split: {TRAIN_SPLIT:.0%}/{1-TRAIN_SPLIT:.0%}")
print()


def parse_voc_xml(xml_file):
    """
    Parse Pascal VOC XML annotation file.
    
    Args:
        xml_file: Path to XML annotation file
        
    Returns:
        tuple: (image_width, image_height, list of annotations)
               Each annotation is (class_id, xmin, ymin, xmax, ymax)
    """
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Parse all objects
    annotations = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        
        # Skip unknown classes
        if class_name not in CLASS_TO_ID:
            continue
            
        class_id = CLASS_TO_ID[class_name]
        
        # Get bounding box
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        
        annotations.append((class_id, xmin, ymin, xmax, ymax))
    
    return width, height, annotations


def convert_to_yolo_format(width, height, annotations):
    """
    Convert Pascal VOC annotations to YOLO format.
    
    YOLO format: <class_id> <x_center> <y_center> <width> <height>
    All coordinates are normalized to [0, 1]
    
    Args:
        width: Image width
        height: Image height
        annotations: List of (class_id, xmin, ymin, xmax, ymax)
        
    Returns:
        List of YOLO format strings
    """
    yolo_annotations = []
    
    for class_id, xmin, ymin, xmax, ymax in annotations:
        # Convert to center coordinates
        x_center = (xmin + xmax) / 2.0
        y_center = (ymin + ymax) / 2.0
        bbox_width = xmax - xmin
        bbox_height = ymax - ymin
        
        # Normalize to [0, 1]
        x_center /= width
        y_center /= height
        bbox_width /= width
        bbox_height /= height
        
        # YOLO format: class x_center y_center width height
        yolo_line = f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}"
        yolo_annotations.append(yolo_line)
    
    return yolo_annotations


def main():
    """Main preparation pipeline."""
    
    # Check source directories
    images_dir = BCCD_ROOT / "JPEGImages"
    annotations_dir = BCCD_ROOT / "Annotations"
    
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not annotations_dir.exists():
        raise FileNotFoundError(f"Annotations directory not found: {annotations_dir}")
    
    # Get all annotation files
    xml_files = sorted(list(annotations_dir.glob("*.xml")))
    print(f"Found {len(xml_files)} annotation files")
    
    if len(xml_files) == 0:
        raise ValueError("No XML annotation files found!")
    
    # Shuffle and split
    random.seed(42)
    random.shuffle(xml_files)
    
    split_idx = int(len(xml_files) * TRAIN_SPLIT)
    train_files = xml_files[:split_idx]
    val_files = xml_files[split_idx:]
    
    print(f"Train samples: {len(train_files)}")
    print(f"Val samples: {len(val_files)}")
    print()
    
    # Create output directories
    for split in ['train', 'val']:
        (OUTPUT_ROOT / split / 'images').mkdir(parents=True, exist_ok=True)
        (OUTPUT_ROOT / split / 'labels').mkdir(parents=True, exist_ok=True)
    
    # Process train set
    print("Processing training set...")
    class_counts_train = {cls: 0 for cls in CLASS_NAMES}
    
    for xml_file in tqdm(train_files):
        # Parse XML
        width, height, annotations = parse_voc_xml(xml_file)
        
        # Count classes
        for class_id, *_ in annotations:
            class_counts_train[CLASS_NAMES[class_id]] += 1
        
        # Convert to YOLO format
        yolo_annotations = convert_to_yolo_format(width, height, annotations)
        
        # Get image filename
        stem = xml_file.stem  # e.g., "BloodImage_00000"
        image_file = images_dir / f"{stem}.jpg"
        
        if not image_file.exists():
            print(f"Warning: Image not found: {image_file}")
            continue
        
        # Copy image
        shutil.copy2(image_file, OUTPUT_ROOT / 'train' / 'images' / image_file.name)
        
        # Save YOLO annotations
        label_file = OUTPUT_ROOT / 'train' / 'labels' / f"{stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    # Process val set
    print("\nProcessing validation set...")
    class_counts_val = {cls: 0 for cls in CLASS_NAMES}
    
    for xml_file in tqdm(val_files):
        # Parse XML
        width, height, annotations = parse_voc_xml(xml_file)
        
        # Count classes
        for class_id, *_ in annotations:
            class_counts_val[CLASS_NAMES[class_id]] += 1
        
        # Convert to YOLO format
        yolo_annotations = convert_to_yolo_format(width, height, annotations)
        
        # Get image filename
        stem = xml_file.stem
        image_file = images_dir / f"{stem}.jpg"
        
        if not image_file.exists():
            print(f"Warning: Image not found: {image_file}")
            continue
        
        # Copy image
        shutil.copy2(image_file, OUTPUT_ROOT / 'val' / 'images' / image_file.name)
        
        # Save YOLO annotations
        label_file = OUTPUT_ROOT / 'val' / 'labels' / f"{stem}.txt"
        with open(label_file, 'w') as f:
            f.write('\n'.join(yolo_annotations))
    
    # Create dataset.yaml
    print("\nCreating dataset.yaml...")
    dataset_config = {
        'path': str(OUTPUT_ROOT.absolute()),
        'train': 'train/images',
        'val': 'val/images',
        'names': {i: name for i, name in enumerate(CLASS_NAMES)}
    }
    
    yaml_file = OUTPUT_ROOT / 'dataset.yaml'
    with open(yaml_file, 'w') as f:
        yaml.dump(dataset_config, f, default_flow_style=False)
    
    print(f"✓ Dataset YAML saved: {yaml_file}")
    
    # Print statistics
    print("\n" + "="*60)
    print("Dataset Statistics")
    print("="*60)
    print(f"Train images: {len(train_files)}")
    print(f"Val images: {len(val_files)}")
    print(f"Total images: {len(xml_files)}")
    print()
    print("Class distribution (train):")
    for cls, count in class_counts_train.items():
        print(f"  {cls}: {count}")
    print()
    print("Class distribution (val):")
    for cls, count in class_counts_val.items():
        print(f"  {cls}: {count}")
    print()
    print(f"✓ BCCD dataset prepared for YOLO training!")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Config: {yaml_file}")


if __name__ == "__main__":
    main()
