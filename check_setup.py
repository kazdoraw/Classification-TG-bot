"""
Script to check if the environment is properly set up.
"""

import sys
from pathlib import Path


def check_python_version():
    """Check if Python version is 3.12+"""
    version = sys.version_info
    if version.major == 3 and version.minor >= 12:
        print(f"✓ Python version: {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"✗ Python version: {version.major}.{version.minor}.{version.micro}")
        print("  Required: Python 3.12+")
        return False


def check_imports():
    """Check if all required packages are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'torchvision': 'torchvision',
        'numpy': 'NumPy',
        'cv2': 'OpenCV (opencv-python)',
        'PIL': 'Pillow',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'aiogram': 'aiogram',
        'dotenv': 'python-dotenv',
        'ultralytics': 'ultralytics (YOLO)',
    }
    
    all_imported = True
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            all_imported = False
    
    return all_imported


def check_directory_structure():
    """Check if required directories exist"""
    required_dirs = [
        'data/primary/train/retina',
        'data/primary/train/blood',
        'data/primary/train/scene',
        'data/primary/val/retina',
        'data/primary/val/blood',
        'data/primary/val/scene',
        'data/raw',
        'models',
        'tmp/uploads',
        'tmp/segmentations',
        'tmp/detections',
        'src',
        'memory-bank',
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        path = Path(dir_path)
        if path.exists():
            print(f"✓ {dir_path}")
        else:
            print(f"✗ {dir_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def check_config():
    """Check if configuration is set up"""
    env_file = Path('.env')
    if env_file.exists():
        print("✓ .env file exists")
        
        # Check if bot token is set
        with open(env_file) as f:
            content = f.read()
            if 'your_bot_token_here' in content:
                print("⚠ WARNING: TELEGRAM_BOT_TOKEN not configured in .env")
                print("  Please update .env with your actual bot token")
                return False
            else:
                print("✓ TELEGRAM_BOT_TOKEN appears to be set")
                return True
    else:
        print("✗ .env file not found")
        print("  Please copy .env.example to .env and configure it")
        return False


def main():
    """Run all checks"""
    print("=" * 50)
    print("Environment Setup Check")
    print("=" * 50)
    
    print("\n1. Python Version")
    print("-" * 50)
    python_ok = check_python_version()
    
    print("\n2. Required Packages")
    print("-" * 50)
    packages_ok = check_imports()
    
    print("\n3. Directory Structure")
    print("-" * 50)
    dirs_ok = check_directory_structure()
    
    print("\n4. Configuration")
    print("-" * 50)
    config_ok = check_config()
    
    print("\n" + "=" * 50)
    if python_ok and packages_ok and dirs_ok and config_ok:
        print("✓ ALL CHECKS PASSED")
        print("Environment is ready!")
    else:
        print("✗ SOME CHECKS FAILED")
        print("Please fix the issues above before proceeding.")
        
        if not packages_ok:
            print("\nTo install packages, run:")
            print("  pip install -r requirements.txt")
    print("=" * 50)


if __name__ == "__main__":
    main()
