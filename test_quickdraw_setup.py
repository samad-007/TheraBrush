#!/usr/bin/env python3
"""
Quick Draw Implementation Test Script

This script helps test and validate the Quick Draw CNN implementation.
Run this after installation to verify everything is working correctly.
"""

import os
import sys
from pathlib import Path

def check_python_version():
    """Check if Python version is compatible."""
    print("\n" + "="*80)
    print("CHECKING PYTHON VERSION")
    print("="*80)
    
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor >= 8:
        print("✅ Python version is compatible")
        return True
    else:
        print("❌ Python 3.8 or higher is required")
        return False

def check_dependencies():
    """Check if required packages are installed."""
    print("\n" + "="*80)
    print("CHECKING DEPENDENCIES")
    print("="*80)
    
    required = {
        'tensorflow': 'TensorFlow',
        'numpy': 'NumPy',
        'PIL': 'Pillow',
        'flask': 'Flask',
        'quickdraw': 'QuickDraw',
        'matplotlib': 'Matplotlib'
    }
    
    missing = []
    installed = []
    
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✅ {name:20s} - Installed")
            installed.append(name)
        except ImportError:
            print(f"❌ {name:20s} - Missing")
            missing.append(name)
    
    print()
    print(f"Installed: {len(installed)}/{len(required)}")
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("\n💡 Install missing packages:")
        print("   pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All required packages are installed")
        return True

def check_dataset():
    """Check if dataset exists."""
    print("\n" + "="*80)
    print("CHECKING DATASET")
    print("="*80)
    
    dataset_dir = Path("pokemon/dataset")
    
    if not dataset_dir.exists():
        print(f"❌ Dataset directory not found: {dataset_dir}")
        print("\n💡 Download the dataset:")
        print("   cd pokemon")
        print("   python quickdraw_dataset.py sample  # For testing (10 classes)")
        print("   python quickdraw_dataset.py         # For full dataset (345 classes)")
        return False
    
    # Count classes
    classes = [d for d in dataset_dir.iterdir() if d.is_dir()]
    num_classes = len(classes)
    
    if num_classes == 0:
        print(f"❌ No classes found in {dataset_dir}")
        return False
    
    # Count images in first class
    if classes:
        first_class = classes[0]
        images = list(first_class.glob("*.png"))
        print(f"✅ Dataset found: {dataset_dir}")
        print(f"   - Classes: {num_classes}")
        print(f"   - Sample class: {first_class.name} ({len(images)} images)")
        
        if num_classes < 345:
            print(f"\n⚠️  Sample dataset detected ({num_classes} classes)")
            print("   This is OK for testing, but full dataset is recommended for production")
        
        return True
    
    return False

def check_model():
    """Check if trained model exists."""
    print("\n" + "="*80)
    print("CHECKING TRAINED MODEL")
    print("="*80)
    
    model_paths = [
        Path("pokemon/models/drawing_model.keras"),
        Path("pokemon/models/drawing_model"),
        Path("pokemon/models/class_names.txt")
    ]
    
    all_exist = True
    for path in model_paths:
        if path.exists():
            if path.is_dir():
                files = list(path.iterdir())
                print(f"✅ {path} ({len(files)} files)")
            else:
                size_mb = path.stat().st_size / (1024 * 1024)
                print(f"✅ {path} ({size_mb:.1f} MB)")
        else:
            print(f"❌ {path} - Not found")
            all_exist = False
    
    if not all_exist:
        print("\n⚠️  Model not found or incomplete")
        print("\n💡 Train the model:")
        print("   cd pokemon")
        print("   python train_quickdraw_model.py")
        return False
    else:
        print("\n✅ Model files present")
        return True

def test_recognizer():
    """Test if recognizer can be imported and initialized."""
    print("\n" + "="*80)
    print("TESTING RECOGNIZER")
    print("="*80)
    
    try:
        # Change to pokemon directory
        original_dir = os.getcwd()
        os.chdir('pokemon')
        
        from quickdraw_recognizer import QuickDrawRecognizer
        
        print("Initializing recognizer...")
        recognizer = QuickDrawRecognizer()
        
        if recognizer.model_loaded:
            print("✅ Recognizer initialized successfully")
            print(f"   - Classes: {len(recognizer.class_names)}")
            print(f"   - Sample classes: {', '.join(recognizer.class_names[:5])}...")
            
            # Change back
            os.chdir(original_dir)
            return True
        else:
            print("❌ Recognizer failed to load model")
            os.chdir(original_dir)
            return False
    
    except Exception as e:
        print(f"❌ Error testing recognizer: {str(e)}")
        os.chdir(original_dir)
        return False

def test_flask_app():
    """Check if Flask app can be imported."""
    print("\n" + "="*80)
    print("TESTING FLASK APPLICATION")
    print("="*80)
    
    try:
        original_dir = os.getcwd()
        os.chdir('pokemon')
        
        # Just check if it can be imported
        import app_quickdraw
        
        print("✅ Flask app can be imported")
        print("\n💡 To start the server:")
        print("   cd pokemon")
        print("   python app_quickdraw.py")
        
        os.chdir(original_dir)
        return True
    
    except Exception as e:
        print(f"❌ Error importing Flask app: {str(e)}")
        os.chdir(original_dir)
        return False

def print_next_steps(checks):
    """Print next steps based on what passed/failed."""
    print("\n" + "="*80)
    print("SUMMARY & NEXT STEPS")
    print("="*80)
    
    all_passed = all(checks.values())
    
    if all_passed:
        print("\n🎉 ALL CHECKS PASSED!")
        print("\n✅ Your Quick Draw CNN implementation is ready to use!")
        print("\n🚀 To start the application:")
        print("   cd pokemon")
        print("   python app_quickdraw.py")
        print("\n   Then open: http://127.0.0.1:5002")
    else:
        print("\n⚠️  SOME CHECKS FAILED")
        print("\n📋 Complete these steps in order:")
        
        if not checks['python']:
            print("\n1. Install Python 3.8 or higher")
        
        if not checks['dependencies']:
            print("\n2. Install dependencies:")
            print("   pip install -r requirements.txt")
        
        if not checks['dataset']:
            print("\n3. Download dataset:")
            print("   cd pokemon")
            print("   python quickdraw_dataset.py sample  # Quick test")
            print("   # OR")
            print("   python quickdraw_dataset.py         # Full dataset")
        
        if not checks['model']:
            print("\n4. Train the model:")
            print("   cd pokemon")
            print("   python train_quickdraw_model.py")
        
        if not checks['recognizer']:
            print("\n5. Test the recognizer:")
            print("   cd pokemon")
            print("   python quickdraw_recognizer.py")
        
        if not checks['flask']:
            print("\n6. Start the Flask application:")
            print("   cd pokemon")
            print("   python app_quickdraw.py")
    
    print("\n" + "="*80)

def main():
    """Run all checks."""
    print("\n" + "="*80)
    print("QUICK DRAW CNN IMPLEMENTATION TEST")
    print("="*80)
    print("This script will check if your implementation is ready to use")
    print("="*80)
    
    checks = {
        'python': check_python_version(),
        'dependencies': check_dependencies(),
        'dataset': check_dataset(),
        'model': check_model(),
        'recognizer': test_recognizer(),
        'flask': test_flask_app()
    }
    
    print_next_steps(checks)

if __name__ == "__main__":
    main()
