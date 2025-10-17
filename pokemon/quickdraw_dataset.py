"""
Quick Draw Dataset Generator
Based on the methodology from: https://dev.to/larswaechter/recognizing-hand-drawn-doodles-using-deep-learning-ki0

This script downloads and prepares the Quick Draw dataset for training.
Uses the quickdraw Python package to access Google's Quick Draw dataset.
"""

import os
from pathlib import Path
from quickdraw import QuickDrawDataGroup, QuickDrawData

# Image dimensions (28x28 as per the article)
IMAGE_SIZE = (28, 28)
STROKE_WIDTH = 3

# Dataset parameters
MAX_DRAWINGS_PER_CLASS = 1200  # As per the article
DATASET_DIR = Path("dataset")

def generate_class_images(name, max_drawings, recognized=True):
    """
    Generate PNG images for a specific drawing class.
    
    Args:
        name: Name of the drawing class (e.g., "airplane", "cat")
        max_drawings: Maximum number of drawings to download
        recognized: Only download drawings recognized by Google's AI
    """
    directory = DATASET_DIR / name
    
    if not directory.exists():
        directory.mkdir(parents=True)
        print(f"Created directory: {directory}")
    
    try:
        print(f"Downloading {max_drawings} images for class '{name}'...")
        images = QuickDrawDataGroup(name, max_drawings=max_drawings, recognized=recognized)
        
        count = 0
        for img in images.drawings:
            filename = directory.as_posix() + "/" + str(img.key_id) + ".png"
            # Get image with stroke width of 3 and resize to 28x28
            img.get_image(stroke_width=STROKE_WIDTH).resize(IMAGE_SIZE).save(filename)
            count += 1
            
            if count % 100 == 0:
                print(f"  Progress: {count}/{max_drawings} images saved")
        
        print(f"✅ Completed: {count} images saved for '{name}'")
        return count
    
    except Exception as e:
        print(f"❌ Error generating images for '{name}': {str(e)}")
        return 0

def generate_full_dataset():
    """
    Generate the complete dataset for all 345 classes.
    This will download 414,000 images in total (345 * 1200).
    """
    print("=" * 80)
    print("QUICK DRAW DATASET GENERATOR")
    print("=" * 80)
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels")
    print(f"Stroke width: {STROKE_WIDTH}")
    print(f"Images per class: {MAX_DRAWINGS_PER_CLASS}")
    print(f"Dataset directory: {DATASET_DIR.absolute()}")
    print("=" * 80)
    
    # Create dataset directory
    DATASET_DIR.mkdir(exist_ok=True)
    
    # Get all available drawing names
    print("\nFetching available drawing classes from Quick Draw...")
    try:
        drawing_names = QuickDrawData().drawing_names
        total_classes = len(drawing_names)
        print(f"Found {total_classes} drawing classes")
        print(f"Total images to download: {total_classes * MAX_DRAWINGS_PER_CLASS}")
    except Exception as e:
        print(f"Error fetching drawing names: {e}")
        return
    
    # Generate images for each class
    print("\n" + "=" * 80)
    print("STARTING DATASET GENERATION")
    print("=" * 80)
    print("⚠️  This will take a significant amount of time!")
    print("⚠️  Estimated download size: ~2-3 GB")
    print("=" * 80 + "\n")
    
    total_images = 0
    successful_classes = 0
    failed_classes = []
    
    for idx, label in enumerate(drawing_names, 1):
        print(f"\n[{idx}/{total_classes}] Processing class: {label}")
        count = generate_class_images(label, max_drawings=MAX_DRAWINGS_PER_CLASS, recognized=True)
        
        if count > 0:
            successful_classes += 1
            total_images += count
        else:
            failed_classes.append(label)
    
    # Summary
    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"✅ Successful classes: {successful_classes}/{total_classes}")
    print(f"✅ Total images downloaded: {total_images}")
    print(f"📁 Dataset location: {DATASET_DIR.absolute()}")
    
    if failed_classes:
        print(f"\n❌ Failed classes ({len(failed_classes)}):")
        for name in failed_classes:
            print(f"   - {name}")
    
    print("=" * 80)

def generate_sample_dataset(num_classes=10, max_drawings=100):
    """
    Generate a smaller sample dataset for testing.
    
    Args:
        num_classes: Number of classes to include
        max_drawings: Number of drawings per class
    """
    print("=" * 80)
    print("QUICK DRAW SAMPLE DATASET GENERATOR")
    print("=" * 80)
    print(f"Image size: {IMAGE_SIZE[0]}x{IMAGE_SIZE[1]} pixels")
    print(f"Stroke width: {STROKE_WIDTH}")
    print(f"Images per class: {max_drawings}")
    print(f"Number of classes: {num_classes}")
    print(f"Dataset directory: {DATASET_DIR.absolute()}")
    print("=" * 80)
    
    # Create dataset directory
    DATASET_DIR.mkdir(exist_ok=True)
    
    # Get sample of drawing names
    print("\nFetching drawing classes from Quick Draw...")
    try:
        all_names = QuickDrawData().drawing_names
        # Select first num_classes for consistency
        drawing_names = all_names[:num_classes]
        print(f"Selected {len(drawing_names)} classes for sample dataset")
        print(f"Classes: {', '.join(drawing_names)}")
    except Exception as e:
        print(f"Error fetching drawing names: {e}")
        return
    
    print("\n" + "=" * 80)
    print("STARTING SAMPLE DATASET GENERATION")
    print("=" * 80 + "\n")
    
    total_images = 0
    
    for idx, label in enumerate(drawing_names, 1):
        print(f"\n[{idx}/{len(drawing_names)}] Processing class: {label}")
        count = generate_class_images(label, max_drawings=max_drawings, recognized=True)
        total_images += count
    
    print("\n" + "=" * 80)
    print("SAMPLE DATASET GENERATION COMPLETE!")
    print("=" * 80)
    print(f"✅ Classes: {len(drawing_names)}")
    print(f"✅ Total images: {total_images}")
    print(f"📁 Dataset location: {DATASET_DIR.absolute()}")
    print("=" * 80)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "sample":
        # Generate sample dataset for testing
        generate_sample_dataset(num_classes=10, max_drawings=100)
    else:
        # Generate full dataset
        print("\n⚠️  WARNING: This will download the FULL Quick Draw dataset!")
        print("⚠️  This includes 345 classes with 1200 images each (414,000 total images)")
        print("⚠️  Estimated time: 2-6 hours depending on internet speed")
        print("⚠️  Estimated size: 2-3 GB\n")
        
        response = input("Do you want to continue? (yes/no): ").strip().lower()
        
        if response in ['yes', 'y']:
            generate_full_dataset()
        else:
            print("\nGeneration cancelled.")
            print("💡 Tip: Run 'python quickdraw_dataset.py sample' to generate a small test dataset")
