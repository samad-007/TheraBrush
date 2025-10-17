"""
Utility script to download the pre-trained drawing recognition model.
This can be run separately to ensure the model is downloaded before starting the app.
"""

import os
import sys
import urllib.request
import zipfile
import argparse

def download_drawing_model(force=False):
    """Download the drawing recognition model if it doesn't exist"""
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    model_path = os.path.join(model_dir, "drawing_model")
    
    # Create the models directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    # Check if model already exists
    if os.path.exists(model_path) and not force:
        print(f"Model already exists at: {model_path}")
        print("Use --force to re-download")
        return True
    
    try:
        # URL to a pre-trained model (placeholder - replace with actual model URL)
        model_url = "https://storage.googleapis.com/therabrush-models/drawing_model_v1.zip"
        temp_zip = os.path.join(model_dir, "temp_model.zip")
        
        print(f"Downloading model from {model_url}...")
        
        # Download with progress indicator
        def show_progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            percent = int(downloaded * 100 / total_size) if total_size > 0 else 0
            sys.stdout.write(f"\rDownloading: {percent}% [{downloaded} / {total_size} bytes]")
            sys.stdout.flush()
        
        urllib.request.urlretrieve(model_url, temp_zip, show_progress)
        print("\nDownload complete!")
        
        # Extract the zip file
        print(f"Extracting model to {model_path}...")
        with zipfile.ZipFile(temp_zip, 'r') as zip_ref:
            zip_ref.extractall(model_dir)
        
        # Remove the temporary zip file
        os.remove(temp_zip)
        print("Model extraction complete")
        
        # Verify the model exists now
        if os.path.exists(model_path):
            print(f"Model successfully installed at {model_path}")
            print(f"Directory contents: {os.listdir(model_path)}")
            return True
        else:
            print(f"Model extraction did not create the expected path: {model_path}")
            return False
    except Exception as e:
        print(f"Error downloading model: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download drawing recognition model")
    parser.add_argument("--force", action="store_true", help="Force re-download even if model exists")
    args = parser.parse_args()
    
    success = download_drawing_model(force=args.force)
    sys.exit(0 if success else 1)
