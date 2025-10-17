"""
Helper script to set up TensorFlow for drawing recognition.
This will install the required packages and create a valid model.
"""

import os
import sys
import subprocess
import importlib.util

def check_package(package):
    """Check if a Python package is installed"""
    return importlib.util.find_spec(package) is not None

def install_package(package):
    """Install a Python package"""
    print(f"Installing {package}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])
    print(f"Successfully installed {package}")

def main():
    print("TheraBrush TensorFlow Setup")
    print("===========================\n")
    
    # Check for required packages
    required_packages = [
        "tensorflow",
        "opencv-python",
        "scikit-learn",
        "numpy",
        "pillow"
    ]
    
    missing_packages = []
    for package in required_packages:
        pkg_name = package.split("==")[0]
        if not check_package(pkg_name.replace('-', '_')):
            missing_packages.append(package)
    
    if missing_packages:
        print(f"The following required packages are missing: {', '.join(missing_packages)}")
        print("Would you like to install them now? (y/n)")
        choice = input().strip().lower()
        if choice == 'y':
            for package in missing_packages:
                install_package(package)
            print("All required packages installed!")
        else:
            print("Cannot continue without required packages.")
            return
    else:
        print("All required packages are already installed.")
    
    # Check TensorFlow version
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        if tf.__version__[0] != '2':
            print("WARNING: You're using TensorFlow version 1.x. It's recommended to use TensorFlow 2.x")
            print("Would you like to upgrade to TensorFlow 2.x? (y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                install_package("tensorflow>=2.0.0")
                print("TensorFlow upgraded to 2.x")
    except ImportError:
        print("Error: TensorFlow is not installed correctly.")
        return
    
    # Create the model
    print("\nWould you like to create a real TensorFlow model for drawing recognition? (y/n)")
    choice = input().strip().lower()
    if choice == 'y':
        try:
            script_path = os.path.join(os.path.dirname(__file__), "create_real_model.py")
            if os.path.exists(script_path):
                print("Running create_real_model.py...")
                subprocess.check_call([sys.executable, script_path, "--force"])
                print("\nModel created successfully!")
            else:
                print("Error: create_real_model.py not found.")
                return
        except Exception as e:
            print(f"Error creating model: {e}")
            return
    
    # Check if model was created successfully
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    keras_path = os.path.join(model_dir, "drawing_model.keras")
    
    if os.path.exists(keras_path):
        print(f"\nSuccess! TensorFlow model is ready at: {keras_path}")
        print("You can now run the application with: python bo.py")
        print("The system will use TensorFlow for drawing recognition.")
    else:
        print("\nModel creation may have failed. Please run 'python verify_tf.py' to check your installation.")

if __name__ == "__main__":
    main()
