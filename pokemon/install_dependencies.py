"""
Ensures all required dependencies for the real TensorFlow model are installed.
"""

import sys
import subprocess
import importlib.util

def check_package_installed(package_name):
    """Check if a Python package is installed"""
    try:
        spec = importlib.util.find_spec(package_name)
        return spec is not None
    except ModuleNotFoundError:
        return False

def install_package(package_name):
    """Install a Python package using pip"""
    print(f"Installing {package_name}...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])

def ensure_dependencies():
    """Ensure all required dependencies are installed"""
    # List of required packages
    required_packages = [
        "tensorflow",  # Core TensorFlow
        "opencv-python",  # OpenCV for image processing
        "numpy",  # Numerical operations
        "pillow",  # Image processing
        "scikit-learn"  # For supplemental analysis
    ]
    
    # Check and install missing packages
    missing_packages = []
    for package in required_packages:
        pkg_name = package.split("==")[0]  # Remove version specifier if any
        if not check_package_installed(pkg_name.replace("-", "_")):  # Handle packages with dashes
            missing_packages.append(package)
    
    # Install missing packages
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        for package in missing_packages:
            install_package(package)
        print("All dependencies installed successfully!")
    else:
        print("All required dependencies are already installed.")
    
    return True

if __name__ == "__main__":
    print("Checking and installing TheraBrush dependencies...")
    success = ensure_dependencies()
    sys.exit(0 if success else 1)
