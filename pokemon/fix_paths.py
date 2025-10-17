"""
A utility to fix Python module import paths before running the application.
This script can help resolve cases where modules are installed but not found.
"""

import os
import sys
import subprocess
import importlib.util
import site

def find_module_path(module_name, package_name=None):
    """Find the path to a module"""
    try:
        if package_name is None:
            package_name = module_name
            
        # Try direct import
        spec = importlib.util.find_spec(module_name)
        if spec is not None and spec.origin is not None:
            return os.path.dirname(spec.origin)
        
        # Try using pip to find the location
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Location:"):
                return line.split(":", 1)[1].strip()
    except Exception as e:
        print(f"Error finding module {module_name}: {e}")
    
    return None

def add_to_path(path):
    """Add a path to sys.path if not already there"""
    if path and path not in sys.path:
        sys.path.insert(0, path)
        print(f"Added {path} to Python path")
        return True
    return False

def fix_common_modules():
    """Fix paths for commonly problematic modules"""
    modules_fixed = []
    
    # Common modules and their package names (if different)
    modules = [
        ("cv2", "opencv-python"),
        ("sklearn", "scikit-learn"),
        ("tensorflow", "tensorflow"),
        ("PIL", "pillow"),
        ("numpy", "numpy")
    ]
    
    for module_name, package_name in modules:
        try:
            # Check if importable first
            importlib.import_module(module_name)
            print(f"✓ {module_name} is already importable")
        except ImportError:
            # Try to find and add the path
            path = find_module_path(module_name, package_name)
            if path:
                if add_to_path(path):
                    # Check if it worked
                    try:
                        importlib.import_module(module_name)
                        print(f"✓ Fixed import for {module_name}")
                        modules_fixed.append(module_name)
                    except ImportError as e:
                        print(f"✗ Still can't import {module_name}: {e}")
                        # Try adding parent directory
                        parent_path = os.path.dirname(path)
                        if add_to_path(parent_path):
                            try:
                                importlib.import_module(module_name)
                                print(f"✓ Fixed import for {module_name} using parent directory")
                                modules_fixed.append(module_name)
                            except ImportError:
                                print(f"✗ Could not fix {module_name} import")
            else:
                print(f"✗ Could not find path for {module_name}")
    
    return modules_fixed

def check_site_packages():
    """Check site-packages directories and add to path if needed"""
    # Get site-packages directories
    site_packages = site.getsitepackages()
    user_site = site.getusersitepackages()
    
    all_site_packages = site_packages + [user_site]
    
    # Add any missing site-packages directories
    for path in all_site_packages:
        if os.path.exists(path) and path not in sys.path:
            add_to_path(path)

def create_path_file():
    """Create a .pth file in site-packages to make path fixes permanent"""
    try:
        # Get user site-packages directory
        user_site = site.getusersitepackages()
        
        # Project path to be added
        project_path = os.path.dirname(os.path.abspath(__file__))
        
        # Create the .pth file
        os.makedirs(os.path.dirname(user_site), exist_ok=True)
        
        pth_file = os.path.join(user_site, "therabrush.pth")
        with open(pth_file, "w") as f:
            f.write(project_path + "\n")
        
        print(f"Created path file at {pth_file}")
        return True
    except Exception as e:
        print(f"Error creating path file: {e}")
        return False

if __name__ == "__main__":
    print("Python Path Fixer Utility")
    print("========================\n")
    
    print(f"Python executable: {sys.executable}")
    print(f"Python version: {sys.version}")
    
    print("\nChecking site packages...")
    check_site_packages()
    
    print("\nFixing common module paths...")
    fixed = fix_common_modules()
    
    if fixed:
        print(f"\nSuccessfully fixed paths for: {', '.join(fixed)}")
        create_path_file()
    else:
        print("\nNo module paths needed to be fixed or unable to fix modules.")
    
    print("\nCurrent Python path:")
    for i, path in enumerate(sys.path):
        print(f"{i+1}. {path}")
