"""
A safer entrypoint for TheraBrush that ensures modules are found correctly.
This script fixes common import issues before launching the main application.
"""

import os
import sys
import importlib.util
import subprocess  # Make sure subprocess is imported at the top level

def ensure_module_importable(module_name, package_name=None):
    """Ensure a module can be imported"""
    if package_name is None:
        package_name = module_name
        
    try:
        # Try direct import first
        importlib.import_module(module_name)
        return True
    except ImportError:
        print(f"Module {module_name} not directly importable, attempting to fix...")
        
        # Try to run the path fixer
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            fix_paths_script = os.path.join(current_dir, "fix_paths.py")
            
            if os.path.exists(fix_paths_script):
                subprocess.run([sys.executable, fix_paths_script], check=True)
                
                # Try import again
                try:
                    importlib.import_module(module_name)
                    print(f"Successfully fixed import for {module_name}")
                    return True
                except ImportError:
                    pass
        except Exception as e:
            print(f"Error running path fixer: {e}")
        
        print(f"Could not fix import for {module_name}")
        return False

def main():
    """Main entrypoint function"""
    # Check key modules
    print("Checking for required modules...")
    
    modules_ok = True
    for module, package in [("cv2", "opencv-python"), ("sklearn", "scikit-learn"), ("tensorflow", "tensorflow")]:
        if not ensure_module_importable(module, package):
            modules_ok = False
            print(f"WARNING: {module} import not resolved.")
        else:
            print(f"✓ {module} is available")
    
    # Continue with application startup
    print("\nStarting TheraBrush application...")
    
    # Import and run main function from bo.py
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        sys.path.insert(0, current_dir)
        
        from bo import main as bo_main
        bo_main()
    except Exception as e:
        print(f"Error starting application: {e}")
        print("For help troubleshooting, try running: python fix_paths.py")
        sys.exit(1)

if __name__ == "__main__":
    main()
