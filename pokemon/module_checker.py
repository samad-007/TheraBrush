"""
Utility to check module availability and troubleshoot import issues.
This script helps identify and resolve module import problems.
"""

import sys
import importlib.util
import subprocess
import os

# Map from package names to their import module names
# Some packages have different names when installed vs when imported
PACKAGE_MODULE_MAP = {
    "opencv-python": "cv2",
    "scikit-learn": "sklearn",
    "tensorflow": "tensorflow",
    "numpy": "numpy",
    "pillow": "PIL",
    "psutil": "psutil",
    "flask": "flask"
}

def is_package_installed(package_name):
    """Check if a package is installed according to pip"""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except Exception:
        return False

def is_module_importable(module_name):
    """Check if a module can be imported"""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False

def find_package_path(package_name, module_name=None):
    """Find the installation path of a package"""
    if module_name is None:
        module_name = PACKAGE_MODULE_MAP.get(package_name, package_name)
    
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None and spec.origin is not None:
            return os.path.dirname(spec.origin)
    except (ImportError, AttributeError):
        pass
    
    try:
        # Try using pip to find the location
        result = subprocess.run(
            [sys.executable, "-m", "pip", "show", package_name],
            capture_output=True,
            text=True
        )
        for line in result.stdout.split("\n"):
            if line.startswith("Location:"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    
    return None

def diagnose_module_issues(packages):
    """Diagnose issues with specified packages"""
    results = {}
    sys_path = sys.path
    
    for package_name in packages:
        module_name = PACKAGE_MODULE_MAP.get(package_name, package_name)
        
        # Check installation status
        pip_installed = is_package_installed(package_name)
        importable = is_module_importable(module_name)
        path = find_package_path(package_name, module_name)
        
        results[package_name] = {
            "module_name": module_name,
            "pip_installed": pip_installed,
            "importable": importable,
            "path": path,
            "in_path": path is not None and any(path.startswith(p) for p in sys_path)
        }
    
    return results

def fix_import_issues(packages):
    """Try to fix common import issues"""
    diagnoses = diagnose_module_issues(packages)
    fixed = []
    
    for package, diagnosis in diagnoses.items():
        # Package is installed but not importable
        if diagnosis["pip_installed"] and not diagnosis["importable"]:
            if diagnosis["path"] and not diagnosis["in_path"]:
                # Path exists but is not in sys.path
                sys.path.append(os.path.dirname(diagnosis["path"]))
                if is_module_importable(diagnosis["module_name"]):
                    fixed.append(package)
    
    return fixed

def check_and_print_module_status():
    """Check and print status of common modules"""
    print("Checking module availability...")
    
    results = diagnose_module_issues(PACKAGE_MODULE_MAP.keys())
    
    print("\nModule Status:")
    print("=============")
    
    for package, data in results.items():
        status = "✓ Available" if data["importable"] else "✗ Not importable"
        if data["pip_installed"] and not data["importable"]:
            status += " (installed but can't import)"
        
        print(f"{package} ({data['module_name']}): {status}")
        if data["path"]:
            print(f"  Path: {data['path']}")
            if not data["in_path"]:
                print("  WARNING: Path not in sys.path!")
    
    # If we found any fixable issues, try to fix them
    fixable = [p for p, d in results.items() if d["pip_installed"] and not d["importable"]]
    if fixable:
        print("\nAttempting to fix import issues for: " + ", ".join(fixable))
        fixed = fix_import_issues(fixable)
        if fixed:
            print("Successfully fixed issues for: " + ", ".join(fixed))

if __name__ == "__main__":
    check_and_print_module_status()
    print("\nSystem path:")
    for path in sys.path:
        print(f"  {path}")
