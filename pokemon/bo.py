import subprocess
import os
import sys
import time
import importlib.util
import webbrowser

def check_module(module_name):
    """Check if a module is available, handling special cases"""
    try:
        # Handle special cases where module name differs from import name
        if module_name == "opencv" or module_name == "cv2":
            import cv2
            return True
        elif module_name == "scikit-learn" or module_name == "sklearn":
            import sklearn
            return True
        else:
            # Standard import attempt
            __import__(module_name)
            return True
    except ImportError:
        return False

def ensure_real_model():
    """Ensure a real TensorFlow model is available"""
    try:
        # Try to use the real model creator
        from create_real_model import create_real_model
        print("Setting up TensorFlow drawing recognition model...")
        if create_real_model():
            print("Successfully set up real TensorFlow model for drawing recognition")
            return True
        else:
            print("Failed to create real model, will use placeholder")
    except ImportError:
        print("Real model creator not available, will use placeholder")
    except Exception as e:
        print(f"Error creating real model: {str(e)}, will use placeholder")
    
    # If we get here, we couldn't create a real model, so set up a placeholder
    model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models/drawing_model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Set up a minimal placeholder structure
    with open(os.path.join(model_dir, 'PLACEHOLDER.txt'), 'w') as f:
        f.write("This is a placeholder only. Not a real TensorFlow model.")
    
    return False

def main():
    print("Starting TheraBrush application...")
    
    # Ensure modules are available by trying to fix paths first
    try:
        from fix_paths import fix_common_modules
        print("Checking and fixing module paths if needed...")
        fixed_modules = fix_common_modules()
        if fixed_modules:
            print(f"Fixed paths for: {', '.join(fixed_modules)}")
    except ImportError:
        print("Path fixer not available, continuing with standard checks.")
    
    # Check required dependencies with improved module checking
    missing_deps = []
    module_map = {
        "opencv-python": "cv2",  # OpenCV is imported as cv2
        "scikit-learn": "sklearn"  # scikit-learn is imported as sklearn
    }
    
    for package_name, module_name in module_map.items():
        try:
            __import__(module_name)
            print(f"✓ {package_name} is available ({module_name})")
        except ImportError as e:
            missing_deps.append(package_name)
            print(f"✗ {package_name} not found: {e}")
    
    if missing_deps:
        print("WARNING: Missing required dependencies: " + ", ".join(missing_deps))
        print("Install them with: pip install " + " ".join(missing_deps))
        print("Continuing with limited image analysis capabilities...")
        
        # Check if packages are installed but not importable
        # NOTE: Don't re-import modules that are already imported above
        try:
            # Use the module imported at the top level, not a re-import
            result = subprocess.run([sys.executable, "-m", "pip", "list"], capture_output=True, text=True)
            packages_list = result.stdout.lower()
            
            for package in missing_deps:
                if package.lower() in packages_list:
                    print(f"NOTE: {package} appears to be installed according to pip but can't be imported.")
                    print("      This might be due to an environment/path issue.")
        except Exception as e:
            print(f"Error checking installed packages: {e}")

    # Make sure the uploads directory exists
    uploads_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
    os.makedirs(uploads_path, exist_ok=True)
    
    # Set up the TensorFlow model
    ensure_real_model()
    
    # Set up Gemini API key
    os.environ["GEMINI_API_KEY"] = os.environ.get("GEMINI_API_KEY", "AIzaSyB3NdYUZLsTMeEouo7J0YgxvK5UrDXR7BI")
    
    # Start the Flask server
    print("Starting Flask server...")
    p1 = subprocess.Popen(
        [sys.executable, "app.py"], 
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=os.environ
    )
    
    # Give the server a moment to start
    time.sleep(2)
    
    # Start the emotion detection script
    print("Starting emotion detection service...")
    p2 = subprocess.Popen(
        [sys.executable, "main.py"],
        cwd=os.path.dirname(os.path.abspath(__file__)),
        env=os.environ
    )
    
    # Open browser windows
    time.sleep(2)
    print("Opening application in browser...")
    webbrowser.open("http://127.0.0.1:5002")
    
    print("Opening performance metrics dashboard...")
    webbrowser.open("http://localhost:8000")
    
    print("TheraBrush application is running!")
    print("Access the web interface at: http://127.0.0.1:5002")
    print("Access performance metrics at: http://localhost:8000")
    
    try:
        # Keep the processes running until interrupted
        p1.wait()
        p2.wait()
    except KeyboardInterrupt:
        print("\nShutting down TheraBrush application...")
        p1.terminate()
        p2.terminate()
        p1.wait()
        p2.wait()
        print("Application shutdown complete.")

if __name__ == "__main__":
    main()