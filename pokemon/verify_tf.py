"""
Utility script to verify TensorFlow installation and model functionality.
Run this script to check if TensorFlow is properly installed and can create/load models.
"""

import os
import sys
import importlib.util
import shutil

def check_tf_available():
    """Check if TensorFlow is available and working"""
    print("Checking TensorFlow availability...")
    
    try:
        import tensorflow as tf
        print(f"TensorFlow version: {tf.__version__}")
        print(f"TensorFlow is installed at: {tf.__file__}")
        
        # Check GPU availability
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"GPU is available: {len(gpus)} GPU(s) detected")
            for gpu in gpus:
                print(f"  - {gpu.name}")
        else:
            print("No GPU detected, using CPU only")
        
        # Try creating a simple model
        print("\nCreating a simple test model...")
        model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(224, 224, 3)),
            tf.keras.layers.Conv2D(8, kernel_size=3, activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        
        print("Model created successfully!")
        model.summary()
        
        # Try saving and loading the model with proper extension
        temp_path = os.path.join(os.path.dirname(__file__), "temp_model.keras")
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            
            print(f"\nSaving model to {temp_path}...")
            model.save(temp_path)
            print("Model saved successfully!")
            
            print(f"\nLoading model from {temp_path}...")
            loaded_model = tf.keras.models.load_model(temp_path)
            print("Model loaded successfully!")
            
            # Clean up
            os.remove(temp_path)
            print("Test file removed")
        except Exception as e:
            print(f"Error saving/loading model: {e}")
            # Try to clean up if file exists
            if os.path.exists(temp_path):
                os.remove(temp_path)
            return False
        
        # Try inference with random data
        print("\nTesting inference...")
        import numpy as np
        random_input = np.random.random((1, 224, 224, 3)).astype(np.float32)
        prediction = model(random_input)
        print(f"Inference successful! Output shape: {prediction.shape}")
        
        print("\nTensorFlow is working correctly!")
        return True
        
    except ImportError as e:
        print(f"TensorFlow is not installed: {e}")
        print("Please install TensorFlow with: pip install tensorflow")
        return False
    except Exception as e:
        print(f"Error checking TensorFlow: {e}")
        return False

def check_real_model():
    """Check if our real model exists and can be loaded"""
    print("\nChecking real model availability...")
    
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    keras_path = os.path.join(model_dir, "drawing_model.keras")
    saved_model_path = os.path.join(model_dir, "drawing_model")
    
    if not os.path.exists(model_dir):
        print(f"Models directory not found: {model_dir}")
        return False
    
    # Check for Keras model
    if os.path.exists(keras_path):
        print(f"Found Keras model at: {keras_path}")
        try:
            import tensorflow as tf
            model = tf.keras.models.load_model(keras_path)
            print("Keras model loaded successfully!")
            model.summary()
            return True
        except Exception as e:
            print(f"Error loading Keras model: {e}")
    else:
        print(f"Keras model not found at: {keras_path}")
    
    # Check for SavedModel
    if os.path.exists(saved_model_path) and os.path.isdir(saved_model_path):
        print(f"Found SavedModel at: {saved_model_path}")
        try:
            import tensorflow as tf
            model = tf.saved_model.load(saved_model_path)
            print("SavedModel loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading SavedModel: {e}")
    else:
        print(f"SavedModel not found at: {saved_model_path}")
    
    print("No valid model found. Please run create_real_model.py to create a model.")
    return False

if __name__ == "__main__":
    print("TensorFlow Verification Tool")
    print("===========================\n")
    
    # Check TensorFlow
    tf_ok = check_tf_available()
    
    if tf_ok:
        # Check real model
        model_ok = check_real_model()
        
        if not model_ok:
            print("\nWould you like to create a real model now? (y/n)")
            choice = input().strip().lower()
            if choice == 'y':
                print("\nCreating real model...")
                try:
                    # Try to import and run the create_real_model function
                    from create_real_model import create_real_model
                    if create_real_model(force=True):
                        print("Successfully created real model!")
                        # Check again
                        model_ok = check_real_model()
                    else:
                        print("Failed to create real model.")
                except ImportError:
                    print("Could not find create_real_model module.")
                except Exception as e:
                    print(f"Error creating real model: {e}")
        
        if model_ok:
            print("\nVerification complete! Your system is ready to use TensorFlow for drawing recognition.")
            sys.exit(0)
        else:
            print("\nYour TensorFlow installation works, but there's an issue with the model.")
            print("Please run 'python create_real_model.py' to create a valid model.")
            sys.exit(1)
    else:
        print("\nThere are issues with your TensorFlow installation.")
        print("Please fix these issues before continuing.")
        sys.exit(1)
