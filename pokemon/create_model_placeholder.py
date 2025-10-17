"""
Creates a placeholder TensorFlow model directory structure.
This prevents errors when loading models and allows the app to fall back to OpenCV.
"""

import os
import json
import numpy as np
import struct

def create_placeholder_model():
    """Create a placeholder model structure with better TF compatibility"""
    model_dir = os.path.join(os.path.dirname(__file__), "models/drawing_model")
    
    # Create the directory structure
    os.makedirs(model_dir, exist_ok=True)
    
    # Create assets directory
    assets_dir = os.path.join(model_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    # Create variables directory
    variables_dir = os.path.join(model_dir, 'variables')
    os.makedirs(variables_dir, exist_ok=True)
    
    # Create a minimal TensorFlow SavedModel that can actually load
    # This will load but not do any useful inference
    try:
        import tensorflow as tf
        # Generate a minimal functional model
        inputs = tf.keras.layers.Input(shape=(224, 224, 3))
        x = tf.keras.layers.Flatten()(inputs)
        outputs = tf.keras.layers.Dense(50, activation='softmax')(x)  # 50 classes (placeholder)
        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        
        # Save this minimal model
        tf.saved_model.save(model, model_dir)
        print("Created minimal functional TensorFlow placeholder model")
    except Exception as e:
        print(f"Could not create TensorFlow model: {e}, falling back to manual placeholder")
        
        # Create a simple minimal SavedModel structure manually
        with open(os.path.join(model_dir, 'saved_model.pb'), 'wb') as f:
            # Use valid TF model format header bytes
            f.write(bytes.fromhex('08031205736572766532010042'))  # Magic TF SavedModel header
            f.write(b'PLACEHOLDER_MODEL')
    
        # Create variables files
        with open(os.path.join(variables_dir, 'variables.index'), 'wb') as f:
            f.write(b'\x0ATF-Checkpoint-V2\x0A')
            f.write(bytes.fromhex('0100000000000000'))
        
        with open(os.path.join(variables_dir, 'variables.data-00000-of-00001'), 'wb') as f:
            f.write(bytes.fromhex('0000000000000000'))
    
    # Create a marker file explicitly identifying this as a placeholder
    with open(os.path.join(model_dir, 'PLACEHOLDER.txt'), 'w') as f:
        f.write("This is a placeholder only. Not a real TensorFlow model.")
    
    # Create a simple readme explaining this is a placeholder
    with open(os.path.join(model_dir, 'README.txt'), 'w') as f:
        f.write("""
This is a PLACEHOLDER model structure to prevent errors when loading the TensorFlow model.
The application will automatically fall back to OpenCV-based recognition.
To use a real model, replace these files with an actual TensorFlow SavedModel.

PLACEHOLDER_MODEL=true
""")
    
    print(f"Created improved placeholder model structure at {model_dir}")
    return model_dir

if __name__ == "__main__":
    model_path = create_placeholder_model()
    print("Placeholder model created successfully.")
    print(f"Model path: {model_path}")
    print("Directory contents:")
    for root, dirs, files in os.walk(model_path):
        level = root.replace(model_path, '').count(os.sep)
        indent = ' ' * 4 * level
        print(f"{indent}{os.path.basename(root)}/")
        sub_indent = ' ' * 4 * (level + 1)
        for f in files:
            print(f"{sub_indent}{f}")
