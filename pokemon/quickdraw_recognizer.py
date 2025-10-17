"""
Quick Draw CNN Recognizer
Based on: https://dev.to/larswaechter/recognizing-hand-drawn-doodles-using-deep-learning-ki0

This module loads the trained Quick Draw CNN model and provides drawing recognition.
Processes canvas drawings exactly as described in the article.
"""

import os
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras

class QuickDrawRecognizer:
    """
    Drawing recognizer using the trained Quick Draw CNN model.
    """
    
    def __init__(self, model_path="models/drawing_model.keras", class_names_path="models/class_names.txt"):
        """
        Initialize the Quick Draw recognizer.
        
        Args:
            model_path: Path to the trained Keras model
            class_names_path: Path to the class names file
        """
        self.model_path = Path(model_path)
        self.class_names_path = Path(class_names_path)
        self.model = None
        self.class_names = []
        self.model_loaded = False
        
        # Load the model and class names
        self._load_model()
        self._load_class_names()
    
    def _load_model(self):
        """Load the trained Keras model."""
        try:
            if not self.model_path.exists():
                print(f"❌ Model not found: {self.model_path}")
                print("💡 Please run 'python train_quickdraw_model.py' first")
                return
            
            print(f"Loading Quick Draw CNN model from {self.model_path}...")
            self.model = keras.models.load_model(self.model_path)
            self.model_loaded = True
            print("✅ Quick Draw CNN model loaded successfully!")
            
            # Display model info
            total_params = self.model.count_params()
            print(f"   - Model parameters: {total_params:,}")
            print(f"   - Input shape: {self.model.input_shape}")
            print(f"   - Output shape: {self.model.output_shape}")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            self.model_loaded = False
    
    def _load_class_names(self):
        """Load the class names from file."""
        try:
            if not self.class_names_path.exists():
                print(f"❌ Class names file not found: {self.class_names_path}")
                return
            
            with open(self.class_names_path, 'r') as f:
                self.class_names = [line.strip() for line in f.readlines()]
            
            print(f"✅ Loaded {len(self.class_names)} class names")
            print(f"   - Sample classes: {', '.join(self.class_names[:5])}...")
            
        except Exception as e:
            print(f"❌ Error loading class names: {str(e)}")
    
    def transform_strokes_to_image(self, strokes, bounding_box, target_size=(28, 28)):
        """
        Transform stroke data to a 28x28 grayscale image.
        This follows the preprocessing method from the article.
        
        Args:
            strokes: List of strokes, each stroke is [[x0,x1,...], [y0,y1,...]]
            bounding_box: [min_x, min_y, max_x, max_y]
            target_size: Target image size (default: 28x28)
        
        Returns:
            PIL Image object (28x28 grayscale)
        """
        if not strokes:
            # Return blank image if no strokes
            return Image.new("L", target_size, color=255)
        
        # Calculate cropped image size
        min_x, min_y, max_x, max_y = bounding_box
        width = max(1, max_x - min_x)
        height = max(1, max_y - min_y)
        
        # Create white canvas
        image = Image.new("RGB", (width, height), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)
        
        # Draw each stroke
        for stroke in strokes:
            if len(stroke) < 2 or not stroke[0] or not stroke[1]:
                continue
            
            # Build list of positions (x, y)
            positions = []
            for i in range(len(stroke[0])):
                x = stroke[0][i] - min_x  # Offset by bounding box
                y = stroke[1][i] - min_y
                positions.append((x, y))
            
            # Draw line connecting all positions
            if len(positions) > 1:
                draw.line(positions, fill=(0, 0, 0), width=3)
        
        # Convert to grayscale and resize to 28x28
        image = image.convert('L')
        image = image.resize(target_size, Image.LANCZOS)
        
        return image
    
    def preprocess_image(self, image):
        """
        Preprocess the image for model input.
        
        Args:
            image: PIL Image (28x28 grayscale)
        
        Returns:
            numpy array ready for model prediction (1, 28, 28, 1)
        """
        # Convert to numpy array
        img_array = np.array(image, dtype=np.float32)
        
        # Ensure it's 28x28
        if img_array.shape != (28, 28):
            img_array = np.resize(img_array, (28, 28))
        
        # Add channel dimension (grayscale)
        img_array = np.expand_dims(img_array, axis=-1)
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        # Model has Rescaling layer, so we don't normalize here
        # The model will handle 0-255 -> 0-1 conversion
        
        return img_array
    
    def predict(self, strokes, bounding_box, top_k=3):
        """
        Predict the drawing class from strokes.
        
        Args:
            strokes: List of strokes [[x0,x1,...], [y0,y1,...]]
            bounding_box: [min_x, min_y, max_x, max_y]
            top_k: Number of top predictions to return
        
        Returns:
            List of dicts with 'class_name', 'probability', 'confidence'
        """
        if not self.model_loaded:
            return [{
                'class_name': 'unknown',
                'probability': 0.0,
                'confidence': 0.0
            }]
        
        try:
            # Transform strokes to 28x28 image
            image = self.transform_strokes_to_image(strokes, bounding_box)
            
            # Preprocess for model
            img_array = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Get top k predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'class_name': self.class_names[idx] if idx < len(self.class_names) else f'class_{idx}',
                    'probability': float(predictions[idx]),
                    'confidence': float(predictions[idx] * 100)
                })
            
            return results
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return [{
                'class_name': 'error',
                'probability': 0.0,
                'confidence': 0.0
            }]
    
    def recognize_from_canvas_data(self, canvas_data, top_k=3):
        """
        Recognize drawing from canvas data (strokes + bounding box).
        
        Args:
            canvas_data: Dict with 'strokes' and 'box' keys
            top_k: Number of top predictions to return
        
        Returns:
            List of predictions
        """
        strokes = canvas_data.get('strokes', [])
        box = canvas_data.get('box', [0, 0, 500, 500])
        
        return self.predict(strokes, box, top_k=top_k)
    
    def recognize_from_image_file(self, image_path, top_k=3):
        """
        Recognize drawing from an image file.
        
        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return
        
        Returns:
            List of predictions
        """
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('L')
            image = image.resize((28, 28), Image.LANCZOS)
            
            # Preprocess for model
            img_array = self.preprocess_image(image)
            
            # Make prediction
            predictions = self.model.predict(img_array, verbose=0)[0]
            
            # Get top k predictions
            top_indices = np.argsort(predictions)[-top_k:][::-1]
            
            results = []
            for idx in top_indices:
                results.append({
                    'class_name': self.class_names[idx] if idx < len(self.class_names) else f'class_{idx}',
                    'probability': float(predictions[idx]),
                    'confidence': float(predictions[idx] * 100)
                })
            
            return results
            
        except Exception as e:
            print(f"Error recognizing from image file: {str(e)}")
            return [{
                'class_name': 'error',
                'probability': 0.0,
                'confidence': 0.0
            }]

# Global recognizer instance
recognizer = None

def get_recognizer():
    """Get or create the global recognizer instance."""
    global recognizer
    if recognizer is None:
        recognizer = QuickDrawRecognizer()
    return recognizer

def recognize_drawing(strokes, bounding_box, top_k=3):
    """
    Convenience function to recognize a drawing.
    
    Args:
        strokes: List of strokes
        bounding_box: Bounding box [min_x, min_y, max_x, max_y]
        top_k: Number of predictions to return
    
    Returns:
        List of predictions
    """
    rec = get_recognizer()
    return rec.predict(strokes, bounding_box, top_k=top_k)

if __name__ == "__main__":
    # Test the recognizer
    print("\n" + "=" * 80)
    print("QUICK DRAW CNN RECOGNIZER TEST")
    print("=" * 80 + "\n")
    
    rec = QuickDrawRecognizer()
    
    if rec.model_loaded:
        print("\n✅ Recognizer ready!")
        print(f"   - Model: {rec.model_path}")
        print(f"   - Classes: {len(rec.class_names)}")
        print("\n💡 The recognizer is ready to process canvas drawings")
    else:
        print("\n❌ Recognizer not ready")
        print("💡 Please train the model first: python train_quickdraw_model.py")
    
    print("\n" + "=" * 80)
