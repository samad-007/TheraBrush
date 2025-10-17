import os
import numpy as np
from PIL import Image
import requests
import json
import base64
from io import BytesIO

# Try importing OpenCV with better error handling
try:
    import cv2
    OPENCV_AVAILABLE = True
    print("OpenCV successfully imported")
except ImportError as e:
    OPENCV_AVAILABLE = False
    print(f"OpenCV import failed: {e}")
    print("Continuing with very simplified recognition (limited capabilities)")

# Try importing scikit-learn for color clustering
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
    print("Scikit-learn successfully imported")
except ImportError as e:
    SKLEARN_AVAILABLE = False
    print(f"Scikit-learn import failed: {e}")
    print("Color analysis will be limited")

# Classes that the QuickDraw model can recognize
QUICKDRAW_CLASSES = [
    'apple', 'banana', 'bear', 'bird', 'book', 'bridge', 'cake', 'car', 'cat', 'chair',
    'cloud', 'cup', 'dog', 'fish', 'flower', 'guitar', 'house', 'moon', 'mountain', 'sun',
    'tree', 'person', 'butterfly', 'clock', 'rain', 'star', 'umbrella', 'ocean', 'face'
]

class QuickDrawRecognizer:
    def __init__(self, model_path=None):
        """Initialize the QuickDraw recognizer"""
        self.current_prediction = None
        self.classes = QUICKDRAW_CLASSES
        self.model_loaded = False
        
        # Check if OpenCV is available
        if not OPENCV_AVAILABLE:
            print("WARNING: OpenCV not available - drawing recognition will be very limited")
    
    def preprocess_image(self, image_path=None, image_data=None):
        """Preprocess the image for analysis"""
        try:
            if image_data is not None:
                # Convert bytes to image
                img = Image.open(BytesIO(image_data))
            elif image_path and os.path.exists(image_path):
                # Load image from path
                img = Image.open(image_path)
            else:
                print("No valid image provided for preprocessing")
                return None, None
                
            # Convert to numpy array
            img_array = np.array(img)
            
            # Check if image is grayscale
            if len(img_array.shape) == 2:
                # Convert to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                # Handle RGBA by converting to RGB
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            
            # Create a grayscale version for analysis
            gray_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            
            return img_array, gray_array
            
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            return None, None
    
    def recognize(self, image_path=None, image_data=None):
        """Recognize the drawing in the image using OpenCV"""
        if not OPENCV_AVAILABLE:
            return [{"class": "undetermined", "confidence": 0.5}]
        
        try:
            # Preprocess image
            img_array, gray_array = self.preprocess_image(image_path, image_data)
            if img_array is None or gray_array is None:
                return [{"class": "error", "confidence": 0.0}]
            
            # Binarize the image
            _, binary = cv2.threshold(gray_array, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Get dimensions
            height, width = binary.shape
            
            # Count non-white pixels
            non_white = np.sum(binary > 0)
            if non_white < (height * width * 0.01):
                # The image is mostly empty
                self.current_prediction = "empty"
                return [{"class": "empty", "confidence": 0.9}]
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Get image stats
            pixel_density = non_white / (height * width)
            
            # Get main contour properties
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                x, y, w, h = cv2.boundingRect(main_contour)
                aspect_ratio = w / h if h > 0 else 0
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Calculate centroid
                M = cv2.moments(main_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = width // 2, height // 2
                
                # Count potential "eyes" (for face detection)
                potential_eyes = []
                for c in contours:
                    if c is not main_contour and cv2.contourArea(c) < area * 0.1:
                        ex, ey, ew, eh = cv2.boundingRect(c)
                        if 0.8 < ew/eh < 1.2:  # Roughly circular
                            potential_eyes.append((ex + ew//2, ey + eh//2))
                
                has_eye_pair = False
                for i, eye1 in enumerate(potential_eyes):
                    for eye2 in potential_eyes[i+1:]:
                        dist = np.sqrt((eye1[0] - eye2[0])**2 + (eye1[1] - eye2[1])**2)
                        if width * 0.1 < dist < width * 0.5:
                            has_eye_pair = True
                            break
                
                # Determine the most likely drawing type
                results = []
                
                # Check for specific shapes
                if 0.9 < circularity < 1.1:
                    # Could be sun, moon, face, or other circular object
                    if y < height * 0.3 and (pixel_density < 0.1 or len(contours) < 3):
                        # Sun/moon in sky position with few details
                        results.append({"class": "sun", "confidence": 0.8})
                        results.append({"class": "moon", "confidence": 0.7})
                    elif has_eye_pair:
                        # Likely a face with eyes
                        results.append({"class": "face", "confidence": 0.8})
                    else:
                        results.append({"class": "circle", "confidence": 0.7})
                        
                elif aspect_ratio > 1.5:
                    # Horizontal rectangle - likely landscape
                    results.append({"class": "landscape", "confidence": 0.7})
                    if y > height * 0.6:
                        # Low in frame - could be horizon
                        results.append({"class": "horizon", "confidence": 0.6})
                    
                elif aspect_ratio < 0.7:
                    # Vertical rectangle - likely a building/tree/person
                    if h > height * 0.7:
                        results.append({"class": "tree", "confidence": 0.6})
                        results.append({"class": "person", "confidence": 0.5})
                    
                elif has_eye_pair:
                    # Has eye-like features
                    results.append({"class": "face", "confidence": 0.7})
                
                # Check for number of contours
                if len(contours) > 10:
                    # Many contours could be clouds, flowers, etc.
                    if cy < height * 0.4:
                        results.append({"class": "cloud", "confidence": 0.6})
                    results.append({"class": "multiple_objects", "confidence": 0.6})
                    
                elif len(contours) <= 3 and pixel_density < 0.05:
                    # Simple drawing with few elements
                    results.append({"class": "sketch", "confidence": 0.6})
                
                # Add pixel density based detection
                if pixel_density > 0.3:
                    results.append({"class": "dense_pattern", "confidence": 0.7})
                    
                # If no results yet, add defaults
                if not results:
                    results.append({"class": "abstract", "confidence": 0.5})
                
                # Extract dominant colors if sklearn is available
                if SKLEARN_AVAILABLE:
                    try:
                        # Get colors from non-white pixels
                        mask = binary > 0
                        masked = img_array.copy()
                        masked[~np.dstack((mask, mask, mask))] = 0
                        pixels = masked.reshape(-1, 3)
                        pixels = pixels[np.any(pixels != [0, 0, 0], axis=1)]
                        
                        if len(pixels) > 0:
                            kmeans = KMeans(n_clusters=min(3, len(pixels)), n_init=10)
                            kmeans.fit(pixels)
                            centroids = kmeans.cluster_centers_
                            
                            # Convert to hex colors
                            colors = [
                                f"#{int(r):02x}{int(g):02x}{int(b):02x}" 
                                for r, g, b in centroids
                            ]
                            
                            # Add colors to the result
                            for result in results:
                                result["colors"] = colors
                    except Exception as e:
                        print(f"Error extracting colors: {str(e)}")
                
                # Set the current prediction
                self.current_prediction = results[0]["class"]
                return results
            else:
                # No contours found
                self.current_prediction = "empty"
                return [{"class": "empty", "confidence": 0.8}]
                
        except Exception as e:
            print(f"Error in drawing recognition: {str(e)}")
            self.current_prediction = "error"
            return [{"class": "error", "confidence": 0.0, "message": str(e)}]
    
    def get_current_prediction(self):
        """Return the current prediction"""
        return self.current_prediction
        
    def simple_recognition(self, image_path=None, image_data=None):
        """A very basic recognition function for when OpenCV is not available"""
        if image_data is not None:
            img = Image.open(BytesIO(image_data))
        elif image_path and os.path.exists(image_path):
            img = Image.open(image_path)
        else:
            return [{"class": "unknown", "confidence": 0.5}]
        
        # Convert to grayscale
        img = img.convert('L')
        img_array = np.array(img)
        
        # Check if the image is mostly empty
        if np.mean(img_array) > 240:  # Mostly white
            return [{"class": "empty", "confidence": 0.9}]
        
        # Calculate simple stats
        height, width = img_array.shape
        non_white = np.sum(img_array < 240)
        
        # Make simple guesses based on image stats
        if non_white < (width * height * 0.05):
            return [{"class": "empty", "confidence": 0.8}]
        
        # Check aspect ratio
        if width > height * 1.5:
            return [{"class": "landscape", "confidence": 0.6}]
        elif height > width * 1.5:
            return [{"class": "tree", "confidence": 0.5}]
            
        # Default to abstract
        self.current_prediction = "abstract"
        return [
            {"class": "abstract", "confidence": 0.6},
            {"class": "sketch", "confidence": 0.4}
        ]