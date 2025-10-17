from flask import Flask, render_template, request, jsonify
import os
import base64
from io import BytesIO
from PIL import Image, ImageStat
import time
import json
import numpy as np
from chatgpt_advisor import ChatGPTAdvisor
import sys

# Add robust OpenCV import
try:
    import cv2
    CV_AVAILABLE = True
except ImportError:
    # Try to recover with our module fixer
    try:
        from module_checker import fix_import_issues
        if 'opencv-python' in fix_import_issues(['opencv-python']):
            import cv2
            CV_AVAILABLE = True
        else:
            CV_AVAILABLE = False
            print("OpenCV (cv2) could not be imported - image analysis will be limited")
    except:
        CV_AVAILABLE = False
        print("OpenCV (cv2) could not be imported - image analysis will be limited")

# Import the TensorFlow drawing recognition model
try:
    from tf_drawing_recognition import TFDrawingRecognizer
    TF_AVAILABLE = True
    print("TensorFlow drawing recognition module imported successfully")
except ImportError as e:
    TF_AVAILABLE = False
    print(f"TensorFlow drawing recognition not available: {str(e)}")
    print("Using OpenCV-based recognition only")

# Robust scikit-learn import
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
    print("Scikit-learn successfully imported")
except ImportError as e:
    # Try to recover with our module fixer
    try:
        from module_checker import fix_import_issues
        if 'scikit-learn' in fix_import_issues(['scikit-learn']):
            from sklearn.cluster import KMeans
            SKLEARN_AVAILABLE = True
            print("Scikit-learn successfully imported after path fix")
        else:
            SKLEARN_AVAILABLE = False
            print(f"Scikit-learn import failed: {e}")
            print("Color analysis will be limited")
    except:
        SKLEARN_AVAILABLE = False
        print(f"Scikit-learn import failed: {e}")
        print("Color analysis will be limited")

from performance_metrics import performance_tracker, track_ai_performance, track_face_recognition_performance
from datetime import datetime

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TEXT_FILE_PATH = os.path.abspath(os.path.join(os.getcwd(), "./emotions.txt"))

# Initialize our advisor
advisor = ChatGPTAdvisor()

# Initialize TensorFlow recognizer if available
tf_recognizer = None
if TF_AVAILABLE:
    try:
        print("Initializing TensorFlow drawing recognition model...")
        # Set a more specific path and ensure the directory exists
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "drawing_model")
        keras_path = os.path.join(model_dir, "drawing_model.keras")
        
        # Check if we need to create a real model
        is_placeholder = False
        if os.path.exists(os.path.join(model_path, 'PLACEHOLDER.txt')):
            is_placeholder = True
        elif os.path.exists(os.path.join(model_path, 'README.txt')) and os.path.exists(model_path):
            with open(os.path.join(model_path, 'README.txt'), 'r') as f:
                content = f.read().lower()
                if 'placeholder' in content:
                    is_placeholder = True
        
        # Also consider it a placeholder if no proper Keras model exists
        if not os.path.exists(keras_path):
            is_placeholder = True
        
        if is_placeholder:
            print("No real model detected or using placeholder, attempting to create one...")
            try:
                # Import here to avoid circular imports
                from create_real_model import create_real_model
                if create_real_model():
                    print("Successfully created real TensorFlow model")
                else:
                    print("Failed to create real model, will try to use existing model")
            except ImportError:
                print("Real model creator not available")
            except Exception as e:
                print(f"Error creating real model: {e}")
        
        # Now initialize the TF recognizer
        tf_recognizer = TFDrawingRecognizer(model_path=model_path)
        
        # Display appropriate message based on the model state
        if tf_recognizer.model_loaded:
            if tf_recognizer.using_fallback:
                print(f"TensorFlow recognizer using OpenCV fallback. Reason: {tf_recognizer.fallback_reason}")
                print("Note: To use TensorFlow for drawing recognition, run 'python create_real_model.py'")
            else:
                print("SUCCESS: TensorFlow model loaded for drawing recognition!")
                print("The system is now using TensorFlow for drawing recognition instead of OpenCV.")
        else:
            print("Failed to load TensorFlow model, using CV-based fallback")
    except Exception as e:
        print(f"Error initializing TensorFlow model: {str(e)}")
        print("Continuing with OpenCV-based recognition only")
        TF_AVAILABLE = False

# Store the latest drawing recognition and suggestions
last_recognition = "abstract"
last_suggestion = None
last_emotion = "neutral"
last_colors = []
last_density_map = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """ Receives base64 image, decodes it, and saves it """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data received"})
            
        # Make sure to handle 'data:image/png;base64,' prefix
        if ',' in data['image']:
            image_data = data['image'].split(',')[1]
        else:
            image_data = data['image']
            
        img_data = base64.b64decode(image_data)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
        
        image = Image.open(BytesIO(img_data))
        image.save(filename)
        
        print(f"Image saved successfully to {filename} at {time.strftime('%H:%M:%S')}")
        return jsonify({"message": "Image saved successfully!"})
    except Exception as e:
        print(f"Error saving image: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/get_emotions', methods=['GET'])
def get_emotions():
    """ Reads emotions.txt and returns the content """
    global last_emotion
    try:
        if os.path.exists(TEXT_FILE_PATH):
            with open(TEXT_FILE_PATH, 'r') as file:
                text_content = file.read()
                
                # Extract emotion from the first line
                lines = text_content.split('\n')
                if lines and ':' in lines[0]:
                    last_emotion = lines[0].split(':')[0].strip().lower()
                    
            return jsonify({"emotions": text_content})
        return jsonify({"emotions": "No emotions detected yet."})
    except Exception as e:
        print(f"Error getting emotions: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/get_art_suggestion', methods=['GET'])
def get_art_suggestion():
    """Get art suggestion based on the current emotion"""
    global last_emotion, last_suggestion
    
    try:
        # Generate a suggestion based on current emotion using tracked method
        suggestion_result = get_simple_art_suggestion(last_emotion, last_suggestion)
        
        last_suggestion = suggestion_result["suggestion"]
        
        return jsonify({
            "emotion": last_emotion,
            "suggestion": last_suggestion,
            "source": suggestion_result["source"]
        })
    except Exception as e:
        print(f"Error getting suggestion: {str(e)}")
        return jsonify({"error": str(e)})

@track_ai_performance
def get_simple_art_suggestion(emotion, previous_suggestions=None):
    """Wrapper function to ensure proper tracking of Gemini API calls"""
    return advisor.get_art_suggestions("canvas", emotion, previous_suggestions)

@track_face_recognition_performance
def analyze_drawing_content(image_path):
    """Analyze drawing content using TensorFlow if available, falling back to OpenCV"""
    global SKLEARN_AVAILABLE, tf_recognizer  # Use the global variables
    
    try:
        # Always try to use TF recognizer first, even if it's using CV fallback underneath
        if TF_AVAILABLE and tf_recognizer:
            print("Using TensorFlow-based drawing recognizer")
            results = tf_recognizer.recognize_drawing(image_path=image_path)
            if results and len(results) > 0:
                # Get the top prediction
                top_prediction = results[0]
                drawing_type = top_prediction["class"]
                confidence = top_prediction["confidence"]
                print(f"Drawing recognized as: {drawing_type} with confidence {confidence:.2f}")
                
                # Extract additional features for more context
                features = tf_recognizer.extract_features(image_path=image_path)
                if not features:
                    return {"drawing_type": drawing_type, "confidence": confidence, "details": {}}
                
                # Extract commonly used details
                details = {
                    "area": features.get("area", 0),
                    "perimeter": features.get("perimeter", 0),
                    "aspect_ratio": features.get("aspect_ratio", 1.0),
                    "circularity": features.get("circularity", 0),
                    "pixel_density": features.get("pixel_density", 0),
                    "contour_count": features.get("contour_count", 0),
                }
                
                # Extract colors
                img = cv2.imread(image_path)
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    pixels = img_rgb.reshape(-1, 3)
                    pixels = pixels[np.any(pixels != [255, 255, 255], axis=1)]  # Filter out white
                    
                    colors = []
                    if len(pixels) > 0 and SKLEARN_AVAILABLE:
                        try:
                            kmeans = KMeans(n_clusters=3, n_init=10)
                            kmeans.fit(pixels)
                            colors = [
                                f"#{int(r):02x}{int(g):02x}{int(b):02x}" 
                                for r, g, b in kmeans.cluster_centers_
                            ]
                        except Exception:
                            # Fallback if KMeans fails
                            rgb_avg = pixels.mean(axis=0)
                            colors = [f"#{int(rgb_avg[0]):02x}{int(rgb_avg[1]):02x}{int(rgb_avg[2]):02x}"]
                    
                    details["colors"] = colors
                    
                    # Create density map
                    binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY_INV)[1]
                    height, width = binary.shape
                    density_map = []
                    for y_grid in range(3):
                        for x_grid in range(3):
                            x1 = x_grid * (width // 3)
                            y1 = y_grid * (height // 3)
                            x2 = (x_grid + 1) * (width // 3)
                            y2 = (y_grid + 1) * (height // 3)
                            
                            grid_section = binary[y1:y2, x1:x2]
                            density = np.sum(grid_section > 0) / grid_section.size
                            density_map.append(density)
                    
                    details["density_map"] = density_map
                    
                    # Check for face-specific features (eyes, mouth, nose)
                    has_eye_pair = False
                    has_mouth = False
                    has_nose = False
                    
                    # Especially important for detecting faces
                    if drawing_type == "face" or drawing_type == "circle":
                        # Re-apply the OpenCV facial feature detection
                        _, binary = cv2.threshold(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), 240, 255, cv2.THRESH_BINARY_INV)
                        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            main_contour = max(contours, key=cv2.contourArea)
                            area = cv2.contourArea(main_contour)
                            M = cv2.moments(main_contour)
                            
                            if M["m00"] != 0:
                                cx = int(M["m10"] / M["m00"])
                                cy = int(M["m01"] / M["m00"])
                            else:
                                cx, cy = width // 2, height // 2
                            
                            # Detect potential facial features
                            potential_eyes = []
                            min_eye_distance = width * 0.05
                            
                            # Sort contours by area
                            sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                            
                            # Look for eyes
                            for contour in sorted_contours[1:]:
                                contour_area = cv2.contourArea(contour)
                                if contour_area < area * 0.2:  # Small enough to be a facial feature
                                    ex, ey, ew, eh = cv2.boundingRect(contour)
                                    # Check if roughly circular or oval (typical for eyes)
                                    if 0.6 <= ew/eh <= 1.8:  
                                        eye_center = (ex + ew//2, ey + eh//2)
                                        # Also calculate circularity of this potential eye
                                        eye_perimeter = cv2.arcLength(contour, True)
                                        if eye_perimeter > 0:
                                            eye_circularity = 4 * np.pi * contour_area / (eye_perimeter * eye_perimeter)
                                            if eye_circularity > 0.5:  # More circular means more likely to be an eye
                                                potential_eyes.append(eye_center)
                            
                            # Find pairs of eyes by checking distances between potential eye centers
                            for i, eye1 in enumerate(potential_eyes):
                                for eye2 in potential_eyes[i+1:]:
                                    horizontal_dist = abs(eye1[0] - eye2[0])
                                    vertical_dist = abs(eye1[1] - eye2[1])
                                    
                                    # Eyes in a face are mostly horizontally aligned
                                    if (min_eye_distance < horizontal_dist < width * 0.5) and (vertical_dist < height * 0.2):
                                        has_eye_pair = True
                                        break
                            
                            # Look for a mouth shape (wider than tall) in the lower part of the face
                            for contour in sorted_contours[1:]:
                                mx, my, mw, mh = cv2.boundingRect(contour)
                                # Check if located in the lower half and has the right shape for a mouth
                                if (my > cy) and (mw > mh) and (mw > width * 0.1):
                                    has_mouth = True
                                    break
                            
                            # Check for potential nose (smaller blob between eyes and mouth)
                            for contour in sorted_contours[1:]:
                                nx, ny, nw, nh = cv2.boundingRect(contour)
                                # Check if located in middle area and smaller than eyes
                                if (ny > cy - height*0.2) and (ny < cy + height*0.2) and (cv2.contourArea(contour) < area * 0.1):
                                    has_nose = True
                                    break
                        
                        # If we detected facial features in a circle, override the classification to face
                        if drawing_type == "circle" and has_eye_pair and (has_mouth or has_nose):
                            drawing_type = "face"
                            confidence = 0.85
                    
                    details["has_eyes"] = has_eye_pair
                    details["has_mouth"] = has_mouth
                    details["has_nose"] = has_nose
                
                return {
                    "drawing_type": drawing_type, 
                    "confidence": float(confidence),
                    "details": details
                }
        
        # Fall back to direct OpenCV-based recognition only if TF recognizer completely failed
        else:
            print("Falling back to direct OpenCV-based drawing recognition")
            # Open the image
            img = cv2.imread(image_path)
            if img is None:
                return {"drawing_type": "abstract", "confidence": 0.5, "details": {}}
            
            # Convert to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Invert the image for better processing (assuming dark drawing on light background)
            _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # If no significant contours, it's likely empty
            if len(contours) == 0:
                return {"drawing_type": "empty", "confidence": 0.9, "details": {}}
            
            # Get image stats
            height, width = gray.shape
            total_pixels = height * width
            non_white_pixels = np.sum(binary > 0)
            pixel_density = non_white_pixels / total_pixels
            
            # Get main contour properties
            if contours:
                main_contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(main_contour)
                perimeter = cv2.arcLength(main_contour, True)
                x, y, w, h = cv2.boundingRect(main_contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Calculate circularity with guard against division by zero
                if perimeter > 0:
                    circularity = 4 * np.pi * area / (perimeter * perimeter)
                else:
                    circularity = 0
                
                # Calculate centroid
                M = cv2.moments(main_contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = width // 2, height // 2
                
                # Detect potential facial features with improved criteria
                potential_eyes = []
                min_eye_distance = width * 0.05  # Smaller min distance to catch more potential eyes
                
                # Sort contours by area (descending) to find the main shape and potential eyes
                sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
                
                # Skip the main contour and look for smaller circular contours that could be eyes
                for contour in sorted_contours[1:]:
                    contour_area = cv2.contourArea(contour)
                    if contour_area < area * 0.2:  # Small enough to be a facial feature
                        ex, ey, ew, eh = cv2.boundingRect(contour)
                        # Check if roughly circular or oval (typical for eyes)
                        if 0.6 <= ew/eh <= 1.8:  
                            eye_center = (ex + ew//2, ey + eh//2)
                            # Also calculate circularity of this potential eye
                            eye_perimeter = cv2.arcLength(contour, True)
                            if eye_perimeter > 0:
                                eye_circularity = 4 * np.pi * contour_area / (eye_perimeter * eye_perimeter)
                                if eye_circularity > 0.5:  # More circular means more likely to be an eye
                                    potential_eyes.append(eye_center)
                
                # Find pairs of eyes by checking distances between potential eye centers
                has_eye_pair = False
                for i, eye1 in enumerate(potential_eyes):
                    for eye2 in potential_eyes[i+1:]:
                        horizontal_dist = abs(eye1[0] - eye2[0])
                        vertical_dist = abs(eye1[1] - eye2[1])
                        
                        # Eyes in a face are mostly horizontally aligned
                        if (min_eye_distance < horizontal_dist < width * 0.5) and (vertical_dist < height * 0.2):
                            has_eye_pair = True
                            break
                
                # Look for a mouth shape (wider than tall) in the lower part of the face
                has_mouth = False
                for contour in sorted_contours[1:]:
                    mx, my, mw, mh = cv2.boundingRect(contour)
                    # Check if located in the lower half and has the right shape for a mouth
                    if (my > cy) and (mw > mh) and (mw > width * 0.1):
                        has_mouth = True
                        break
                
                # Check for potential nose (smaller blob between eyes and mouth)
                has_nose = False
                for contour in sorted_contours[1:]:
                    nx, ny, nw, nh = cv2.boundingRect(contour)
                    # Check if located in middle area and smaller than eyes
                    if (ny > cy - height*0.2) and (ny < cy + height*0.2) and (cv2.contourArea(contour) < area * 0.1):
                        has_nose = True
                        break
                
                # Now classify based on features with improved logic
                # First, check for circular shapes - more strict threshold
                if 0.85 < circularity < 1.15:
                    # Strongly circular shape - could be sun, ball, etc.
                    drawing_type = "circle"
                    confidence = min(0.9, circularity)
                    
                    # If it's in the top portion of the image, it's more likely to be a sun
                    if cy < height * 0.4:
                        drawing_type = "sun"
                        confidence = 0.85
                
                # Check for face based on multiple facial features       
                elif has_eye_pair and (has_mouth or has_nose):
                    drawing_type = "face"
                    # Higher confidence if we have multiple facial features
                    confidence = 0.85 if (has_mouth and has_nose) else 0.75
                
                # Check for face with just eyes (simpler face)
                elif has_eye_pair:
                    drawing_type = "face"
                    confidence = 0.7
                
                # Now check landscape or horizontal shapes
                elif aspect_ratio > 1.5 and y > height * 0.6:
                    drawing_type = "horizon"
                    confidence = 0.8
                elif aspect_ratio > 1.5:
                    drawing_type = "landscape"
                    confidence = 0.7
                
                # Vertical shapes - could be tree, person, etc.
                elif aspect_ratio < 0.7 and h > height * 0.5:
                    drawing_type = "tree"
                    confidence = 0.7
                    # If top heavy, could be a person
                    if cv2.contourArea(main_contour[:y+h//3]) > area * 0.3:
                        drawing_type = "person"
                        confidence = 0.6
                
                # Check for multiple objects BEFORE dense pattern
                elif len(contours) > 8:
                    drawing_type = "multiple_objects"
                    confidence = 0.7 + min(0.2, (len(contours) - 8) / 30)
                
                # Check for dense patterns - INCREASED threshold to >60%
                elif pixel_density > 0.6:
                    drawing_type = "dense_pattern"
                    confidence = min(0.8, pixel_density)
                elif len(contours) > 5:
                    drawing_type = "multiple_objects"
                    confidence = 0.6 + min(0.2, (len(contours) - 5) / 30)
                
                # If nothing specific is detected, fallback to shape-based analysis
                else:
                    # Use improved shape detection based on circularity
                    if circularity > 0.8:  # High circularity = circle
                        drawing_type = "circle"
                        confidence = min(0.85, circularity * 0.9)
                    elif 0.65 < circularity < 0.85:  # Medium circularity
                        if 0.8 < aspect_ratio < 1.2:  # Nearly square
                            drawing_type = "square"
                            confidence = 0.75
                        elif aspect_ratio > 1.3:
                            drawing_type = "rectangle"
                            confidence = 0.75
                        elif aspect_ratio < 0.7:
                            drawing_type = "rectangle"
                            confidence = 0.7
                        else:
                            drawing_type = "square"
                            confidence = 0.65
                    elif 0.4 < circularity < 0.65:  # Lower circularity = triangle
                        drawing_type = "triangle"
                        confidence = 0.7
                    else:
                        drawing_type = "abstract"
                        confidence = 0.5
                
                # Extract dominant colors
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pixels = img_rgb.reshape(-1, 3)
                pixels = pixels[np.any(pixels != [255, 255, 255], axis=1)]  # Filter out white
                
                colors = []
                if len(pixels) > 0:
                    if SKLEARN_AVAILABLE:
                        try:
                            kmeans = KMeans(n_clusters=3, n_init=10)
                            kmeans.fit(pixels)
                            colors = [
                                f"#{int(r):02x}{int(g):02x}{int(b):02x}" 
                                for r, g, b in kmeans.cluster_centers_
                            ]
                        except Exception as e:
                            print(f"Error using KMeans for color analysis: {str(e)}")
                            # Fallback if KMeans fails
                            rgb_avg = pixels.mean(axis=0)
                            colors = [f"#{int(rgb_avg[0]):02x}{int(rgb_avg[1]):02x}{int(rgb_avg[2]):02x}"]
                    else:
                        # Simple color extraction without KMeans
                        rgb_avg = pixels.mean(axis=0)
                        colors = [f"#{int(rgb_avg[0]):02x}{int(rgb_avg[1]):02x}{int(rgb_avg[2]):02x}"]
                
                # Create a density map (simplified)
                density_map = []
                for y_grid in range(3):
                    for x_grid in range(3):
                        x1 = x_grid * (width // 3)
                        y1 = y_grid * (height // 3)
                        x2 = (x_grid + 1) * (width // 3)
                        y2 = (y_grid + 1) * (height // 3)
                        
                        grid_section = binary[y1:y2, x1:x2]
                        density = np.sum(grid_section > 0) / grid_section.size
                        density_map.append(density)
                
                details = {
                    "area": float(area),
                    "perimeter": float(perimeter),
                    "aspect_ratio": float(aspect_ratio),
                    "circularity": float(circularity),
                    "pixel_density": float(pixel_density),
                    "contour_count": len(contours),
                    "centroid": [int(cx), int(cy)],
                    "bounding_box": [int(x), int(y), int(w), int(h)],
                    "colors": colors,
                    "density_map": density_map,
                    "has_eyes": has_eye_pair,
                    "has_mouth": has_mouth,
                    "has_nose": has_nose
                }
                
                return {
                    "drawing_type": drawing_type, 
                    "confidence": float(confidence),
                    "details": details
                }
            
            return {"drawing_type": "abstract", "confidence": 0.5, "details": {}}
        
    except Exception as e:
        print(f"Error analyzing drawing: {str(e)}")
        return {"drawing_type": "abstract", "confidence": 0.5, "details": {}}

@app.route('/analyze_drawing', methods=['POST'])
@track_ai_performance
def analyze_drawing():
    """Analyze the current drawing and provide suggestions"""
    global last_recognition, last_suggestion, last_emotion, last_colors, last_density_map
    
    try:
        data = request.get_json()
        if not data or 'canvas_data' not in data:
            return jsonify({"error": "No canvas data received"})
        
        # Decode the canvas image
        canvas_data = data['canvas_data']
        if ',' in canvas_data:
            image_data = canvas_data.split(',')[1]
            img_data = base64.b64decode(image_data)
            
            # Save a copy for recognition
            analysis_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'analysis.png')
            image = Image.open(BytesIO(img_data))
            image.save(analysis_filename)
            
            # Analyze the drawing using our custom function
            analysis_result = analyze_drawing_content(analysis_filename)
            drawing_type = analysis_result["drawing_type"]
            
            # Update last recognition
            last_recognition = drawing_type
            print(f"Drawing recognized as: {last_recognition}")
            
            # Extract dominant colors
            if 'details' in analysis_result and 'colors' in analysis_result['details']:
                last_colors = analysis_result['details']['colors']
                
            # Extract density map
            if 'details' in analysis_result and 'density_map' in analysis_result['details']:
                last_density_map = analysis_result['details']['density_map']
            
            # Create a detailed context for the AI
            drawing_context = {
                "type": drawing_type,
                "colors": last_colors,
                "density_map": last_density_map,
                "details": analysis_result.get("details", {})
            }
            
            # Get suggestions using a properly tracked method
            suggestion_result = get_art_suggestion_with_tracking(drawing_context, last_emotion, last_suggestion)
            
            last_suggestion = suggestion_result["suggestion"]
            
            return jsonify({
                "recognized": drawing_type,
                "emotion": last_emotion,
                "suggestion": last_suggestion,
                "analysis": analysis_result
            })
        else:
            return jsonify({"error": "Invalid canvas data format"})
        
    except Exception as e:
        print(f"Error analyzing drawing: {str(e)}")
        return jsonify({"error": str(e)})

@track_ai_performance
def get_art_suggestion_with_tracking(drawing_context, emotion, previous_suggestions=None):
    """Wrapper function to ensure proper tracking of Gemini API calls"""
    return advisor.get_detailed_art_suggestions(drawing_context, emotion, previous_suggestions)

@app.route('/performance_metrics', methods=['GET'])
def get_performance_metrics():
    """Get current performance metrics"""
    try:
        metrics = performance_tracker.get_current_metrics()
        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/gemini_metrics', methods=['GET'])
def get_gemini_metrics():
    """Get specific Gemini metrics with more details"""
    try:
        with performance_tracker.lock:
            gemini = performance_tracker.metrics["gemini"]
            # Convert deque objects to lists for JSON serialization
            metrics = {
                "latency": list(gemini["latency"]),
                "tokens_per_sec": list(gemini["tokens_per_sec"]),
                "input_size": list(gemini["input_size"]),
                "output_size": list(gemini["output_size"]),
                "calls": gemini["calls"],
                "errors": gemini["errors"],
                "avg_latency": float(np.mean(list(gemini["latency"]))) if gemini["latency"] else 0,
                "avg_tokens_per_sec": float(np.mean(list(gemini["tokens_per_sec"]))) if gemini["tokens_per_sec"] else 0,
                "error_rate": float(gemini["errors"] / gemini["calls"]) if gemini["calls"] > 0 else 0
            }
        
        print(f"Returning Gemini metrics: {metrics['calls']} calls, {len(metrics['latency'])} latency samples")
        
        return jsonify({
            "gemini": metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error getting Gemini metrics: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/face_metrics', methods=['GET'])
def get_face_metrics():
    """Get specific Face Recognition metrics with more details"""
    try:
        with performance_tracker.lock:
            face = performance_tracker.metrics["face_recognition"]
            # Convert deque objects to lists for JSON serialization
            face_metrics = {
                "latency": list(face["latency"]),
                "fps": list(face["fps"]),
                "confidence": list(face["confidence"]),
                "faces_detected": list(face["faces_detected"]),
                "calls": face["calls"],
                "errors": face["errors"],
                "avg_latency": float(np.mean(list(face["latency"]))) if face["latency"] else 0,
                "avg_fps": float(np.mean(list(face["fps"]))) if face["fps"] else 0,
                "avg_confidence": float(np.mean(list(face["confidence"]))) if face["confidence"] else 0,
                "avg_faces": float(np.mean(list(face["faces_detected"]))) if face["faces_detected"] else 0,
                "total_calls": face["calls"],
                "error_rate": float(face["errors"] / face["calls"]) if face["calls"] > 0 else 0
            }
        
        print(f"Returning Face Recognition metrics: {face_metrics['calls']} calls, {len(face_metrics['latency'])} latency samples")
        
        return jsonify({
            "face_recognition": face_metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error getting Face Recognition metrics: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/classification_metrics', methods=['GET'])
def get_classification_metrics():
    """Get classification metrics including confusion matrix, precision, recall, and F1 score"""
    try:
        metrics = performance_tracker.get_classification_metrics()
        return jsonify({
            "classification": metrics,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        print(f"Error getting classification metrics: {str(e)}")
        return jsonify({"error": str(e)})

# Use a function directly rather than the decorator to avoid name conflicts
@app.route('/test_gemini_api', methods=['GET'])
def test_gemini_api_endpoint():
    """Test endpoint to verify Gemini metrics tracking"""
    # Wrap this manually instead of using the decorator
    # to avoid naming conflicts
    start_time = time.time()
    input_text = "test:happy"
    tracking_info = performance_tracker.track_gemini_request(input_text)
    
    try:
        result = advisor.get_art_suggestions("test", "happy")
        
        # Get the response text for metrics
        response_text = ""
        if isinstance(result, dict) and "suggestion" in result:
            response_text = result["suggestion"]
            print(f"Got suggestion with {len(response_text)} characters")
        else:
            response_text = str(result)
            print(f"Got non-dictionary result: {type(result)}")
        
        # Manually track the response
        print(f"Manual tracking - latency: {time.time() - start_time:.2f}s, response length: {len(response_text)}")
        performance_tracker.track_gemini_response(tracking_info, response_text)
        
        # Check that metrics were actually updated
        with performance_tracker.lock:
            metrics_count = len(performance_tracker.metrics["gemini"]["latency"])
            print(f"Updated metrics count: {metrics_count}")
        
        return jsonify({
            "status": "success",
            "message": "Gemini API call tracked",
            "result": result,
            "metrics_count": metrics_count
        })
    except Exception as e:
        print(f"Error during test_gemini_api: {str(e)}")
        performance_tracker.track_gemini_response(tracking_info, None, error=str(e))
        return jsonify({
            "status": "error",
            "message": str(e)
        })

@app.route('/debug_metrics')
def debug_metrics_page():
    """Debug page for metrics"""
    return render_template('debug_metrics.html')

@app.route('/metrics_dashboard')
def metrics_dashboard_page():
    """View metrics dashboard directly from the main app"""
    # Import the function from main.py
    from main import get_dashboard_html
    return get_dashboard_html()

if __name__ == '__main__':
    app.run(debug=True, port=5002, use_reloader=False)