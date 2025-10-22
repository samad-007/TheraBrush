
from flask import Flask, render_template, request, jsonify
import os
import base64
from io import BytesIO
from PIL import Image
import time
import json
import numpy as np
from dotenv import load_dotenv
from chatgpt_advisor import ChatGPTAdvisor
from performance_metrics import performance_tracker

# Load environment variables from .env file
load_dotenv()

# Import the Quick Draw recognizer
try:
    from quickdraw_recognizer import get_recognizer
    RECOGNIZER_AVAILABLE = True
    print("✅ Quick Draw recognizer module imported successfully")
except ImportError as e:
    RECOGNIZER_AVAILABLE = False
    print(f"❌ Quick Draw recognizer not available: {str(e)}")
    print("💡 Please run 'python train_quickdraw_model.py' to train the model")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
TEXT_FILE_PATH = os.path.abspath(os.path.join(os.getcwd(), "./emotions.txt"))

# Initialize advisor
advisor = ChatGPTAdvisor()

# Initialize Quick Draw recognizer
recognizer = None
if RECOGNIZER_AVAILABLE:
    try:
        print("\n" + "=" * 80)
        print("INITIALIZING QUICK DRAW CNN RECOGNIZER")
        print("=" * 80)
        recognizer = get_recognizer()
        
        if recognizer and recognizer.model_loaded:
            print("=" * 80)
            print("🎨 THERABRUSH DRAWING RECOGNITION SYSTEM")
            print("=" * 80)
            print("✅ Quick Draw CNN model loaded successfully")
            print(f"   - Classes: {len(recognizer.class_names)}")
            print(f"   - Model: {recognizer.model_path}")
            print("   - Architecture: Based on dev.to article methodology")
            print("=" * 80 + "\n")
        else:
            print("=" * 80)
            print("❌ Quick Draw model not loaded")
            print("💡 Please train the model first:")
            print("   1. python quickdraw_dataset.py sample  (for testing)")
            print("   2. python train_quickdraw_model.py")
            print("=" * 80 + "\n")
    except Exception as e:
        print(f"Error initializing recognizer: {str(e)}")
        recognizer = None

# Store the latest state
last_recognition = "abstract"
last_suggestion = None
last_emotion = "neutral"

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    """
    Receives base64 image, decodes it, and saves it.
    """
    try:
        data = request.get_json()
        if not data or 'image' not in data:
            return jsonify({"error": "No image data received"})
        
        # Handle 'data:image/png;base64,' prefix
        if ',' in data['image']:
            image_data = data['image'].split(',')[1]
        else:
            image_data = data['image']
        
        img_data = base64.b64decode(image_data)
        filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
        
        image = Image.open(BytesIO(img_data))
        image.save(filename)
        
        print(f"✅ Image saved successfully to {filename} at {time.strftime('%H:%M:%S')}")
        return jsonify({"message": "Image saved successfully!"})
    
    except Exception as e:
        print(f"❌ Error saving image: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/get_emotions', methods=['GET'])
def get_emotions():
    """
    Reads emotions.txt and returns the content.
    """
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
        print(f"❌ Error getting emotions: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/analyze_drawing', methods=['POST'])
def analyze_drawing():
    """
    Analyze the canvas drawing using Quick Draw CNN.
    Expects JSON with 'canvas_data' (base64 PNG) OR 'strokes' and 'box' (bounding box).
    """
    global last_recognition, last_emotion, last_suggestion
    
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No data received"})
        
        # Check if recognizer is available
        if not recognizer or not recognizer.model_loaded:
            return jsonify({
                "recognized_shape": "unknown",
                "confidence": 0.0,
                "top_predictions": [],
                "error": "Model not loaded. Please train the Quick Draw model first."
            })
        
        print(f"\n{'='*60}")
        print(f"🎨 ANALYZING DRAWING")
        print(f"{'='*60}")
        print(f"🔍 Data keys: {list(data.keys())}")
        
        # Handle PNG image data from canvas
        if 'canvas_data' in data:
            import base64
            from PIL import Image
            import io
            
            # Decode base64 image
            canvas_data = data['canvas_data']
            if ',' in canvas_data:
                canvas_data = canvas_data.split(',')[1]  # Remove data:image/png;base64, prefix
            
            print(f"📊 Received canvas_data (base64 PNG)")
            print(f"📏 Base64 length: {len(canvas_data)} characters")
            
            # Convert to PIL Image
            img_bytes = base64.b64decode(canvas_data)
            img = Image.open(io.BytesIO(img_bytes))
            print(f"🖼️  Image loaded: size={img.size}, mode={img.mode}")
            
            # SIMPLIFIED PREPROCESSING - Match Quick Draw expectations exactly
            # Quick Draw uses BLACK STROKES on WHITE BACKGROUND (not inverted!)
            
            # Step 1: Handle RGBA - composite onto WHITE background
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
                print(f"🔄 Converted RGBA to RGB with white background")
            
            # Step 2: Convert to grayscale
            img = img.convert('L')
            img_array = np.array(img, dtype=np.float32)
            print(f"📊 Original: mean={img_array.mean():.1f}, min={img_array.min():.1f}, max={img_array.max():.1f}")
            
            # Step 3: Boost contrast for faint drawings
            if img_array.min() > 150:
                # Very faint drawing, stretch to full range
                old_min = img_array.min()
                old_max = img_array.max()
                if old_max > old_min:
                    img_array = ((img_array - old_min) / (old_max - old_min)) * 255
                    print(f"🔆 Contrast boost: {old_min:.1f}-{old_max:.1f} → 0-255")
            
            # Step 4: DO NOT INVERT - Quick Draw uses black strokes on white background!
            # Canvas has white bg with dark strokes = CORRECT format
            # Mean should be high (white background), strokes should be dark/black
            print(f"✅ Keeping original orientation: black strokes on white background (Quick Draw format)")
            
            # Step 5: Resize to intermediate size first (100x100) to preserve details
            img = Image.fromarray(img_array.astype(np.uint8))
            img = img.resize((100, 100), Image.Resampling.LANCZOS)
            print(f"🔄 Resized to 100x100 (intermediate)")
            
            # Step 6: Apply light Gaussian blur for smoothing
            img_array = np.array(img, dtype=np.float32)
            from scipy.ndimage import gaussian_filter
            img_array = gaussian_filter(img_array, sigma=0.5)
            print(f"🔧 Applied Gaussian smoothing")
            
            # Step 7: Normalize AFTER Gaussian (it reduces max values)
            if img_array.max() > 0:
                img_array = (img_array / img_array.max()) * 255.0
                print(f"🔆 Re-normalized after blur: max → 255")
            
            # Step 8: Final resize to 28x28 (model input size)
            img = Image.fromarray(img_array.astype(np.uint8))
            img = img.resize((28, 28), Image.Resampling.LANCZOS)
            print(f"🔄 Final resize to 28x28")
            
            # Step 9: Final array for model
            img_array = np.array(img, dtype=np.float32)
            
            print(f"📊 Final: shape={img_array.shape}, mean={img_array.mean():.1f}, min={img_array.min():.1f}, max={img_array.max():.1f}, non-zero={np.count_nonzero(img_array)}/{img_array.size}")
            
            # Save debug image to see what model receives
            debug_img = Image.fromarray(img_array.astype(np.uint8))
            debug_path = os.path.join('uploads', 'debug_drawing_28x28.png')
            debug_img.save(debug_path)
            print(f"💾 Saved debug image: {debug_path}")
            
            # Add batch and channel dimensions
            img_array = np.expand_dims(img_array, axis=(0, -1))
            print(f"🔢 Final shape: {img_array.shape}")
            
            # Get predictions directly
            print(f"🤖 Running model.predict()...")
            predictions = recognizer.model.predict(img_array, verbose=0)
            print(f"✅ Raw predictions shape: {predictions.shape}")
            print(f"📊 Prediction stats: min={predictions.min():.6f}, max={predictions.max():.6f}, sum={predictions.sum():.6f}")
            
            top_k = 3
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            print(f"🏆 Top {top_k} indices: {top_indices}")
            
            results = []
            for idx in top_indices:
                class_name = recognizer.class_names[idx]
                probability = float(predictions[0][idx])
                confidence = float(predictions[0][idx] * 100)
                results.append({
                    'class_name': class_name,
                    'probability': probability,
                    'confidence': confidence
                })
                print(f"  #{len(results)}: {class_name} - {confidence:.2f}% (prob: {probability:.6f})")
            
        # Handle stroke data (original format)
        elif 'strokes' in data:
            # Get strokes and bounding box from canvas
            strokes = data.get('strokes', [])
            bounding_box = data.get('box', [0, 0, 500, 500])
            
            if not strokes:
                return jsonify({
                    "recognized_shape": "empty",
                    "confidence": 0.0,
                    "top_predictions": []
                })
            
            print(f"📊 Strokes count: {len(strokes)}")
            print(f"📦 Bounding box: {bounding_box}")
            print(f"📝 First stroke (if available): {strokes[0] if strokes else 'None'}")
            print(f"📏 First stroke length: {len(strokes[0]) if strokes else 0} points")
            
            # Use Quick Draw CNN to recognize the drawing
            results = recognizer.predict(strokes, bounding_box, top_k=3)
        
        else:
            return jsonify({
                "error": "No canvas_data or strokes provided",
                "recognized_shape": "error",
                "confidence": 0.0
            })
        
        if results:
            top_prediction = results[0]
            last_recognition = top_prediction['class_name']
            
            print(f"\n🎯 RECOGNITION RESULTS:")
            print(f"{'='*60}")
            for i, pred in enumerate(results, 1):
                print(f"{i}. {pred['class_name']}: {pred['confidence']:.2f}%")
            print(f"{'='*60}\n")
            
            # Generate AI suggestion immediately after recognition
            try:
                print(f"🤖 Generating Gemini AI suggestion...")
                start_time = time.time()
                suggestion_result = advisor.get_art_suggestions(
                    drawing_type=last_recognition,
                    emotion=last_emotion,
                    previous_suggestions=[last_suggestion] if last_suggestion else None
                )
                
                # Track AI performance
                response_time = time.time() - start_time
                tracking_info = performance_tracker.track_gemini_request(
                    input_text=f"{last_emotion}:{last_recognition}"
                )
                
                # Extract suggestion from result
                if isinstance(suggestion_result, dict):
                    suggestion_text = suggestion_result.get("suggestion", suggestion_result.get("suggestions", [{}])[0].get("text", ""))
                else:
                    suggestion_text = str(suggestion_result)
                
                last_suggestion = suggestion_text
                
                # Complete tracking
                performance_tracker.track_gemini_response(
                    tracking_info=tracking_info,
                    response_text=last_suggestion or ""
                )
                
                print(f"✅ Gemini suggestion generated ({response_time:.2f}s)")
                print(f"💡 Suggestion: {last_suggestion[:100] if last_suggestion else 'None'}...")
                
            except Exception as e:
                print(f"⚠️ Error generating suggestion: {str(e)}")
                suggestion_text = f"Keep expressing yourself through art! Your {last_recognition} drawing shows great creativity."
            
            return jsonify({
                "recognized_shape": top_prediction['class_name'],
                "recognized": top_prediction['class_name'],  # Add this for frontend compatibility
                "confidence": top_prediction['confidence'],
                "top_predictions": results,
                "suggestion": suggestion_text,  # Add Gemini suggestion
                "analysis": {  # Add analysis object for frontend
                    "drawing_type": top_prediction['class_name'],
                    "confidence": top_prediction['confidence'] / 100,
                    "details": {
                        "top_predictions": results
                    }
                }
            })
        else:
            return jsonify({
                "recognized_shape": "unknown",
                "confidence": 0.0,
                "top_predictions": []
            })
    
    except Exception as e:
        print(f"❌ Error analyzing drawing: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "recognized_shape": "error",
            "confidence": 0.0
        })

@app.route('/get_art_suggestion', methods=['GET'])
def get_art_suggestion():
    """
    Get art suggestion based on the current emotion and recognized drawing.
    """
    global last_emotion, last_suggestion, last_recognition
    
    try:
        # Generate suggestion using Gemini (use the correct method name)
        start_time = time.time()
        suggestion_result = advisor.get_art_suggestions(
            drawing_type=last_recognition,
            emotion=last_emotion,
            previous_suggestions=[last_suggestion] if last_suggestion else None
        )
        
        # Track AI performance manually
        response_time = time.time() - start_time
        tracking_info = performance_tracker.track_gemini_request(
            input_text=f"{last_emotion}:{last_recognition}"
        )
        
        # Extract suggestion from result
        if isinstance(suggestion_result, dict):
            last_suggestion = suggestion_result.get("suggestion", suggestion_result.get("suggestions", [{}])[0].get("text", ""))
        else:
            last_suggestion = str(suggestion_result)
        
        # Complete tracking
        performance_tracker.track_gemini_response(
            tracking_info=tracking_info,
            response_text=last_suggestion or ""
        )
        
        print(f"\n💡 AI SUGGESTION:")
        print(f"{'='*60}")
        print(f"Emotion: {last_emotion}")
        print(f"Drawing: {last_recognition}")
        print(f"Suggestion: {last_suggestion[:100] if last_suggestion else 'No suggestion'}...")
        print(f"{'='*60}\n")
        
        return jsonify({
            "emotion": last_emotion,
            "suggestion": last_suggestion or f"Keep expressing yourself through art. Your {last_recognition} drawing shows creativity!",
            "colors": ["#FF6B6B", "#4ECDC4", "#95E1D3", "#F38181"]
        })
    
    except Exception as e:
        print(f"❌ Error getting art suggestion: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "emotion": last_emotion,
            "suggestion": f"Keep expressing yourself through art. Your {last_recognition} drawing shows creativity!",
            "colors": ["#FF6B6B", "#4ECDC4"]
        })

@app.route('/metrics', methods=['GET'])
@app.route('/get_metrics', methods=['GET'])
def get_metrics():
    """
    Get performance metrics.
    """
    try:
        metrics = performance_tracker.get_current_metrics()
        print(f"\n📊 METRICS REQUEST:")
        print(f"Gemini calls: {metrics.get('gemini', {}).get('total_calls', 0)}")
        print(f"Face calls: {metrics.get('face_recognition', {}).get('total_calls', 0)}")
        print(f"Classification samples: {metrics.get('classification', {}).get('total_samples', 0)}")
        return jsonify(metrics)
    except Exception as e:
        print(f"❌ Error getting metrics: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)})

@app.route('/gemini_metrics', methods=['GET'])
def get_gemini_metrics():
    """Get Gemini AI metrics"""
    try:
        metrics = performance_tracker.get_current_metrics()
        return jsonify(metrics.get("gemini", {}))
    except Exception as e:
        print(f"❌ Error getting Gemini metrics: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/face_metrics', methods=['GET'])
def get_face_metrics():
    """Get Face recognition metrics"""
    try:
        metrics = performance_tracker.get_current_metrics()
        return jsonify(metrics.get("face_recognition", {}))
    except Exception as e:
        print(f"❌ Error getting face metrics: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/classification_metrics', methods=['GET'])
def get_classification_metrics():
    """Get classification metrics"""
    try:
        metrics = performance_tracker.get_classification_metrics()
        return jsonify(metrics)
    except Exception as e:
        print(f"❌ Error getting classification metrics: {str(e)}")
        return jsonify({"error": str(e)})

@app.route('/metrics_dashboard')
def metrics_dashboard():
    """
    Render the metrics dashboard.
    """
    try:
        return render_template('metrics_dashboard.html')
    except Exception as e:
        print(f"❌ Error rendering metrics dashboard: {str(e)}")
        return f"Error: {str(e)}", 500

@app.route('/test_gemini_api', methods=['GET'])
def test_gemini_api():
    """Test endpoint to verify Gemini API is working"""
    try:
        print(f"\n{'='*60}")
        print(f"🧪 TESTING GEMINI API")
        print(f"{'='*60}")
        
        # Test with a simple drawing type
        test_result = advisor.get_art_suggestions(
            drawing_type="butterfly",
            emotion="calm",
            previous_suggestions=None
        )
        
        if isinstance(test_result, dict):
            suggestion = test_result.get("suggestion", "")
            source = test_result.get("source", "unknown")
            text_length = test_result.get("text_length", 0)
            
            print(f"✅ Gemini API test successful!")
            print(f"   Source: {source}")
            print(f"   Text length: {text_length} characters")
            print(f"   Suggestion: {suggestion[:100]}...")
            print(f"{'='*60}\n")
            
            return jsonify({
                "success": True,
                "source": source,
                "text_length": text_length,
                "suggestion": suggestion
            })
        else:
            print(f"⚠️  Unexpected response format: {type(test_result)}")
            return jsonify({
                "success": False,
                "error": "Unexpected response format",
                "response": str(test_result)
            })
            
    except Exception as e:
        print(f"❌ Gemini API test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🎨 STARTING THERABRUSH APPLICATION")
    print("=" * 80)
    print("Flask Application with Quick Draw CNN")
    print("=" * 80 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5002)
