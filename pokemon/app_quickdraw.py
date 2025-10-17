"""
TheraBrush Flask Application
Using Quick Draw CNN for drawing recognition
Based on: https://dev.to/larswaechter/recognizing-hand-drawn-doodles-using-deep-learning-ki0
"""

from flask import Flask, render_template, request, jsonify
import os
import base64
from io import BytesIO
from PIL import Image
import time
import json
from dotenv import load_dotenv
from chatgpt_advisor import ChatGPTAdvisor
from performance_metrics import performance_tracker, track_ai_performance

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
    Expects JSON with 'strokes' and 'box' (bounding box).
    """
    global last_recognition
    
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
        
        # Get strokes and bounding box from canvas
        strokes = data.get('strokes', [])
        bounding_box = data.get('box', [0, 0, 500, 500])
        
        if not strokes:
            return jsonify({
                "recognized_shape": "empty",
                "confidence": 0.0,
                "top_predictions": []
            })
        
        print(f"\n{'='*60}")
        print(f"🎨 ANALYZING DRAWING")
        print(f"{'='*60}")
        print(f"Strokes: {len(strokes)}")
        print(f"Bounding box: {bounding_box}")
        
        # Use Quick Draw CNN to recognize the drawing
        predictions = recognizer.predict(strokes, bounding_box, top_k=3)
        
        if predictions:
            top_prediction = predictions[0]
            last_recognition = top_prediction['class_name']
            
            print(f"\n🎯 RECOGNITION RESULTS:")
            print(f"{'='*60}")
            for i, pred in enumerate(predictions, 1):
                print(f"{i}. {pred['class_name']}: {pred['confidence']:.2f}%")
            print(f"{'='*60}\n")
            
            return jsonify({
                "recognized_shape": top_prediction['class_name'],
                "confidence": top_prediction['confidence'],
                "top_predictions": predictions
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
        # Generate suggestion using Gemini
        suggestion_result = advisor.get_therapeutic_suggestion(
            emotion=last_emotion,
            drawing_type=last_recognition,
            previous_suggestion=last_suggestion
        )
        
        # Track AI performance
        track_ai_performance(
            model_used="gemini",
            response_time=0.5,  # Placeholder
            success=True
        )
        
        last_suggestion = suggestion_result.get("suggestion", "")
        
        print(f"\n💡 AI SUGGESTION:")
        print(f"{'='*60}")
        print(f"Emotion: {last_emotion}")
        print(f"Drawing: {last_recognition}")
        print(f"Suggestion: {last_suggestion[:100]}...")
        print(f"{'='*60}\n")
        
        return jsonify(suggestion_result)
    
    except Exception as e:
        print(f"❌ Error getting art suggestion: {str(e)}")
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
        metrics = performance_tracker.get_summary()
        return jsonify(metrics)
    except Exception as e:
        print(f"❌ Error getting metrics: {str(e)}")
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

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("🎨 STARTING THERABRUSH APPLICATION")
    print("=" * 80)
    print("Flask Application with Quick Draw CNN")
    print("Based on: https://dev.to/larswaechter/recognizing-hand-drawn-doodles-using-deep-learning-ki0")
    print("=" * 80 + "\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5002)
