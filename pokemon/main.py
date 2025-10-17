import os
import requests
import gc
import time
import threading
import http.server
import socketserver
from PIL import Image
from performance_metrics import track_face_recognition_performance, performance_tracker
import json
import numpy as np
from datetime import datetime
import os

# Load API keys from environment variables
API_KEY = os.environ.get("FACEPP_API_KEY", "")
API_SECRET = os.environ.get("FACEPP_API_SECRET", "")
FACEPP_URL = "https://api-us.faceplusplus.com/facepp/v3/detect"
IMAGE_PATH = './uploads/image.png'
TEXT_FILE_PATH = './emotions.txt'

@track_face_recognition_performance
def detect_emotions(image_path):
    """ Sends image to Face++ API, extracts emotion data, saves top 2 emotions to a text file """
    if not API_KEY or not API_SECRET:
        return {"error": "API keys missing! Set FACEPP_API_KEY and FACEPP_API_SECRET in env vars."}
    
    try:
        # Check if file exists and has content
        if not os.path.exists(image_path):
            print(f"Image file not found: {image_path}")
            return {"error": "Image file not found"}
            
        # Check file size to ensure it's not empty
        file_size = os.path.getsize(image_path)
        if file_size == 0:
            print(f"Image file is empty: {image_path}")
            return {"error": "Image file is empty"}
            
        print(f"Processing image: {image_path}, size: {file_size} bytes")
        
        # Check if it's a valid image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                print(f"Image dimensions: {width}x{height}")
        except Exception as e:
            print(f"Invalid image file: {str(e)}")
            return {"error": f"Invalid image file: {str(e)}"}
            
        with open(image_path, 'rb') as image_file:
            files = {'image_file': image_file}
            data = {
                'api_key': API_KEY,
                'api_secret': API_SECRET,
                'return_attributes': 'emotion'
            }
            
            print("Sending request to Face++ API...")
            response = requests.post(FACEPP_URL, data=data, files=files)
            
        if response.status_code == 200:
            result = response.json()
            print(f"API response: {result}")
            
            if 'faces' in result and result['faces']:
                emotions = result['faces'][0]['attributes']['emotion']
                top_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)[:2]
                
                print(f"Top emotions detected: {top_emotions}")
                
                # Extract confidence for performance metrics
                confidence = max(emotions.values()) / 100.0  # Normalize to 0-1 range
                
                # Map Face++ emotions to our frontend emotion names if needed
                emotion_mapping = {
                    'happiness': 'happiness',
                    'neutral': 'neutral',
                    'surprise': 'surprise', 
                    'sadness': 'sadness',
                    'anger': 'anger',
                    'disgust': 'disgust',
                    'fear': 'fear',
                    'contempt': 'contempt'
                }
                
                with open(TEXT_FILE_PATH, 'w') as file:
                    for emotion, value in top_emotions:
                        # Use consistent emotion names
                        emotion_name = emotion_mapping.get(emotion, emotion)
                        file.write(f"{emotion_name}: {value:.2f}\n")

                # Return with confidence value for performance tracking
                return {
                    "message": "Emotions updated!", 
                    "emotions": dict(top_emotions),
                    "confidence": confidence,
                    "faces": 1  # Changed from 'faces_detected' to 'faces'
                }
            else:
                print("No face detected in the image")
                with open(TEXT_FILE_PATH, 'w') as file:
                    file.write(f"neutral: 99.00\n")
                return {
                    "error": "No face detected",
                    "confidence": 0,
                    "faces": 0  # Changed from 'faces_detected' to 'faces'
                }
        else:
            print(f"API Error {response.status_code}: {response.text}")
            return {"error": f"API Error {response.status_code}", "details": response.text}
    
    except Exception as e:
        print(f"Exception in detect_emotions: {str(e)}")
        return {"error": f"Exception: {str(e)}"}
    finally:
        gc.collect()

# Add a helper function at the top of the file to import traceback
def import_traceback():
    import traceback
    return traceback

# Create HTTP server for performance metrics dashboard
class MetricsDashboardHandler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            # Serve the metrics dashboard HTML
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(get_dashboard_html().encode())
        elif self.path == '/metrics':
            # Serve the JSON metrics data
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            metrics = performance_tracker.get_current_metrics()
            self.wfile.write(json.dumps(metrics).encode())
        elif self.path == '/gemini_metrics':
            # Add direct support for gemini metrics endpoint
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                with performance_tracker.lock:
                    gemini = performance_tracker.metrics["gemini"]
                    # Convert deque objects to lists for JSON serialization
                    metrics = {
                        "gemini": {
                            "latency": list(gemini["latency"]),
                            "tokens_per_sec": list(gemini["tokens_per_sec"]),
                            "input_size": list(gemini["input_size"]),
                            "output_size": list(gemini["output_size"]),
                            "calls": gemini["calls"],
                            "errors": gemini["errors"],
                            "avg_latency": float(np.mean(list(gemini["latency"]))) if gemini["latency"] else 0,
                            "avg_tokens_per_sec": float(np.mean(list(gemini["tokens_per_sec"]))) if gemini["tokens_per_sec"] else 0,
                            "error_rate": gemini["errors"] / gemini["calls"] if gemini["calls"] > 0 else 0
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                
                # Convert to JSON and handle any potential serialization errors
                json_response = json.dumps(metrics, default=lambda x: float(x) if isinstance(x, np.float32) or isinstance(x, np.float64) else str(x))
                self.wfile.write(json_response.encode())
            except Exception as e:
                error_response = {"error": str(e), "traceback": str(import_traceback().format_exc())}
                self.wfile.write(json.dumps(error_response).encode())
        elif self.path == '/test_gemini_api':
            # Add endpoint to trigger a Gemini API test
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            
            try:
                # Import locally to avoid circular imports
                from chatgpt_advisor import ChatGPTAdvisor
                
                # Create tracking info
                start_time = time.time()
                input_text = "test:happy"
                tracking_info = performance_tracker.track_gemini_request(input_text)
                
                # Make the API call
                advisor = ChatGPTAdvisor()
                result = advisor.get_art_suggestions("test", "happy")
                
                # Get response text
                response_text = ""
                if isinstance(result, dict) and "suggestion" in result:
                    response_text = result["suggestion"]
                else:
                    response_text = str(result)
                
                # Track the response
                performance_tracker.track_gemini_response(tracking_info, response_text)
                
                # Return success
                response = {
                    "status": "success",
                    "message": "Gemini API call tracked",
                    "result": result
                }
                self.wfile.write(json.dumps(response).encode())
            except Exception as e:
                error_response = {"status": "error", "message": str(e)}
                self.wfile.write(json.dumps(error_response).encode())
        else:
            # Default handler for any other paths
            super().do_GET()
    
    def log_message(self, format, *args):
        # Suppress HTTP server logs to avoid cluttering console
        return

def get_dashboard_html():
    """Generate HTML for the metrics dashboard"""
    # Instead of the inline HTML, read from file
    try:
        with open('templates/metrics_dashboard.html', 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Fall back to the original inline HTML if file not found
        print("Warning: metrics_dashboard.html not found, using inline HTML")
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>TheraBrush Performance Metrics</title>
            <style>
                body {
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }
                /* ... rest of inline CSS ... */
            </style>
        </head>
        <body>
            <!-- ... rest of inline HTML ... -->
        </body>
        </html>
        """

def start_metrics_server(port=8000):
    """Start HTTP server for metrics dashboard"""
    try:
        Handler = MetricsDashboardHandler
        httpd = socketserver.TCPServer(("", port), Handler)
        print(f"Starting metrics dashboard server on port {port}")
        metrics_thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        metrics_thread.start()
        return httpd
    except Exception as e:
        print(f"Error starting metrics server: {str(e)}")
        return None

if __name__ == "__main__":
    print("Starting emotion detection loop...")
    
    # Start metrics dashboard server
    metrics_server = start_metrics_server(port=8000)
    print("Metrics dashboard available at http://localhost:8000")
    
    last_modified_time = 0
    
    while True:
        try:
            # Check if image file has been modified since last check
            if os.path.exists(IMAGE_PATH):
                current_modified_time = os.path.getmtime(IMAGE_PATH)
                
                if current_modified_time > last_modified_time:
                    print(f"New image detected at {time.strftime('%H:%M:%S')}")
                    result = detect_emotions(IMAGE_PATH)
                    print(result)
                    last_modified_time = current_modified_time
                else:
                    # Print less noise
                    print(".", end="", flush=True)
            else:
                print(f"Waiting for image at {IMAGE_PATH}...")
                
        except Exception as e:
            print(f"Error in main loop: {str(e)}")
            
        time.sleep(2)