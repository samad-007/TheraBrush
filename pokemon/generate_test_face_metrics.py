"""
Script to generate test face recognition metrics to ensure they display properly.
"""
import time
import random
from performance_metrics import performance_tracker

def generate_test_face_metrics(num_calls=5):
    """Generate test metrics for Face Recognition"""
    print(f"Generating {num_calls} test Face Recognition metrics...")
    
    for i in range(num_calls):
        # Track request
        tracking_info = performance_tracker.track_face_recognition_request(image=None)
        
        # Simulate processing time
        time.sleep(random.uniform(0.2, 0.8))
        
        # Generate random number of faces
        num_faces = random.randint(0, 2)
        
        # Generate random confidence
        confidence = random.uniform(0.7, 0.95) if num_faces > 0 else 0
        
        # Track response
        performance_tracker.track_face_recognition_response(
            tracking_info, 
            faces=num_faces,
            confidence=confidence
        )
        
        print(f"Generated test metric {i+1}/{num_calls}")
    
    # Get current metrics
    metrics = performance_tracker.get_current_metrics()
    face_metrics = metrics.get('face_recognition', {})
    
    print("\nGenerated Face Recognition metrics:")
    print(f"Total calls: {face_metrics.get('total_calls', 0)}")
    print(f"Avg latency: {face_metrics.get('avg_latency', 0):.4f} seconds")
    print(f"Avg FPS: {face_metrics.get('avg_fps', 0):.2f}")
    print(f"Avg confidence: {face_metrics.get('avg_confidence', 0):.2f}")
    print(f"Avg faces: {face_metrics.get('avg_faces', 0):.2f}")
    
    print("\nMetrics have been generated. Go to http://localhost:8000 to view them.")
    
if __name__ == "__main__":
    print("TheraBrush Test Face Recognition Metrics Generator")
    print("=================================================")
    
    # Generate test face metrics
    generate_test_face_metrics(num_calls=10)
    
    # Also generate some with errors to test error rate display
    print("\nGenerating a few error samples...")
    for i in range(3):
        tracking_info = performance_tracker.track_face_recognition_request()
        time.sleep(0.1)
        performance_tracker.track_face_recognition_response(tracking_info, faces=None, error="Test error")
    
    print("Done! Metrics dashboard should now display both successful and error metrics.")
