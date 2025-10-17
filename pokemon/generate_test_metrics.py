"""
Script to generate test metrics to ensure metrics are being tracked and displayed.
"""
import time
import random
from performance_metrics import performance_tracker, track_ai_performance

@track_ai_performance
def generate_test_gemini_metrics(input_text="test input", num_calls=5):
    """Generate test metrics for Gemini"""
    print(f"Generating {num_calls} test Gemini metrics...")
    
    for i in range(num_calls):
        # Track request
        tracking_info = performance_tracker.track_gemini_request(input_text)
        
        # Simulate processing time
        time.sleep(random.uniform(0.5, 1.5))
        
        # Generate random response
        response_text = f"Test response {i+1} with {random.randint(50, 200)} characters" * 3
        
        # Track response
        performance_tracker.track_gemini_response(tracking_info, response_text)
        
        print(f"Generated test metric {i+1}/{num_calls}")
    
    # Get current metrics
    metrics = performance_tracker.get_current_metrics()
    gemini_metrics = metrics.get('gemini', {})
    
    print("\nGenerated Gemini metrics:")
    print(f"Total calls: {gemini_metrics.get('total_calls', 0)}")
    print(f"Avg latency: {gemini_metrics.get('avg_latency', 0):.4f} seconds")
    print(f"Avg tokens/sec: {gemini_metrics.get('avg_tokens_per_sec', 0):.2f}")
    
    print("\nMetrics have been generated. Go to http://localhost:8000 to view them.")
    
if __name__ == "__main__":
    print("TheraBrush Test Metrics Generator")
    print("================================")
    
    # Generate test Gemini metrics
    generate_test_gemini_metrics(num_calls=10)
