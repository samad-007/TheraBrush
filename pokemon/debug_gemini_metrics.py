"""
Helper script to debug and verify Gemini metrics.
Run this directly to check if metrics are being properly tracked.
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Import the performance tracker
from performance_metrics import performance_tracker, track_ai_performance
from chatgpt_advisor import ChatGPTAdvisor

def check_metrics_file():
    """Check if the metrics file exists and print contents"""
    log_file = 'performance_logs.json'
    
    if os.path.exists(log_file):
        print(f"Metrics file {log_file} exists, size: {os.path.getsize(log_file)} bytes")
        try:
            with open(log_file, 'r') as f:
                data = json.load(f)
                
            # Print summary of Gemini metrics
            if 'gemini' in data:
                gemini = data['gemini']
                print("\n--- GEMINI METRICS SUMMARY ---")
                print(f"Total calls recorded: {gemini.get('calls', 0)}")
                print(f"Total errors recorded: {gemini.get('errors', 0)}")
                
                # Check if we have latency samples
                latency = gemini.get('latency', [])
                if latency:
                    print(f"Latency samples: {len(latency)}")
                    print(f"Average latency: {sum(latency)/len(latency):.4f} seconds")
                    print(f"Min latency: {min(latency):.4f}, Max latency: {max(latency):.4f}")
                else:
                    print("No latency samples recorded")
                
                # Check token rate samples
                tps = gemini.get('tokens_per_sec', [])
                if tps:
                    print(f"Token rate samples: {len(tps)}")
                    print(f"Average tokens/sec: {sum(tps)/len(tps):.2f}")
                else:
                    print("No token rate samples recorded")
                
                # Check call timestamps if available
                if 'timestamps' in gemini:
                    print(f"Last recorded call: {gemini['timestamps'][-1]}")
            else:
                print("\nNo Gemini metrics found in the file")
                
            # Check last updated timestamp
            print(f"\nLast updated: {data.get('last_updated', 'unknown')}")
        except Exception as e:
            print(f"Error reading metrics file: {str(e)}")
    else:
        print(f"Metrics file {log_file} not found")

@track_ai_performance
def test_gemini_call():
    """Make a test call to Gemini API and verify metrics are tracked"""
    print("\nMaking test call to Gemini API...")
    advisor = ChatGPTAdvisor()
    
    # Track start time manually to verify metrics
    start_time = time.time()
    
    # Make a call to Gemini
    result = advisor.get_art_suggestions("test_drawing", "happy", "previous suggestion")
    
    # Calculate actual latency
    actual_latency = time.time() - start_time
    print(f"Call completed in {actual_latency:.2f} seconds")
    
    # Print result summary
    if isinstance(result, dict) and "suggestion" in result:
        suggestion = result["suggestion"]
        print(f"Received suggestion with {len(suggestion)} characters")
    else:
        print(f"Received non-standard result: {type(result)}")
    
    # Wait for metrics to be updated
    time.sleep(1)
    
    # Get current metrics
    metrics = performance_tracker.get_current_metrics()
    if 'gemini' in metrics:
        gemini = metrics['gemini']
        print("\n--- CURRENT METRICS ---")
        print(f"Average latency: {gemini['avg_latency']:.4f} seconds")
        print(f"Average tokens/sec: {gemini['avg_tokens_per_sec']:.2f}")
        print(f"Total calls: {gemini['total_calls']}")
        print(f"Error rate: {gemini['error_rate']:.2%}")
    else:
        print("No Gemini metrics available in current metrics")
    
    return result

def plot_metrics():
    """Plot metrics from the metrics file"""
    log_file = 'performance_logs.json'
    
    if not os.path.exists(log_file):
        print("No metrics file to plot")
        return
    
    with open(log_file, 'r') as f:
        data = json.load(f)
    
    if 'gemini' not in data or not data['gemini'].get('latency'):
        print("No Gemini metrics to plot")
        return
    
    gemini = data['gemini']
    latency = gemini.get('latency', [])
    tps = gemini.get('tokens_per_sec', [])
    
    plt.figure(figsize=(12, 6))
    
    # Plot latency
    plt.subplot(1, 2, 1)
    plt.plot(latency)
    plt.title('Gemini API Latency')
    plt.xlabel('Call #')
    plt.ylabel('Latency (seconds)')
    plt.grid(True)
    
    # Plot tokens per second
    plt.subplot(1, 2, 2)
    plt.plot(tps)
    plt.title('Gemini API Tokens Per Second')
    plt.xlabel('Call #')
    plt.ylabel('Tokens/sec')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('gemini_metrics.png')
    print("Metrics plots saved to gemini_metrics.png")

if __name__ == "__main__":
    print("Debugging Gemini metrics...")
    
    # Check if metrics file exists and print contents
    check_metrics_file()
    
    # Ask if user wants to make a test call
    choice = input("\nDo you want to make a test call to Gemini API? (y/n): ")
    if choice.lower() == 'y':
        test_gemini_call()
        
        # Check metrics file again after the call
        print("\nChecking metrics file after test call:")
        check_metrics_file()
        
        # Generate plots
        try:
            plot_metrics()
        except Exception as e:
            print(f"Error plotting metrics: {str(e)}")
    
    print("\nDone debugging Gemini metrics")
