"""
Script to generate test classification metrics to display in the dashboard.
"""
import time
import random
from performance_metrics import performance_tracker

def generate_test_classification_metrics(num_samples=50):
    """Generate test classification metrics"""
    print(f"Generating {num_samples} test classification results...")
    
    # Define some classes that will be used for classification
    classes = ['circle', 'square', 'triangle', 'rectangle', 'face', 'sun', 'house']
    
    # Generate random classification results with some predictable patterns
    for i in range(num_samples):
        # Choose a random true label
        true_label = random.choice(classes)
        
        # For 70% of cases, predict correctly
        if random.random() < 0.7:
            pred_label = true_label
        else:
            # For confusable classes, make specific errors
            if true_label == 'circle':
                pred_label = random.choice(['face', 'sun', 'circle'])
            elif true_label == 'rectangle':
                pred_label = random.choice(['square', 'rectangle'])
            elif true_label == 'face':
                pred_label = random.choice(['circle', 'face'])
            else:
                # Otherwise choose a random different class
                other_classes = [c for c in classes if c != true_label]
                pred_label = random.choice(other_classes)
        
        # Track the result
        performance_tracker.track_classification_result(true_label, pred_label)
        
        print(f"Generated result {i+1}/{num_samples}: true={true_label}, pred={pred_label}")
        
        # Small delay to simulate processing time
        time.sleep(0.01)
    
    # Get the metrics
    metrics = performance_tracker.get_classification_metrics()
    
    print("\nGenerated Classification Metrics:")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Average precision: {metrics['avg_precision']:.4f}")
    print(f"Average recall: {metrics['avg_recall']:.4f}")
    print(f"Average F1-score: {metrics['avg_f1']:.4f}")
    
    print("\nPer-class metrics:")
    for label in metrics['labels']:
        if label in metrics['precision']:
            print(f"  {label}: precision={metrics['precision'][label]:.4f}, " +
                  f"recall={metrics['recall'][label]:.4f}, " +
                  f"F1={metrics['f1_score'][label]:.4f}")
    
    print("\nConfusion Matrix:")
    print("True \\ Pred | " + " | ".join(metrics['labels']))
    print("-" * (10 + 10 * len(metrics['labels'])))
    
    for i, true_label in enumerate(metrics['labels']):
        row = [true_label.ljust(10)]
        for j, pred_label in enumerate(metrics['labels']):
            row.append(str(metrics['confusion_matrix'][i][j]).center(10))
        print(" | ".join(row))
    
    print("\nMetrics have been generated. Go to http://localhost:8000 to view them in the dashboard.")

if __name__ == "__main__":
    print("TheraBrush Test Classification Metrics Generator")
    print("===============================================")
    
    # Generate test classification metrics
    generate_test_classification_metrics(num_samples=100)
    
    print("Done! Metrics dashboard should now display classification metrics.")
