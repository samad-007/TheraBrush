"""
Performance metrics tracking for TheraBrush application.
Tracks and logs performance metrics for AI models used in the application.
"""

import time
import threading
import json
import os
import psutil
import numpy as np
from datetime import datetime
from collections import deque

class PerformanceTracker:
    def __init__(self, log_file='performance_logs.json', max_samples=100):
        self.log_file = log_file
        self.max_samples = max_samples
        
        # Initialize metrics storage with deque for efficient fixed-size collection
        self.metrics = {
            "gemini": {
                "latency": deque(maxlen=max_samples),
                "tokens_per_sec": deque(maxlen=max_samples),
                "input_size": deque(maxlen=max_samples),
                "output_size": deque(maxlen=max_samples),
                "calls": 0,
                "errors": 0,
            },
            "face_recognition": {
                "latency": deque(maxlen=max_samples),
                "fps": deque(maxlen=max_samples),
                "confidence": deque(maxlen=max_samples),
                "faces_detected": deque(maxlen=max_samples),
                "calls": 0,
                "errors": 0,
            },
            "system": {
                "cpu": deque(maxlen=max_samples),
                "memory": deque(maxlen=max_samples),
                "timestamps": deque(maxlen=max_samples)
            },
            "classification": {
                "true_labels": deque(maxlen=max_samples),
                "predicted_labels": deque(maxlen=max_samples),
                "confusion_matrix": {},
                "precision": {},
                "recall": {},
                "f1_score": {},
                "calls": 0
            }
        }
        
        # Create a lock for thread-safe updates
        self.lock = threading.Lock()
        
        # Start system metrics monitoring in separate thread
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_system)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
    
    def track_gemini_request(self, input_text, start_time=None):
        """Start tracking a Gemini AI request"""
        if start_time is None:
            start_time = time.time()
        return {
            "start_time": start_time,
            "input_size": len(input_text) if input_text else 0,
            "model": "gemini"
        }
    
    def track_gemini_response(self, tracking_info, response_text, error=None):
        """Complete tracking for a Gemini AI request"""
        end_time = time.time()
        latency = end_time - tracking_info["start_time"]
        
        with self.lock:
            if error:
                self.metrics["gemini"]["errors"] += 1
                return
            
            # Calculate metrics
            input_size = tracking_info["input_size"]
            output_size = len(response_text) if response_text else 0
            tokens_per_sec = output_size / latency if latency > 0 else 0
            
            # Store metrics
            self.metrics["gemini"]["latency"].append(latency)
            self.metrics["gemini"]["tokens_per_sec"].append(tokens_per_sec)
            self.metrics["gemini"]["input_size"].append(input_size)
            self.metrics["gemini"]["output_size"].append(output_size)
            self.metrics["gemini"]["calls"] += 1
            
            # Periodically save to file
            if self.metrics["gemini"]["calls"] % 5 == 0:
                self._save_metrics()
    
    def track_face_recognition_request(self, image=None, start_time=None):
        """Start tracking a face recognition request"""
        if start_time is None:
            start_time = time.time()
        return {
            "start_time": start_time,
            "model": "face_recognition"
        }
    
    def track_face_recognition_response(self, tracking_info, faces=None, confidence=None, error=None):
        """Complete tracking for a face recognition request"""
        end_time = time.time()
        latency = end_time - tracking_info["start_time"]
        
        with self.lock:
            if error:
                self.metrics["face_recognition"]["errors"] += 1
            
            # Calculate metrics
            fps = 1.0 / latency if latency > 0 else 0
            
            # Handle both direct faces parameter and faces_detected parameter
            if isinstance(faces, int) or faces is None:
                num_faces = faces if faces is not None else 0
            else:
                # Original behavior for list of faces
                num_faces = len(faces) if faces else 0
            
            # Handle confidence calculation if face objects have 'confidence' property
            avg_confidence = confidence if confidence is not None else 0
            if isinstance(faces, list) and faces and isinstance(faces[0], dict) and 'confidence' in faces[0]:
                # If we have a list of face objects with confidence values
                confidences = [face.get('confidence', 0) for face in faces]
                avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            
            # Store metrics
            self.metrics["face_recognition"]["latency"].append(latency)
            self.metrics["face_recognition"]["fps"].append(fps)
            self.metrics["face_recognition"]["confidence"].append(avg_confidence)
            self.metrics["face_recognition"]["faces_detected"].append(num_faces)
            self.metrics["face_recognition"]["calls"] += 1
            
            # Periodically save to file
            if self.metrics["face_recognition"]["calls"] % 5 == 0:
                self._save_metrics()
    
    def track_classification_result(self, true_label, predicted_label):
        """Track a classification result to build confusion matrix"""
        with self.lock:
            self.metrics["classification"]["true_labels"].append(true_label)
            self.metrics["classification"]["predicted_labels"].append(predicted_label)
            self.metrics["classification"]["calls"] += 1
            
            # Update confusion matrix
            if true_label not in self.metrics["classification"]["confusion_matrix"]:
                self.metrics["classification"]["confusion_matrix"][true_label] = {}
            
            if predicted_label not in self.metrics["classification"]["confusion_matrix"][true_label]:
                self.metrics["classification"]["confusion_matrix"][true_label][predicted_label] = 0
            
            self.metrics["classification"]["confusion_matrix"][true_label][predicted_label] += 1
            
            # Update precision, recall, and F1 score
            self._update_classification_metrics()
            
            # Periodically save to file
            if self.metrics["classification"]["calls"] % 5 == 0:
                self._save_metrics()

    def _update_classification_metrics(self):
        """Update precision, recall, and F1 score based on current confusion matrix"""
        cm = self.metrics["classification"]["confusion_matrix"]
        precision = {}
        recall = {}
        f1_score = {}
        
        # Calculate precision and recall for each class
        for true_label in cm:
            # True positives: cm[true_label][true_label]
            tp = cm[true_label].get(true_label, 0)
            
            # False positives: sum of all predictions for this label that are incorrect
            fp = 0
            for other_label in cm:
                if other_label != true_label:
                    fp += cm[other_label].get(true_label, 0)
            
            # False negatives: sum of all incorrect predictions for this true label
            fn = 0
            for pred_label in cm[true_label]:
                if pred_label != true_label:
                    fn += cm[true_label][pred_label]
            
            # Calculate metrics
            if tp + fp > 0:
                precision[true_label] = tp / (tp + fp)
            else:
                precision[true_label] = 0
                
            if tp + fn > 0:
                recall[true_label] = tp / (tp + fn)
            else:
                recall[true_label] = 0
                
            if precision[true_label] + recall[true_label] > 0:
                f1_score[true_label] = 2 * (precision[true_label] * recall[true_label]) / (precision[true_label] + recall[true_label])
            else:
                f1_score[true_label] = 0
        
        # Update metrics
        self.metrics["classification"]["precision"] = precision
        self.metrics["classification"]["recall"] = recall
        self.metrics["classification"]["f1_score"] = f1_score

    def get_classification_metrics(self):
        """Get current classification metrics"""
        with self.lock:
            # Get unique labels
            true_labels = set(self.metrics["classification"]["true_labels"])
            pred_labels = set(self.metrics["classification"]["predicted_labels"])
            all_labels = sorted(list(true_labels.union(pred_labels)))
            
            # Format confusion matrix for display
            cm = []
            for true_label in all_labels:
                row = []
                for pred_label in all_labels:
                    if true_label in self.metrics["classification"]["confusion_matrix"] and \
                       pred_label in self.metrics["classification"]["confusion_matrix"][true_label]:
                        row.append(self.metrics["classification"]["confusion_matrix"][true_label][pred_label])
                    else:
                        row.append(0)
                cm.append(row)
            
            # Calculate average metrics
            avg_precision = 0
            avg_recall = 0
            avg_f1 = 0
            count = 0
            
            for label in self.metrics["classification"]["precision"]:
                avg_precision += self.metrics["classification"]["precision"][label]
                avg_recall += self.metrics["classification"]["recall"][label]
                avg_f1 += self.metrics["classification"]["f1_score"][label]
                count += 1
            
            if count > 0:
                avg_precision /= count
                avg_recall /= count
                avg_f1 /= count
            
            return {
                "confusion_matrix": cm,
                "labels": all_labels,
                "precision": self.metrics["classification"]["precision"],
                "recall": self.metrics["classification"]["recall"],
                "f1_score": self.metrics["classification"]["f1_score"],
                "avg_precision": avg_precision,
                "avg_recall": avg_recall,
                "avg_f1": avg_f1,
                "total_samples": self.metrics["classification"]["calls"]
            }

    def get_current_metrics(self):
        """Get current metrics summary"""
        with self.lock:
            current = {}
            
            # Gemini metrics
            gemini = self.metrics["gemini"]
            current["gemini"] = {
                "avg_latency": float(np.mean(list(gemini["latency"]))) if gemini["latency"] else 0,
                "avg_tokens_per_sec": float(np.mean(list(gemini["tokens_per_sec"]))) if gemini["tokens_per_sec"] else 0,
                "total_calls": gemini["calls"],
                "error_rate": float(gemini["errors"] / gemini["calls"]) if gemini["calls"] > 0 else 0,
                # Add these for debugging
                "latency_count": len(gemini["latency"]),
                "samples": list(gemini["latency"])[:5] if gemini["latency"] else []
            }
            
            # Face recognition metrics
            face = self.metrics["face_recognition"]
            current["face_recognition"] = {
                "avg_latency": float(np.mean(list(face["latency"]))) if face["latency"] else 0,
                "avg_fps": float(np.mean(list(face["fps"]))) if face["fps"] else 0,
                "avg_confidence": float(np.mean(list(face["confidence"]))) if face["confidence"] else 0,
                "avg_faces": float(np.mean(list(face["faces_detected"]))) if face["faces_detected"] else 0,
                "total_calls": face["calls"],
                "error_rate": float(face["errors"] / face["calls"]) if face["calls"] > 0 else 0
            }
            
            # Classification metrics
            class_metrics = self.get_classification_metrics()
            current["classification"] = {
                "avg_precision": class_metrics["avg_precision"],
                "avg_recall": class_metrics["avg_recall"],
                "avg_f1": class_metrics["avg_f1"],
                "total_samples": class_metrics["total_samples"]
            }
            
            return current
    
    def _monitor_system(self):
        """Monitor system metrics in a background thread"""
        while self.running:
            try:
                with self.lock:
                    self.metrics["system"]["cpu"].append(psutil.cpu_percent())
                    self.metrics["system"]["memory"].append(psutil.virtual_memory().percent)
                    self.metrics["system"]["timestamps"].append(datetime.now().strftime("%H:%M:%S"))
                time.sleep(1)  # Update once per second
            except Exception as e:
                print(f"Error monitoring system: {str(e)}")
                time.sleep(5)  # Wait longer if there was an error
    
    def _save_metrics(self):
        """Save metrics to JSON file"""
        try:
            # Convert deques to lists for JSON serialization
            output = {}
            for category, metrics in self.metrics.items():
                output[category] = {}
                for key, value in metrics.items():
                    if isinstance(value, deque):
                        output[category][key] = list(value)
                    else:
                        output[category][key] = value
            
            # Add timestamp
            output["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Save to file
            with open(self.log_file, 'w') as f:
                json.dump(output, f, indent=2)
        except Exception as e:
            print(f"Error saving metrics: {str(e)}")
    
    def stop(self):
        """Stop the monitoring thread"""
        self.running = False
        self._save_metrics()

# Create global instance
performance_tracker = PerformanceTracker()

def track_ai_performance(func):
    """Decorator to track Gemini AI performance"""
    def wrapper(*args, **kwargs):
        # Determine appropriate input text based on method name and args
        input_text = ""
        
        # More robust method to extract input text
        try:
            if len(args) > 1:
                if 'drawing_context' in func.__code__.co_varnames:
                    # For get_detailed_art_suggestions
                    input_text = str(args[1])
                elif len(args) > 2:
                    # For get_art_suggestions
                    input_text = f"{args[1]}:{args[2]}"
        except Exception as e:
            print(f"Error extracting input text: {str(e)}")
        
        print(f"Tracking Gemini API call: {func.__name__} with input size {len(input_text)}")
        tracking_info = performance_tracker.track_gemini_request(input_text)
        
        try:
            result = func(*args, **kwargs)
            # Make sure we get the response text for metrics
            response_text = ""
            if isinstance(result, dict) and "suggestion" in result:
                response_text = result["suggestion"]
                print(f"Gemini API response size: {len(response_text)} chars")
            else:
                response_text = str(result)
                print(f"Gemini API response converted to string: {len(response_text)} chars")
            
            # Log the metrics we're about to store
            print(f"Storing Gemini metrics - latency: {time.time() - tracking_info['start_time']:.2f}s, " +
                 f"input size: {len(input_text)}, output size: {len(response_text)}")
                
            performance_tracker.track_gemini_response(tracking_info, response_text)
            
            # Print current metrics count
            with performance_tracker.lock:
                print(f"Current Gemini metrics count: calls={performance_tracker.metrics['gemini']['calls']}, " +
                     f"latency samples={len(performance_tracker.metrics['gemini']['latency'])}")
            
            return result
        except Exception as e:
            print(f"Error in Gemini API call: {str(e)}")
            performance_tracker.track_gemini_response(tracking_info, None, error=str(e))
            raise
    return wrapper

def track_face_recognition_performance(func):
    """Decorator to track face recognition performance"""
    def wrapper(*args, **kwargs):
        tracking_info = performance_tracker.track_face_recognition_request()
        try:
            result = func(*args, **kwargs)
            
            # Extract metrics from the result
            error = "error" in result
            
            # Use the 'faces' parameter instead of 'faces_detected'
            faces = result.get("faces", 0)
            confidence = result.get("confidence", 0)
            
            # Pass the metrics to the tracker
            performance_tracker.track_face_recognition_response(
                tracking_info, 
                faces=faces,  # This parameter name matches the method signature
                confidence=confidence, 
                error=error
            )
            return result
        except Exception as e:
            performance_tracker.track_face_recognition_response(tracking_info, None, None, error=str(e))
            raise
    return wrapper