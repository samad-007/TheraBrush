"""
Quick Draw CNN Model Training
Based on the exact methodology from: https://dev.to/larswaechter/recognizing-hand-drawn-doodles-using-deep-learning-ki0

This script creates and trains the CNN model using the Quick Draw dataset.
Architecture matches the article exactly:
- Input: 28x28x1 grayscale images
- 345 output classes (all Quick Draw categories)
- Sequential model with Conv2D, MaxPooling, Dense layers
"""

import os
import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Rescaling, BatchNormalization, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.callbacks import TensorBoard
import matplotlib.pyplot as plt

# Dataset parameters (matching the article)
DATASET_DIR = Path("dataset")
IMAGE_SIZE = (28, 28)
BATCH_SIZE = 32
N_CLASSES = 345  # Total classes in Quick Draw dataset
INPUT_SHAPE = (28, 28, 1)  # Grayscale images

# Training parameters (matching the article)
EPOCHS = 14
VALIDATION_SPLIT = 0.2

def load_dataset():
    """
    Load the Quick Draw dataset using Keras image_dataset_from_directory.
    Splits into 80/20 train/validation as per the article.
    
    Returns:
        train_ds: Training dataset
        val_ds: Validation dataset
    """
    print("=" * 80)
    print("LOADING QUICK DRAW DATASET")
    print("=" * 80)
    print(f"Dataset directory: {DATASET_DIR.absolute()}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Validation split: {VALIDATION_SPLIT * 100}%")
    print("=" * 80 + "\n")
    
    # Load training set (80%)
    train_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123,
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Load validation set (20%)
    val_ds = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        color_mode="grayscale",
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Calculate dataset statistics
    num_training_batches = tf.data.experimental.cardinality(train_ds).numpy()
    num_validation_batches = tf.data.experimental.cardinality(val_ds).numpy()
    num_classes = len(train_ds.class_names)
    
    print(f"✅ Dataset loaded successfully!")
    print(f"   - Classes: {num_classes}")
    print(f"   - Training images: ~{num_training_batches * BATCH_SIZE}")
    print(f"   - Validation images: ~{num_validation_batches * BATCH_SIZE}")
    print(f"   - Class names: {train_ds.class_names[:5]}... (showing first 5)")
    print()
    
    return train_ds, val_ds, train_ds.class_names

def visualize_samples(train_ds, num_samples=9):
    """
    Visualize random samples from the training dataset.
    
    Args:
        train_ds: Training dataset
        num_samples: Number of samples to display (default: 9 for 3x3 grid)
    """
    print("Generating sample visualization...")
    
    plt.figure(figsize=(8, 8))
    for images, labels in train_ds.take(1):
        for i in range(min(num_samples, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            data = images[i].numpy().astype("uint8")
            plt.imshow(data, cmap='gray', vmin=0, vmax=255)
            plt.title(train_ds.class_names[labels[i]])
            plt.axis("off")
    
    # Save visualization
    viz_path = "dataset_samples.png"
    plt.savefig(viz_path)
    print(f"✅ Visualization saved to {viz_path}")
    plt.close()

def build_model(n_classes=N_CLASSES, input_shape=INPUT_SHAPE):
    """
    Build the CNN model with the exact architecture from the article.
    
    Architecture:
    - Rescaling layer (1./255)
    - BatchNormalization
    - 3x Conv2D layers (6, 8, 10 filters) with 3x3 kernels, same padding, ReLU
    - BatchNormalization
    - MaxPooling2D (2x2)
    - Flatten
    - Dense 700 units, ReLU
    - BatchNormalization
    - Dropout 0.2
    - Dense 500 units, ReLU
    - BatchNormalization
    - Dropout 0.2
    - Dense 400 units, ReLU
    - Dropout 0.2
    - Dense n_classes units, softmax
    
    Args:
        n_classes: Number of output classes
        input_shape: Input shape (28, 28, 1)
    
    Returns:
        model: Compiled Keras Sequential model
    """
    print("=" * 80)
    print("BUILDING CNN MODEL")
    print("=" * 80)
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {n_classes}")
    print("Architecture: Matching dev.to article exactly")
    print("=" * 80 + "\n")
    
    model = Sequential([
        # Input and normalization
        Rescaling(1. / 255, input_shape=input_shape),
        BatchNormalization(),
        
        # Convolutional layers
        Conv2D(6, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(8, kernel_size=(3, 3), padding="same", activation="relu"),
        Conv2D(10, kernel_size=(3, 3), padding="same", activation="relu"),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten
        Flatten(),
        
        # Dense layers with dropout
        Dense(700, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(500, activation="relu"),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(400, activation="relu"),
        Dropout(0.2),
        
        # Output layer
        Dense(n_classes, activation="softmax")
    ])
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Print model summary
    print("\nModel Summary:")
    print("-" * 80)
    model.summary()
    print("-" * 80)
    
    # Count parameters
    total_params = model.count_params()
    print(f"\n✅ Model built successfully!")
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Expected: ~2,068,019 (as per article)")
    print()
    
    return model

def train_model(model, train_ds, val_ds, epochs=EPOCHS):
    """
    Train the model for the specified number of epochs.
    Uses TensorBoard for visualization.
    
    Args:
        model: Compiled Keras model
        train_ds: Training dataset
        val_ds: Validation dataset
        epochs: Number of training epochs
    
    Returns:
        history: Training history object
    """
    print("=" * 80)
    print("STARTING MODEL TRAINING")
    print("=" * 80)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {BATCH_SIZE}")
    print("Optimizer: Adam")
    print("Loss: Sparse Categorical Crossentropy")
    print("=" * 80 + "\n")
    
    # Setup TensorBoard callback
    log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(log_dir, histogram_freq=1)
    
    print(f"TensorBoard logs: {log_dir}")
    print("To view training progress, run: tensorboard --logdir=logs")
    print()
    
    # Train the model
    print("Training started...")
    print("-" * 80)
    
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        verbose=1,
        callbacks=[tensorboard_callback]
    )
    
    print("-" * 80)
    print("✅ Training complete!")
    print()
    
    # Display final metrics
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    
    print("Final Metrics:")
    print(f"   - Training accuracy: {final_train_acc * 100:.2f}%")
    print(f"   - Validation accuracy: {final_val_acc * 100:.2f}%")
    print(f"   - Training loss: {final_train_loss:.4f}")
    print(f"   - Validation loss: {final_val_loss:.4f}")
    print()
    
    return history

def save_model(model, class_names):
    """
    Save the trained model in both Keras and SavedModel formats.
    
    Args:
        model: Trained Keras model
        class_names: List of class names
    """
    print("=" * 80)
    print("SAVING MODEL")
    print("=" * 80)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save in Keras format (.keras)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    keras_path = models_dir / "drawing_model.keras"
    model.save(keras_path)
    print(f"✅ Saved Keras model: {keras_path}")
    
    # Save in SavedModel format (for TensorFlow Serving)
    try:
        savedmodel_path = models_dir / "drawing_model"
        model.export(str(savedmodel_path))
        print(f"✅ Saved SavedModel format: {savedmodel_path}")
    except Exception as e:
        print(f"⚠️  Could not save SavedModel format: {e}")
    
    # Save timestamped backup in Keras format
    try:
        backup_path = models_dir / f"drawing_model_{timestamp}.keras"
        model.save(backup_path)
        print(f"✅ Saved backup: {backup_path}")
    except Exception as e:
        print(f"⚠️  Could not save backup: {e}")
    
    # Save class names
    class_names_path = models_dir / "class_names.txt"
    with open(class_names_path, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    print(f"✅ Saved class names: {class_names_path}")
    
    print()
    print("=" * 80)
    print("MODEL SAVED SUCCESSFULLY!")
    print("=" * 80)
    print(f"Main model: {keras_path}")
    print(f"Class names: {class_names_path}")
    print(f"Total files in models directory: {len(list(models_dir.iterdir()))}")
    print("=" * 80)

def plot_training_history(history):
    """
    Plot training history (accuracy and loss).
    
    Args:
        history: Training history object
    """
    print("Generating training history plots...")
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = "training_history.png"
    plt.savefig(plot_path)
    print(f"✅ Training history plot saved to {plot_path}")
    plt.close()

def main():
    """
    Main training pipeline.
    """
    print("\n" + "=" * 80)
    print("QUICK DRAW CNN TRAINING PIPELINE")
    print("Based on: https://dev.to/larswaechter/recognizing-hand-drawn-doodles-using-deep-learning-ki0")
    print("=" * 80 + "\n")
    
    # Check if dataset exists
    if not DATASET_DIR.exists():
        print(f"❌ Error: Dataset directory not found: {DATASET_DIR}")
        print("\n💡 Please run 'python quickdraw_dataset.py' first to download the dataset")
        return
    
    # Load dataset
    train_ds, val_ds, class_names = load_dataset()
    
    # Visualize samples
    visualize_samples(train_ds)
    
    # Build model
    model = build_model(n_classes=len(class_names))
    
    # Train model
    history = train_model(model, train_ds, val_ds, epochs=EPOCHS)
    
    # Plot training history
    plot_training_history(history)
    
    # Save model
    save_model(model, class_names)
    
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE!")
    print("=" * 80)
    print("\n📊 Next steps:")
    print("   1. View training logs: tensorboard --logdir=logs")
    print("   2. Check training history: training_history.png")
    print("   3. Test the model in your application")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
