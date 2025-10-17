# TheraBrush

TheraBrush is an AI-powered therapeutic art analysis tool that combines drawing recognition, emotion detection, and AI-based therapeutic guidance to provide insights into users' emotional well-being through their artwork.

## Features

- **Drawing Recognition**: Uses Quick Draw CNN model trained on 345+ categories from Google's Quick Draw dataset
- **Emotion Detection**: Integrates with Face++ API to detect emotions from facial expressions
- **AI Therapeutic Advisor**: Leverages Gemini AI to provide therapeutic insights based on drawings and detected emotions
- **Performance Tracking**: Built-in performance metrics tracking for API calls and model inference
- **Real-time Analysis**: Web-based interface for instant feedback and analysis

## Project Structure

```
therabrush/
├── pokemon/                    # Main application directory
│   ├── app.py                 # Flask web application (legacy)
│   ├── app_quickdraw.py       # Flask app with Quick Draw CNN
│   ├── main.py                # Core emotion detection logic
│   ├── chatgpt_advisor.py     # Gemini AI integration for therapeutic advice
│   ├── quickdraw_dataset.py   # Quick Draw dataset downloader
│   ├── train_quickdraw_model.py  # CNN model training script
│   ├── quickdraw_recognizer.py   # Quick Draw CNN recognizer
│   ├── tf_drawing_recognition.py # TensorFlow drawing recognition (legacy)
│   ├── performance_metrics.py # Performance tracking utilities
│   ├── models/                # Machine learning models
│   │   ├── drawing_model.keras    # Trained Quick Draw CNN model
│   │   └── class_names.txt        # 345 Quick Draw categories
│   ├── dataset/               # Quick Draw training data (gitignored)
│   ├── logs/                  # TensorBoard training logs (gitignored)
│   ├── static/                # CSS and JavaScript files
│   ├── templates/             # HTML templates
│   └── uploads/               # User-uploaded images
├── requirements.txt           # Python dependencies
├── requirements-minimal.txt   # Minimal dependencies
└── test_quickdraw_setup.py    # Automated validation script
```

## Setup

### Prerequisites

- Python 3.8+
- Node.js (for face-api.js dependencies)
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TheraBrush.git
cd TheraBrush
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Install Node.js dependencies (for face-api.js):
```bash
cd pokemon
npm install
```

5. Set up environment variables:
Create a `.env` file in the root directory with your API keys:
```
FACEPP_API_KEY=your_facepp_api_key
FACEPP_API_SECRET=your_facepp_api_secret
GEMINI_API_KEY=your_gemini_api_key
```

### Quick Draw CNN Setup

The drawing recognition system uses a CNN trained on Google's Quick Draw dataset.

#### Option 1: Quick Test (Sample Dataset)
```bash
cd pokemon

# Download sample dataset (10 classes, ~10 minutes)
python quickdraw_dataset.py sample

# Train the model (~20 minutes)
python train_quickdraw_model.py
```

#### Option 2: Full Production (Complete Dataset)
```bash
cd pokemon

# Download full dataset (345 classes, 2-6 hours, ~2-3 GB)
python quickdraw_dataset.py

# Train the model (2-4 hours)
python train_quickdraw_model.py
```

#### Validate Setup
```bash
# Run automated validation
python test_quickdraw_setup.py
```

### Running the Application

1. Navigate to the pokemon directory:
```bash
cd pokemon
```

2. Run the Flask application:
```bash
# Using Quick Draw CNN (recommended)
python app_quickdraw.py

# Or using legacy app
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5002
```

## Usage

1. **Upload or Draw**: Users can upload an image or use the drawing canvas
2. **Automatic Analysis**: The system analyzes the drawing using TensorFlow models
3. **Emotion Detection**: If a face is present, Face++ API detects emotions
4. **Therapeutic Insights**: ChatGPT provides personalized therapeutic feedback
5. **Performance Metrics**: View detailed performance metrics in the dashboard

## Quick Draw CNN Architecture

The drawing recognition system uses a Convolutional Neural Network trained on Google's Quick Draw dataset:

- **Dataset**: 345 categories from Google Cloud Platform
- **Training Images**: 414,000 images (1,200 per category)
- **Model Architecture**: 
  - Input: 28×28 grayscale images
  - 3 Conv2D layers (6, 8, 10 filters)
  - MaxPooling and BatchNormalization
  - 3 Dense layers (700, 500, 400 units)
  - Output: 345 classes with softmax
  - Total Parameters: ~2,068,019
- **Expected Accuracy**: ~61% validation accuracy
- **Inference Time**: 15-30ms per drawing

### API Endpoints

#### POST `/analyze_drawing`
Analyzes canvas drawing and returns recognized shape with confidence.

**Request:**
```json
{
  "strokes": [[[x0, x1, ...], [y0, y1, ...]], ...],
  "box": [min_x, min_y, max_x, max_y]
}
```

**Response:**
```json
{
  "recognized_shape": "cat",
  "confidence": 85.3,
  "top_predictions": [
    {"class_name": "cat", "probability": 0.853, "confidence": 85.3},
    {"class_name": "dog", "probability": 0.092, "confidence": 9.2},
    {"class_name": "face", "probability": 0.041, "confidence": 4.1}
  ]
}
```

#### GET `/get_art_suggestion`
Returns therapeutic suggestions based on emotion and recognized drawing.

#### GET `/metrics`
Returns performance metrics for API calls and model inference.

## API Integrations

- **Face++**: For facial emotion recognition
- **Google Gemini AI**: For therapeutic advice generation
- **TensorFlow/Keras**: For CNN training and inference
- **Quick Draw Dataset**: Google Cloud Platform dataset for training

## Development

### Running Tests

```bash
python -m pytest
```

### Performance Monitoring

Access the metrics dashboard at:
```
http://localhost:5000/metrics
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Implementation Reference

The Quick Draw CNN implementation is based on the methodology described in:
- **Article**: [Recognizing hand drawn Doodles using Deep Learning](https://dev.to/larswaechter/recognizing-hand-drawn-doodles-using-deep-learning-ki0) by Lars Wächter
- **Dataset**: [Quick Draw Dataset](https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/simplified) on Google Cloud Platform

## Acknowledgments

- Face++ API for emotion detection
- Google Gemini AI for therapeutic advice generation
- Google Quick Draw team for the dataset
- TensorFlow team for the machine learning framework
- Flask community for the web framework
- Lars Wächter for the CNN architecture methodology

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This project is intended for research and educational purposes. It should not replace professional mental health services.
