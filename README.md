# TheraBrush

TheraBrush is an AI-powered therapeutic art analysis tool that combines drawing recognition, emotion detection, and ChatGPT-based therapeutic guidance to provide insights into users' emotional well-being through their artwork.

## Features

- **Drawing Recognition**: Uses TensorFlow/Keras models to recognize drawings and analyze artistic patterns
- **Emotion Detection**: Integrates with Face++ API to detect emotions from facial expressions
- **AI Therapeutic Advisor**: Leverages ChatGPT to provide therapeutic insights based on drawings and detected emotions
- **Performance Tracking**: Built-in performance metrics tracking for API calls and model inference
- **Real-time Analysis**: Web-based interface for instant feedback and analysis

## Project Structure

```
therabrush/
├── pokemon/                    # Main application directory
│   ├── app.py                 # Flask web application
│   ├── main.py                # Core emotion detection logic
│   ├── chatgpt_advisor.py     # ChatGPT integration for therapeutic advice
│   ├── tf_drawing_recognition.py  # TensorFlow drawing recognition
│   ├── performance_metrics.py # Performance tracking utilities
│   ├── models/                # Machine learning models
│   ├── static/                # CSS and JavaScript files
│   ├── templates/             # HTML templates
│   └── uploads/               # User-uploaded images
├── requirements.txt           # Python dependencies
└── requirements-minimal.txt   # Minimal dependencies
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
OPENAI_API_KEY=your_openai_api_key
```

### Running the Application

1. Navigate to the pokemon directory:
```bash
cd pokemon
```

2. Run the Flask application:
```bash
python app.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Usage

1. **Upload or Draw**: Users can upload an image or use the drawing canvas
2. **Automatic Analysis**: The system analyzes the drawing using TensorFlow models
3. **Emotion Detection**: If a face is present, Face++ API detects emotions
4. **Therapeutic Insights**: ChatGPT provides personalized therapeutic feedback
5. **Performance Metrics**: View detailed performance metrics in the dashboard

## API Integrations

- **Face++**: For facial emotion recognition
- **OpenAI ChatGPT**: For therapeutic advice generation
- **TensorFlow/Keras**: For drawing classification and pattern recognition

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

## Acknowledgments

- Face++ API for emotion detection
- OpenAI for ChatGPT integration
- TensorFlow team for the machine learning framework
- Flask community for the web framework

## Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This project is intended for research and educational purposes. It should not replace professional mental health services.
