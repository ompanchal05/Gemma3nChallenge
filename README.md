# 🌿 Gemma3n Leaf Detection Application

An AI-powered leaf detection system that uses computer vision and machine learning to identify different types of leaves and provide information about them using the Gemma3n API.

## Features

- **Real-time leaf detection** using computer vision
- **Machine learning classification** with KNN algorithm
- **Text-to-speech feedback** for detected leaves
- **AI-powered information** about leaf properties using Gemma3n API
- **AI Assistant** with comprehensive voice guidance about leaves
- **Web interface** for easy interaction
- **Camera integration** for live video feed
- **Database logging** of detection history

## Prerequisites

- Python 3.7 or higher
- Webcam or camera device
- Gemma3n API key (optional, for real AI responses)

## Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd Gemma3n
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up your API key (optional)**
   - Get your Gemma3n API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set the environment variable:
     - **Windows**: `set GEMMA3N_API_KEY=your_api_key_here`
     - **Linux/Mac**: `export GEMMA3N_API_KEY=your_api_key_here`

## Usage

### Quick Start
Run the helper script for automatic setup:
```bash
python run_app.py
```

### Manual Start
```bash
python app.py
```

### AI Assistant
```bash
# Run the interactive AI assistant
python assistant.py

# Run a demo of the assistant
python demo_assistant.py
```

### Access the Application
1. Open your web browser
2. Navigate to `http://localhost:5000`
3. Allow camera access when prompted
4. Point your camera at leaves to detect them

## How It Works

1. **Image Processing**: The application captures video frames and processes them to detect green regions (leaves)
2. **Feature Extraction**: For each detected leaf, it extracts color histogram and contour features
3. **Machine Learning**: Uses a KNN classifier trained on sample leaf images to identify the leaf type
4. **AI Integration**: Queries the Gemma3n API for detailed information about detected leaves
5. **Audio Feedback**: Provides text-to-speech information about the detected leaves

## File Structure

```
Gemma3n/
├── app.py                 # Main Flask application
├── assistant.py           # AI Assistant with speech functionality
├── demo_assistant.py      # Demo script for assistant
├── run_app.py            # Helper script for easy startup
├── test_app.py           # Testing script
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── static/
│   ├── images/          # Training images for ML
│   ├── style.css        # Web interface styling
│   └── app.js           # Frontend JavaScript
└── templates/
    ├── index.html       # Main web interface
    ├── assistant.html   # AI Assistant web interface
    └── *.html           # Other leaf information pages
```

## Troubleshooting

### Common Issues

1. **Camera not working**
   - Ensure your camera is connected and not being used by another application
   - Try running the application with administrator privileges
   - Check if your browser allows camera access

2. **API key errors**
   - Verify your Gemma3n API key is correct
   - Ensure the environment variable is set properly
   - The application will work with placeholder responses if no API key is provided

3. **Missing dependencies**
   - Run `pip install -r requirements.txt`
   - If you encounter issues, try installing packages individually

4. **Image loading errors**
   - Ensure all training images are present in `static/images/`
   - The application will create dummy data if images are missing

5. **Text-to-speech not working**
   - Install system text-to-speech drivers
   - On Windows, ensure Windows Speech Recognition is enabled
   - The application will continue to work without audio feedback

### Error Messages

- **"Camera not available"**: Camera device cannot be accessed
- **"No valid training images found"**: Training images are missing or corrupted
- **"API request failed"**: Network or API key issues
- **"Text-to-speech error"**: Audio system issues

## Configuration

### Environment Variables
- `GEMMA3N_API_KEY`: Your Gemma3n API key for real AI responses

### Customization
- Add your own leaf images to `static/images/` for training
- Modify the color detection thresholds in `detect_leaf()` function
- Adjust the KNN parameters for better classification

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## License

This project is open source and available under the MIT License.

## Support

If you encounter any issues or have questions, please:
1. Check the troubleshooting section above
2. Review the error logs in the console
3. Create an issue in the repository

---

**Note**: This application requires a camera for full functionality. The AI features work best with a valid Gemma3n API key, but the application will function with placeholder responses if no key is provided.
