# Detecto - Real-Time Object Detection Model

Detecto is an AI-powered object detection system that identifies and locates multiple objects in real-time video streams. It uses state-of-the-art deep learning models to deliver fast, accurate, and reliable object detection while ignoring person detections for privacy purposes.

## Features

- 🚀 Real-time object detection using DETR (DEtection TRansformer) model
- 🔒 Privacy-focused: Automatically ignores person detections
- 🎥 Live camera feed processing with bounding boxes and labels
- ⚡ Optimized for performance with multi-threading
- 🖼️ Image and webcam snapshot testing capabilities
- 🧠 91 object classes detection (excluding persons)
- 🖥️ Cross-platform compatibility

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd Detecto-Object-detection-model
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

## Usage

### Real-time Camera Detection

Run the main application to start real-time object detection:
```bash
python main.py
```

Controls:
- Press 'q' to quit the application

### Testing Suite

The project includes a comprehensive testing suite:
```bash
python test.py
```

Available test modes:
1. Test on existing image files
2. Capture from webcam and test detection
3. Quick model verification

## Technical Details

### Model

The system uses the `facebook/detr-resnet-50` model from Hugging Face Transformers, which is based on the DETR (DEtection TRansformer) architecture. This model can detect 91 different object classes with high accuracy.

### Performance Optimizations

- **Multi-threading**: Separates camera capture, object detection, and display into different threads
- **Frame skipping**: Processes every Nth frame to maintain real-time performance
- **GPU acceleration**: Automatically utilizes CUDA if available
- **Confidence thresholding**: Filters low-confidence detections

### Privacy Features

- **Person detection filtering**: All person detections are automatically filtered out
- **No data storage**: No images or videos are saved by default
- **Local processing**: All processing happens on your device

## Requirements

See [requirements.txt](requirements.txt) for detailed dependencies.

## Project Structure

```
.
├── main.py              # Real-time camera object detection
├── test.py              # Image and webcam testing suite
├── requirements.txt     # Python dependencies
├── README.md            # This file
└── .gitignore           # Git ignore rules
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- DETR model by Facebook AI Research
- Hugging Face Transformers library
- OpenCV for computer vision operations