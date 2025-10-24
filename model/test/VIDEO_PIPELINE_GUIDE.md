# Video-to-Gesture Classification Pipeline

## Overview
This pipeline converts video files to skeleton data and classifies gestures using your trained CLIP model.

## Files Created

### Core Files:
- `video_to_gesture.py` - Main pipeline for video processing and classification
- `test_video_pipeline.py` - Test script with sample video
- `requirements_video.txt` - Additional dependencies for video processing

## Installation

### 1. Install Dependencies:
```bash
pip install -r requirements_video.txt
```

### 2. Install MediaPipe:
```bash
pip install mediapipe
```

## Usage

### Basic Usage:
```bash
# Classify gesture from video
python video_to_gesture.py --video input_video.mp4 --model clip_gesture_model.pth --scaler clip_gesture_scaler.pkl --info clip_gesture_info.json

# Create output video with skeleton overlay
python video_to_gesture.py --video input_video.mp4 --model clip_gesture_model.pth --scaler clip_gesture_scaler.pkl --info clip_gesture_info.json --visualize --output result_video.mp4
```

### Test the Pipeline:
```bash
# Run test with sample video
python test_video_pipeline.py
```

## Pipeline Components

### 1. VideoToSkeleton Class
- **Purpose**: Extract skeleton keypoints from video frames
- **Technology**: MediaPipe Pose detection
- **Output**: 75-dimensional skeleton data (25 joints × 3 coordinates)

### 2. GestureClassifier Class
- **Purpose**: Classify gestures using trained CLIP model
- **Input**: Skeleton sequences
- **Output**: Gesture name, description, confidence score

### 3. Visualization
- **Purpose**: Create output video with skeleton overlay
- **Features**: Pose landmarks, gesture labels, confidence scores

## Key Features

### Skeleton Extraction:
- **25 keypoints** per frame
- **3D coordinates** (x, y, z)
- **Robust detection** with confidence thresholds
- **Missing data handling** (zero padding)

### Gesture Classification:
- **Sequence processing** (multiple frames)
- **Temporal modeling** (gesture dynamics)
- **Confidence scoring** (prediction reliability)
- **Multi-gesture support** (all trained gestures)

### Video Processing:
- **Real-time skeleton overlay**
- **Gesture label display**
- **Confidence visualization**
- **Frame-by-frame processing**

## Input Requirements

### Video Format:
- **Supported**: MP4, AVI, MOV
- **Resolution**: Any (recommended: 640x480 or higher)
- **Duration**: Any (recommended: 3-10 seconds)
- **FPS**: Any (recommended: 30 FPS)

### Content Requirements:
- **Person visible** in video
- **Clear pose** (not occluded)
- **Good lighting** for pose detection
- **Stable camera** (minimal shaking)

## Output Format

### Classification Results:
```python
{
    "gesture_name": "Elbow Flexion Left",
    "description": "Flexion and extension movement of the left elbow joint",
    "confidence": 0.85,
    "processing_time": 2.3
}
```

### Video Output:
- **Skeleton overlay** on original video
- **Gesture labels** with confidence
- **Frame-by-frame** pose tracking
- **Same format** as input video

## Troubleshooting

### Common Issues:

1. **MediaPipe not detecting pose**:
   - Check video quality and lighting
   - Ensure person is visible and not occluded
   - Try different video angles

2. **Low confidence scores**:
   - Video may not match training data
   - Try different video angles or lighting
   - Check if gesture is in training set

3. **Processing errors**:
   - Check video file format
   - Ensure all model files exist
   - Verify dependencies are installed

### Performance Tips:

1. **Faster processing**:
   - Use shorter videos (3-5 seconds)
   - Lower resolution videos
   - Reduce frame rate

2. **Better accuracy**:
   - Use higher resolution videos
   - Ensure good lighting
   - Stable camera position

## Example Workflow

### 1. Prepare Video:
```bash
# Record or obtain video of gesture
# Ensure person is visible and gesture is clear
```

### 2. Run Classification:
```bash
python video_to_gesture.py --video my_gesture.mp4 --model clip_gesture_model.pth --scaler clip_gesture_scaler.pkl --info clip_gesture_info.json
```

### 3. View Results:
```bash
# Check console output for classification results
# View output video with skeleton overlay
```

## Integration with Trained Model

### Model Requirements:
- **Trained CLIP model** (clip_gesture_model.pth)
- **Data scaler** (clip_gesture_scaler.pkl)
- **Model info** (clip_gesture_info.json)

### Model Loading:
```python
# Load trained model
model = torch.load('clip_gesture_model.pth')
scaler = joblib.load('clip_gesture_scaler.pkl')
with open('clip_gesture_info.json', 'r') as f:
    info = json.load(f)
```

## Future Enhancements

### Planned Features:
- **Real-time processing** (webcam input)
- **Batch processing** (multiple videos)
- **Gesture confidence** over time
- **Custom gesture training** from video

### Advanced Features:
- **Multi-person detection**
- **Gesture sequence** recognition
- **Temporal smoothing** of predictions
- **Custom visualization** options
