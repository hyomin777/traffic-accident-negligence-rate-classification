# Traffic Accident Negligence Rate Classification

This project implements an AI model that automatically classifies negligence rates in traffic accidents by analyzing video footage. It integrates video frames, object detection results, and metadata to predict negligence rates.

## Key Features

- Traffic accident video-based negligence rate classification (0:100 to 100:0, 11 classes)
- Video feature extraction using Swin Transformer 3D
- YOLO-based object detection integration
- Metadata analysis (accident type, location, progress information)

## Model Architecture

The model consists of the following main components:

1. **Video Backbone (Swin Transformer 3D)**
   - Spatiotemporal feature extraction from video frames
   - Pre-trained weights utilization

2. **Object Detection Encoder**
   - Processing of YOLO-detected object information
   - Transformer-based temporal processing

3. **Metadata Predictor**
   - Accident type prediction
   - Accident location prediction
   - Location feature prediction
   - Vehicle progress information prediction

4. **Multimodal Fusion Network**
   - Integration of video, object, and metadata features
   - Final negligence rate prediction

## Installation

1. Clone the repository:
```bash
git clone https://github.com/hyomin777/traffic-accident-negligence-rate-classification.git
cd traffic-accident-negligence-rate-classification
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Model Training

```bash
python main.py \
    --train_dir [train_data_directory] \
    --val_dir [validation_data_directory] \
    --yolo_weights [yolo_weights_path] \
    [--model_weights [pretrained_model_weights]] \
    [--batch_size BATCH_SIZE] \
    [--epochs EPOCHS] \
    [--lr LEARNING_RATE] \
    [--aux_lambda AUX_LAMBDA]
```

## Negligence Rate Classes

| Class | Negligence Rate |
|-------|-----------------|
| 0     | 0:100           |
| 1     | 10:90           |
| 2     | 20:80           |
| 3     | 30:70           |
| 4     | 40:60           |
| 5     | 50:50           |
| 6     | 60:40           |
| 7     | 70:30           |
| 8     | 80:20           |
| 9     | 90:10           |
| 10    | 100:0           |

## System Requirements

- Python 3.8 or higher
- CUDA-capable GPU (recommended)
- Minimum 16GB RAM
- Minimum 20GB disk space