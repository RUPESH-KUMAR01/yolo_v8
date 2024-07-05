markdown
Copy code
# YOLOv8 Custom Object Detection

This repository is a customized version of the Ultralytics YOLOv8 code, tailored specifically for object detection. The original YOLOv8 implementation by Ultralytics supports multiple image tasks such as segmentation, pose estimation, and oriented bounding boxes. However, this repository focuses solely on object detection, simplifying the code and making it easier to read and modify.

## Features
- **Simplified Code**: The codebase has been streamlined for easier readability and modification.
- **Focused Functionality**: Removed support for segmentation, pose estimation, and oriented bounding boxes to concentrate solely on object detection.


### Training

To train the model, use the following command:

```bash
python main.py --mode train --model yolov8.yaml --data VOC.yaml --epochs 10
```

#### Training Options

- `--model`: Path to the model configuration file (default: `yolov8.yaml`).
- `--data`: Path to the data configuration file (default: `VOC.yaml`).
- `--epochs`: Number of epochs to train for (default: 1).

Additional options can be found in `main.py`.

### Validation

To validate the model, use the following command:

```bash
python main.py --mode val --model runs\detect\train\weights\best.pt --data VOC.yaml
```

#### Validation Options

- `--model`: Path to the model configuration file (default: `yolov8.yaml`).
- `--data`: Path to the data configuration file (default: `VOC.yaml`).

Additional options can be found in `main.py`.

### Prediction

To run predictions on images, use the following command:

```bash
python main.py --mode predict --model runs/detect/train/weights/best.pt --source assets/bus.jpg
```

To use a webcam for predictions, use the following command:

```bash
python main.py --mode predict --model runs/detect/train/weights/best.pt --webcam True
```

#### Prediction Options

- `--model`: Path to the model weights file (default: `runs/detect/train/weights/best.pt`).
- `--source`: Source directory for images or videos (default: `assets`).
- `--webcam`: Use the system webcam for live predictions (default: `False`).

Additional options can be found in `main.py`.

## Available Weights

This repository contains weights trained on the following datasets:

- Pascal VOC dataset: Trained for 10 epochs

Those weights are being used as default for val and predict 

## Additional Information

For a comprehensive list of all configurable options, refer to the argument parser in `main.py`.