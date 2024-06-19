# YOLOv8 Custom Object Detection

This repository is a customized version of the Ultralytics YOLOv8 code, tailored specifically for object detection. The original YOLOv8 implementation by Ultralytics supports multiple image tasks such as segmentation, pose estimation, and oriented bounding boxes. However, this repository focuses solely on object detection, simplifying the code and making it easier to read and modify.

## Features

- **Simplified Code**: The codebase has been streamlined for easier readability and modification.
- **Focused Functionality**: Removed support for segmentation, pose estimation, and oriented bounding boxes to concentrate solely on object detection.

## Instructions to Use

1. **Clone the Repository**
    ```bash
    git clone https://github.com/RUPESH-KUMAR01/yolo_v8.git
    ```

2. **Configure Paths**
   - Update the `DATASETS_DIR` and `ASSETS` variables in the `cfg/__init__.py` file according to your usage.

3. **Update Input Settings**
   - The `main.py` script uses `cfg/default.yaml` for inputs. Modify the values in the `default.yaml` file before calling functions in the `main.py`.

## Future Developments

- **Argument Parser**: Modify the `main.py` file to include an `argparser` so that we don't have to change the `default.yaml` file every time.

## Additional Information
- This repository contains weights that are trained on yolov8n.yaml configuration model in two different datasets.
    1. Pascal VOC dataset trained for 10 epochs
    2. coco dataset trained for 2 epochs
- In the default the weights from the coco(trained for 2 epochs) is being used.
