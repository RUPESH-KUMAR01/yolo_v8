import argparse
import cv2
from cfg import DEFAULT_CFG_DICT, ROOT
from main_parser import parse_args
from model.predict import DetectionPredictor
from model.train import DetectionTrainer
from model.validator import DetectionValidator
from utils import LOGGER


def train(overrides):
    trainer=DetectionTrainer(overrides=overrides)
    trainer.train()
def val(overrides):
    validator=DetectionValidator(args=overrides)
    validator()
def predict_source(overrides,model=r"runs\coco\weights\best.pt",source=r"assets\bus.jpg"):
    predictor=DetectionPredictor(overrides=overrides)
    predictor.setup_model(model=model, verbose=False)
    results=predictor(source=source,stream=True)
    for i in results:
        pass

def webcam(model=r'runs\coco\weights\best.pt',source=0):
    # Open the video file
    cap = cv2.VideoCapture(source)
    predictor=DetectionPredictor(overrides={"mode":"predict","save":False})
    predictor.setup_model(model=model, verbose=True)
    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = predictor(frame)
            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed or window is closed
            if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty('YOLOv8 Inference', cv2.WND_PROP_VISIBLE) < 1:
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
def main(overrides):
    if overrides["mode"]=="val":
        overrides.pop("webcam")
        if ("model" in overrides )& (overrides["model"].endswith(".pt")):
            LOGGER.info(f"Using default dataset for validation {DEFAULT_CFG_DICT['data']}")
            val(overrides)
    elif overrides["mode"]=="predict":
        if overrides["webcam"]:
            webcam(overrides["model"] if "model" in overrides else None ,overrides["source"] if "source" in overrides else 0)
        else:
            overrides.pop("webcam")
            predict_source(overrides=overrides,model=overrides["model"] if "model" in overrides else None,source=overrides["source"] if "source" in overrides else None)
    else:
        overrides.pop("webcam")
        train(overrides)

if __name__ == "__main__":
    overrides = parse_args()
    if "model" in overrides:
        if overrides["model"].endswith(".yaml"):
            overrides["mode"]="train"
    if "source" in overrides:
        overrides["mode"]="predict"
    main(overrides)