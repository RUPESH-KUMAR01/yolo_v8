import cv2
from model.predict import DetectionPredictor
from model.train import DetectionTrainer
from model.validator import DetectionValidator


def train():
    trainer=DetectionTrainer()
    trainer.train()
def val():
    validator=DetectionValidator()
    validator()
def predict_source(model=r"runs\coco\weights\best.pt",source=r"classroom.mp4"):
    predictor=DetectionPredictor()
    predictor.setup_model(model=model, verbose=False)
    results=predictor(source=source,stream=True)
    for i in results:
        pass
# predict_source()
# train()
val()

# # Open the video file
# video_path = r"C:\Users\thata\anaconda3\Lib\site-packages\ultralytics\classroom.mp4"
# cap = cv2.VideoCapture("http://192.168.4.129:8080/video")
# predictor=DetectionPredictor()
# predictor.setup_model(model=r"C:\Users\thata\intern\code\pre-built-models\modified\runs\coco\weights\best.pt", verbose=True)
# # Loop through the video frames
# while cap.isOpened():
#     # Read a frame from the video
#     success, frame = cap.read()

#     if success:
#         # Run YOLOv8 inference on the frame
#         results = predictor(frame)
#         # Visualize the results on the frame
#         annotated_frame = results[0].plot()

#         # Display the annotated frame
#         cv2.imshow("YOLOv8 Inference", annotated_frame)

#         # Break the loop if 'q' is pressed or window is closed
#         if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty('YOLOv8 Inference', cv2.WND_PROP_VISIBLE) < 1:
#             break
#     else:
#         # Break the loop if the end of the video is reached
#         break

# # Release the video capture object and close all OpenCV windows
# cap.release()
# cv2.destroyAllWindows()
