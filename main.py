from model.predict import DetectionPredictor
from model.train import DetectionTrainer
from model.validator import DetectionValidator


def train():
    trainer=DetectionTrainer()
    trainer.train()
def val():
    validator=DetectionValidator()
    validator()
def predict_source():
    predictor=DetectionPredictor()
    predictor.setup_model(model=r"C:\Users\thata\intern\code\pre-built-models\modified\runs\coco\weights\best.pt", verbose=False)
    predictor.stream_inference(source=r"C:\Users\thata\intern\code\pre-built-models\modified\classroom.mp4",stream=True)
predict_source()