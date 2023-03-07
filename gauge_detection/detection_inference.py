from ultralytics import YOLO


def detection_gauge_face(img, model_path='best.pt'):
    model = YOLO(model_path)  # load model

    results = model(img)  # run inference

    boxes = results[0].boxes
    box = boxes[0]  # returns one box
    return box.xyxy[0].int()
