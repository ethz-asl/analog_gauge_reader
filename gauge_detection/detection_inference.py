from ultralytics import YOLO


def detection_gauge_face(img, model_path='best.pt'):
    '''
    uses yolo v8 to get bounding box of gauge face
    :param img: numpy image
    :param model_path: path to yolov8 detection model
    :return: highest confidence box for further processing and list of all boxes for visualization
    '''
    model = YOLO(model_path)  # load model

    results = model(img)  # run inference, detects gauge face and needle

    # get list of detected boxes, already sorted by confidence
    boxes = results[0].boxes

    gauge_face_box = None

    # get first box which is of a gauge face
    for box in boxes:
        if box.cls == 0:
            gauge_face_box = box
            break

    if gauge_face_box is None:
        raise Exception('No gauge face detected in image.')

    return gauge_face_box.xyxy[0].int(), boxes
