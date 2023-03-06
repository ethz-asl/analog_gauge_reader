from ultralytics import YOLO


def train_yolo_model(task, data_file, model_name):

    training_epochs = 5
    plots = True
    conf = 0.5  # confidence threshold yolo
    export_format = "onnx"

    model = YOLO(model_name)
    model.train(task=task,
                data=data_file,
                plots=plots,
                epochs=training_epochs,
                conf=conf)
    model.val()
    model.export(format=export_format)


if __name__ == "__main__":

    detection_data_file = "data\\detection\\data.yaml"
    detection_model_name = "yolov8n.pt"

    segmentation_data_file = "data\\segmentation\\data.yaml"
    segmentation_model_name = "yolov8n-seg.pt"

    train_yolo_model('detect', segmentation_data_file, segmentation_model_name)
    train_yolo_model('segment', segmentation_data_file,
                     segmentation_model_name)
