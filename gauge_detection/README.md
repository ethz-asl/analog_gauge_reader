# Gauge Detection

We use yolov8 from ultralytics <https://github.com/ultralytics/ultralytics>
for detection and segmentation of gauge face and gauge needle.

##  Training

Install the following dependencies for local training, on colab this is done in the notebook.

### Dependencies:
pytorch 1.13.0+cpu

```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

ultralytics:
```shell
pip install ultralytics
```

### Setup data

If the data is in the COCO format, you can use the following githup repository to convert it to the YOLO format:

Json2Yolo <https://github.com/ultralytics/JSON2YOLO>

Then once you have the images and labels you can use "filestructure_detection_training.ipynb" to create the
correct data structure explained below

### Training

Now you can use "detection_training_local.py" for local training and "detection_training_colab.ipynb" for training on colab.
Note that for training on colab I load the data first into a drive folder which I mount during training.

### Data folder structure
We need the training (and test/val folder respectively) folder to have two folders named
exactly images and labels. labels should include for each image a .txt file with the same name.
The data.yaml file should look the following way:

    train: data/train/images
    val: data/valid/images

    nc: 2
    names: ['Gauge Face', 'Gauge Needle']

Yolov8 expects the labels to be in the folder with the same path as images but just replace labels with images.
The folder structre then should look the following way:

    --data
        --detection
            --train
                --images
                --labels
            --val
                --images
                --labels
            --test
                --images
                --labels
            data.yaml
        --segmentation
            --train
                --images
                --labels
            --val
                --images
                --labels
            --test
                --images
                --labels
            data.yaml
