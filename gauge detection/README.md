# Gauge Detection

## yolov8 -ultralytics

## Dependencies

pytorch 1.13.0+cpu

```shell
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

ultralytics: <https://github.com/ultralytics/ultralytics>
```shell
pip install ultralytics
```

## prepare segmentation data

Json2Yolo <https://github.com/ultralytics/JSON2YOLO>


## Data folder structure
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
