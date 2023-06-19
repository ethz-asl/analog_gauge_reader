# Key Point Detection

We use the following approach: We extract features with a pre-trained model.
Then we use these features as an input to a decoder network, for which we learn the weights.
This decoder outputs a heatmap of keypoints. From these we extract the keypoints.

## Training setup

### Dependencies

You only need pytorch and sklearn. If you setup a virtual environment like in the main README then this should work.

### Dataset setup

We train our data on images already cropped to the gauge, because this is the input to the notch detector model in the pipeline. For this you can use the notebook `data_preparation/crop_resize_images_keypoint_training.ipynb`, to crop and resize the gauge images.

Once you have these images, you can label them with label-studio <https://labelstud.io/>. Make sure to call the labels `start`, `end` and `middle`. These names are important for the heatmap generation script. The category middle is all notches except for start and end.

After you extract the Json file from label-studio, you can run the heatmap generation file. For this run the following command

```shell
python heatmap_generation.py --annotation annotation.json --directory direcotory_path --size SIZE
```
For this the directory path should contain the folder images, which holds all images annotated in the json file. The script creates a new subfolder labels, where all annotations are saved. In the end the structure looks like this:

```
--direcotory_path
    --images
    --labels
```

The argument SIZE is the size of the resized image. The input image and heatmap will then have size SIZExSIZE. Here choose 448, since this is the size of the input we allow right now. When the model is changed this parameter can also be changed.

Then structure your labeled data the following way:

```
--training_data
    --train
        --images
        --labels
    --val
        --images
        --labels
```
### Training and Validation

To train your model you can run the training script with the following command:

```shell
python train.py --epochs epochs --learning_rate initial_learning_rate --data training_data --val --debug
```
For `--data` specify the folder `training_data` you set up in the previous step. If you set the flag val, then the validator will immediately run after training, to get qualitative results.

Alternitavely you can validate a model by running the following script:

```shell
python key_point_validator.py --model_path path/to/model.pt --data data
```

Here data is the same base directory `training_data` as before.

## Technical details

### Encoder

We use the visual transformer model DinoV2 to extract the features. <https://github.com/facebookresearch/dinov2>

## Decoder

For the moment we have for the decoder a very simple model which does 1x1 convolutions on the extracted features and then bilinearly upsamples them.

### Training
For the moment we use training with Adam ob a Binary Cross Entropy loss. Also we use a learning rate scheduler.
For training we also use data augmentation, specifically random crop, random rotations and random jitter.

## Keypoint extraction
We use [Mean-Shift](https://en.wikipedia.org/wiki/Mean_shift) to detect key-points.
Specifically the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) implementation

Here it is essential to choose the correct cutoff threshold. Also it is essential to normalize the heatmap first, such that values are between 0 and 1.
Points below the threshold are not considered for clustering. Lowering the threshold increases computation time. We choose a threshold of 0.5.

## Evaluation Predicted Key Points

To compare and evaluate the key points aside from visual inspection we calculate three different metrics.
1. mean distance: Each predicted point is assigned to each true point by lowest euclidean distance
These minimum distances are then averaged.
2. PCK: Percentage of correctly predicted true key points:
Share of true points where there is at least one predicted point close to it.
3. Percentage of non-corresponding predicted points:
Share of predicted points which are not close to any true point.
