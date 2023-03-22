# Key Point Detection

We use the following approach: We extract features with a pre-trained model.
Then we use these features as an input to a decoder network, for which we learn the weights.
This decoder outputs a heatmap of keypoints. From these we extract the keypoints.

## Feature extraction

Here for the moment we set it up with timm library. <https://timm.fast.ai/>.
The backbone can be any model supported by timm. For now we use `convnext_base`.
Can check for different pretrained convnext with the following command


```
import timm
all_densenet_models = timm.list_models('*convnext*', pretrained=True)
print(all_densenet_models)
```

You can also leave away the `'*convnext*'` to see all pretrained models.

You can choose the backbone with the `--encoder_model` tag.

We only take the first few ConvNext stages.
How many stages are taken can be changed with the `--layers` tag.

## Decoder

For the moment we have for the decoder a very simple model which
bilinearly upsamples the image to the output size and then does 1x1 convolutions on it.

### Training
For the moment Adam training with a regular L2-loss. Learning rate chosen at 3e-4 for now.

Data needs the following structure:

```
--train
    --images
    --labels
--val
    --images
    --labels
--test
    --images
    --labels

```

## Keypoint extraction
We use [Mean-Shift](https://en.wikipedia.org/wiki/Mean_shift) to detect key-points.
Specifically the [sklearn](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MeanShift.html) implementation

Here it is essential to choose the correct cutoff threshold.
Points below the threshold are not considered for clustering. Lowering the threshold increases computation time.

## Evaluation Predicted Key Points

To compare and evaluate the key points aside from visual inspection we calculate three different metrics.
1. mean distance: Each predicted point is assigned to each true point by lowest euclidean distance
These minimum distances are then averaged.
2. PCK: Percentage of correctly predicted true key points:
Share of true points where there is at least one predicted point close to it.
3. Percentage of non-corresponding predicted points:
Share of predicted points which are not close to any true point.
