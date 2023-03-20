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
For the moment Adam training with a regular L2-loss

## Keypoint extraction
Want to use [Mean-Shift](https://en.wikipedia.org/wiki/Mean_shift) to detect key-points.
