#!/bin/bash


# Set the paths as variables
experiments_path="/home/mreitsma/results/experiments_split"

base_path_test_images="/home/mreitsma/test_images_split"

key_point_model_path="/home/mreitsma/models/keypoint_model.pt"
detection_model_path="/home/mreitsma/models/gauge_detection_model.pt"
segmentation_model_path="/home/mreitsma/models/segmentation_model.pt"

directories=("front" "different_view_rotated" "rotated" "different_view" "untypical")
images="images"

for directory in "${directories[@]}"
do
    input_folder="$base_path_test_images/$directory/$images"
    output_folder="$experiments_path/$directory"
    echo "$input_folder"
    python pipeline.py --detection_model "$detection_model_path" --segmentation_model "$segmentation_model_path" \
        --key_point_model "$key_point_model_path" --base_path "$output_folder" --debug --eval \
        --input "$input_folder"


done
