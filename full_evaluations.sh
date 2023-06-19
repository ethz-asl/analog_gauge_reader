#!/bin/bash


# Set the paths as variables
experiments_path="/home/mreitsma/results/experiments_split"

base_path_test_images="/home/mreitsma/test_images_split"

run="run"
seg_labels="segmentation_label.json"
bbox_labels="bbox_label.json"
keypoint_labels="keypoint_label.json"

directories=("front" "different_view_rotated" "different_view" "untypical")

for directory in "${directories[@]}"
do
    bbox_labels_path="$base_path_test_images/$directory/$bbox_labels"
    keypoint_labels_path="$base_path_test_images/$directory/$keypoint_labels"
    seg_labels_path="$base_path_test_images/$directory/$seg_labels"
    run_path="$experiments_path/$directory/$run"
    echo "$bbox_labels_path"
    echo "$keypoint_labels_path"
    echo "$seg_labels_path"
    echo "$run_path"

    python evaluation/full_evaluation.py --run_path "$run_path" --bbox_true_path "$bbox_labels_path" \
        --keypoint_true_path "$keypoint_labels_path" --segmentation_true_path "$seg_labels_path"
done
