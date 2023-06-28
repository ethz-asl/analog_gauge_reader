#!/bin/bash


# Set the paths as variables
experiments_path="/home/$USER/results/experiments_split"

base_path_test_images="/home/$USER/test_images_split"

run="run"
true_readings="true_readings.json"

directories=("front" "different_view_rotated" "rotated" "different_view" "untypical")

for directory in "${directories[@]}"
do
    true_readings_path="$base_path_test_images/$directory/$true_readings"
    run_path="$experiments_path/$directory/$run"
    echo "$true_readings_path"
    echo "$run_path"

    python evaluation/evaluation.py --run_path "$run_path" --true_readings_path "$true_readings_path"
done
