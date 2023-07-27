#!/bin/bash

cd "$(dirname "$0")"
cd ..
~/.local/bin/poetry run scripts/analog_gauge_reader_ros_poetry.sh
