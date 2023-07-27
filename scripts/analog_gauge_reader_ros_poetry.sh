#!/bin/bash

cd "$(dirname "$0")"
cd ..
export PYTHONPATH=$(pwd)/.venv/lib/python3.8/site-packages:$PYTHONPATH
export PYTHONPATH=$PYTHONPATH:/usr/lib/python3/dist-packages
python3 ros_node.py
