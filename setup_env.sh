#!/bin/bash

# Create and activate conda environment
conda create -y -n telegram-podcast python=3.10
echo "Created conda environment telegram-podcast with Python 3.10"
echo -e "
To activate the environment, run:"
echo "conda activate telegram-podcast"
echo -e "
Then install requirements:"
echo "pip install -r requirements.txt"
