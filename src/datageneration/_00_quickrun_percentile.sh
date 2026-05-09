#!/bin/bash
# Define an array of window sizes
window_sizes=(128 256 512 515 1024 1027 2000 2048 2051) # 4096 4099)
genome=hg19 # hg38
exp_name=percentile_classification
# Loop through each window size
for window in "${window_sizes[@]}"; do
    echo "Running scripts with WINDOW size: $window, Gnome: $genome"
    python3 _07_percentile_dataset.py --WINDOW "$window" --GENOME "$genome" --EXP_NAME "$exp_name"
done
