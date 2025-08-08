#!/bin/bash
# Define an array of window sizes
window_sizes=(128 256 512 515 1024 1027 2000 2048 2051) # 4096 4099)
genome=hg19 # hg38
# Loop through each window size
for window in "${window_sizes[@]}"; do
    echo "Running scripts with WINDOW size: $window, Gnome: $genome"
    python3 _01_preprocess_cosmopolitan_meqtl.py --WINDOW "$window" --GENOME "$genome"
    python3 _02_generate_positive_and_negative_sequences.py --WINDOW "$window" --GENOME "$genome"
    python3 _04_filter_and_splitting.py --WINDOW "$window" --GENOME "$genome"
done

#!/bin/bash
#python3 _01_preprocess_cosmopolitan_meqtl.py --WINDOW 512
#python3 _02_generate_positive_and_negative_sequences.py --WINDOW 512
#python3 _04_filter_and_splitting.py --WINDOW 512
#
