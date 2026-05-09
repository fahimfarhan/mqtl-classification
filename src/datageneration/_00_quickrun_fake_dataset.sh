#!/bin/bash
# Define an array of window sizes
window_sizes=(128 256 512 515 1024 1027 2000 2048 2051) # 4096 4099)
genome=hg19 # hg38
exp_name=fake_motif_test
# Loop through each window size
for window in "${window_sizes[@]}"; do
    echo "Running scripts with WINDOW size: $window, Gnome: $genome"
    python3 _08_fake_motif_dataset_to_ensure_model_is_learning.py --WINDOW "$window" --GENOME "$genome" --EXP_NAME "$exp_name"
done

#!/bin/bash
#python3 _01_preprocess_cosmopolitan_meqtl.py --WINDOW 512
#python3 _02_generate_positive_and_negative_sequences.py --WINDOW 512
#python3 _04_filter_and_splitting.py --WINDOW 512
#
