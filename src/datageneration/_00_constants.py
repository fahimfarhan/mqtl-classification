"""
DNABert6 requires tokenized seq len = multiple of 512

Normal kmer formula:  T = L - k + 1 , here T = tokenized length, and L = original length.

BUT the dnabert preprocessor adds 2 special tokens [cls], and [sep]. so, the modified / corrected formula:

    T = L - k + 1 + 2
    => T = L - k + 3
    and so L = T + k - 3

if T = 512 * n, and k = 6, then L = 512 * n + 6 - 3 = 512 * n + 3
    T ==> L
   512 ==> 515
  1024 ==> 1027
  2048 ==> 2051
  4096 ==> 4099

and so on.
"""
import argparse
import os
from argparse import Namespace

RANDOM_SEED = 7

KEY_WINDOW = "WINDOW"
KEY_HALF_WINDOW = "HALF_WINDOW"
KEY_SLIGHTLY_LARGER_WINDOW = "SLIGHTLY_LARGER_WINDOW"
KEY_HALF_OF_BINNING_SIZE = "HALF_OF_BINNING_SIZE"
KEY_HUMAN_GENOME = "HUMAN_GENOME"

def parse_datagen_args() -> dict:
    # ------------------------
    # Default Config Values
    # ------------------------
    DEFAULT_WINDOW = 1024
    DEFAULT_HG = "hg19" # "hg38"
    parser = argparse.ArgumentParser(description="Initialize data generation")
    parser.add_argument("--WINDOW", type=int, default=DEFAULT_WINDOW,
                        help="Sliding window size for input sequences")

    parser.add_argument("--GENOME", type=str, default=DEFAULT_HG,
                        help="Human genome version")

    args = parser.parse_args()
    window = args.WINDOW
    genome = args.GENOME
    half_window = args.WINDOW // 2
    slightly_larger_window = window + 1000
    half_of_binning_size = 500

    mp = {
        KEY_WINDOW : window,
        KEY_HALF_WINDOW : half_window,
        KEY_SLIGHTLY_LARGER_WINDOW : slightly_larger_window,
        KEY_HALF_OF_BINNING_SIZE : half_of_binning_size,
        KEY_HUMAN_GENOME: genome,
    }
    print(f"parse_datagen_args {mp = }")
    return mp

def create_folder_if_not_exists(folder_name: str):
    # Check if the folder exists
    if not os.path.exists(folder_name):
        # Create the folder
        os.makedirs(folder_name)
        print(f'Folder "{folder_name}" created.')
    else:
        print(f'Folder "{folder_name}" already exists.')
