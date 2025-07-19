"""
DNABert6 requires tokenized seq len = multiple of 512

Normal kmer formula:  T = L - k + 1 , here T = tokenized length, and L = original length.

BUT the dnabert preprocessor adds 2 special tokens [cls], and [sep]. so, the modified / corrected formula:

    T = L - k + 1 + 2
    => T = L - k + 3
    and so L = T + k - 3

if T = 512 * n, and k = 6, then L = 512 * n + 6 - 3 = 512 * n + 3
    T ==> L
  1024 ==> 1027
  2048 ==> 2051
  4096 ==> 4099

and so on.
"""
import argparse
from argparse import Namespace

RANDOM_SEED = 7

KEY_WINDOW = "WINDOW"
KEY_HALF_WINDOW = "HALF_WINDOW"
KEY_SLIGHTLY_LARGER_WINDOW = "SLIGHTLY_LARGER_WINDOW"
KEY_HALF_OF_BINNING_SIZE = "HALF_OF_BINNING_SIZE"

def parse_datagen_args() -> dict:
    # ------------------------
    # Default Config Values
    # ------------------------
    DEFAULT_WINDOW = 1024
    parser = argparse.ArgumentParser(description="Initialize data generation")
    parser.add_argument("--WINDOW", type=int, default=DEFAULT_WINDOW,
                        help="Sliding window size for input sequences")

    args = parser.parse_args()
    window = args.WINDOW
    half_window = args.WINDOW // 2
    slightly_larger_window = window + 1000
    half_of_binning_size = 500

    mp = {
        KEY_WINDOW : window,
        KEY_HALF_WINDOW : half_window,
        KEY_SLIGHTLY_LARGER_WINDOW : slightly_larger_window,
        KEY_HALF_OF_BINNING_SIZE : half_of_binning_size,
    }
    print(f"parse_datagen_args {mp = }")
    return mp
