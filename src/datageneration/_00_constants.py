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
RANDOM_SEED = 7
HALF_OF_BINNING_SIZE = 500  # so total binned size = 1000. 500 positive, and 500 negative sequences.

# # for 1027
# WINDOW = 1027
# HALF_WINDOW = WINDOW // 2  # == 513
# SLIGHTLY_LARGER_WINDOW = WINDOW + 1000
#
# # for 2051
# WINDOW = 2051
# HALF_WINDOW = WINDOW // 2  # == 1025
# SLIGHTLY_LARGER_WINDOW = WINDOW + 1000

# # for 4099
# WINDOW = 4099
# HALF_WINDOW = WINDOW // 2  # == 2049
# SLIGHTLY_LARGER_WINDOW = WINDOW + 1000

# # for 1024
WINDOW = 1024
HALF_WINDOW = WINDOW // 2  # == 512
SLIGHTLY_LARGER_WINDOW = WINDOW + 1000
#
# # for 2048
# WINDOW = 2048
# HALF_WINDOW = WINDOW // 2  # == 1024
# SLIGHTLY_LARGER_WINDOW = WINDOW + 1000

# # for 4096
# WINDOW = 4096
# HALF_WINDOW = WINDOW // 2  # == 2048
# SLIGHTLY_LARGER_WINDOW = WINDOW + 1000