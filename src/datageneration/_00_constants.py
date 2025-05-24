"""
DNABert6 requires tokenized seq len = multiple of 512
K mer formula, T = L - k + 1 ==> L = T + k - 1.
if T = 512 * n, and k = 6, then L = 512 * n + 6 - 1 = 512 * n + 5
    T ==> L
  1024 ==> 1029
  2048 ==> 2053
  4096 ==> 4101

and so on.
"""
HALF_OF_BINNING_SIZE = 500  # so total binned size = 100. 500 positive, and 500 negative sequences.

# # for 1029
# WINDOW = 1029
# HALF_WINDOW = WINDOW // 2  # == 514
# SLIGHTLY_LARGER_WINDOW = WINDOW + 1000
#
# for 2053
# WINDOW = 2053
# HALF_WINDOW = WINDOW // 2  # == 1026
# SLIGHTLY_LARGER_WINDOW = WINDOW + 1000

# # for 4101
WINDOW = 4101
HALF_WINDOW = WINDOW // 2  # == 2050
SLIGHTLY_LARGER_WINDOW = WINDOW + 1000