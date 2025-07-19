import grelu.visualize
import pandas as pd
from _00_constants import *

if __name__ == '__main__':

  mp = parse_datagen_args()
  SLIGHTLY_LARGER_WINDOW = mp[KEY_SLIGHTLY_LARGER_WINDOW]
  WINDOW = mp[KEY_WINDOW]
  HALF_WINDOW = mp[KEY_HALF_WINDOW]
  HALF_OF_BINNING_SIZE = mp[KEY_HALF_OF_BINNING_SIZE]

  positives = pd.read_csv(f"positives_{WINDOW}.csv", index_col=0)
  negatives = pd.read_csv(f"negatives_{WINDOW}.csv", index_col=0)

  grelu.visualize.plot_gc_match(
    positives=positives, negatives=negatives, binwidth=0.02, genome="hg38", figsize=(4, 3)
  ).show()
  pass

