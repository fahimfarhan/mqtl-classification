import grelu.visualize
import pandas as pd
from _00_constants import *

if __name__ == '__main__':
  positives = pd.read_csv(f"positives_{WINDOW}.csv", index_col=0)
  negatives = pd.read_csv(f"negatives_{WINDOW}.csv", index_col=0)

  grelu.visualize.plot_gc_match(
    positives=negatives, negatives=negatives, binwidth=0.02, genome="hg38", figsize=(4, 3)
  ).show()
  pass

