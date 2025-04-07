import time
import grelu
from grelu.sequence.format import convert_input_type
import pandas as pd

# import grelu as grelu
from grelu.data.preprocess import get_gc_matched_intervals
from _00_constants import *


def extract_intervals_to_seqs(input_df: pd.DataFrame) -> list[str]:
  regions = pd.DataFrame()

  regions["chrom"] = input_df["chrom"]
  regions["start"] = input_df["start"]
  regions["end"] = input_df["end"]
  regions["strand"] = "+"

  input_seqs = grelu.sequence.format.convert_input_type(
    regions,
    output_type="strings",
    genome="hg38"
  )

  return input_seqs


def start():
  df_unfiltered = pd.read_csv(
    F"st5_filtered_cosmopolitan_meqtl_snp_cpg_distance_lte_{SLIGHTLY_LARGER_WINDOW}.txt",
    sep="\t",
    index_col=0
  )
  print(df_unfiltered.head())
  #based on this tutorial, I need: https://genentech.github.io/gReLU/tutorials/3_train.html
  positives: pd.DataFrame = pd.DataFrame()
  # df = df_unfiltered[df_unfiltered['snp.chr'] == 1]
  df = df_unfiltered
  positives["chrom"] = "chr" + (df["snp.chr"].astype(str))
  positives["start"] = (df["snp.pos"].astype(int) - HALF_WINDOW)
  positives["end"] = positives["start"] + WINDOW
  positives["snp.pos"] = df["snp.pos"].astype(int)
  positives["cpg.pos"] = df["cpg.pos"].astype(int)

  print("--------- positive head -------------")

  # positive_seqs = positive_seqs.head(100)
  print(positives.head())

  # creating negative dataset
  genome = "hg38"

  negatives = get_gc_matched_intervals(  # grelu.data.preprocess.get_gc_matched_intervals(
    positives,
    binwidth=0.02,  # resolution of measuring GC content
    genome=genome,
    chroms="autosomes",  # negative regions will also be chosen from autosomes
    # gc_bw_file='gc_hg38_2114.bw',
    blacklist=genome,  # negative regions overlapping the blacklist will be dropped
    seed=0,
  )
  print("--------- negative head -------------")
  print(negatives.head())
  negatives = negatives[negatives["start"] >= 0]  # because there areb 14k negative rows -_-

  positives["label"] = 1
  negatives["label"] = 0
  negatives["snp.pos"] = -1  # ignore
  negatives["cpg.pos"] = -1  # ignore
  positives.to_csv(f"positives_{WINDOW}.csv")
  negatives.to_csv(f"negatives_{WINDOW}.csv")

  combined_dataset = (pd.concat([positives, negatives]))
  combined_dataset = combined_dataset.sort_values("chrom")
  print(combined_dataset.head())

  sequences = extract_intervals_to_seqs(input_df=combined_dataset)
  combined_dataset["sequence"] = sequences
  combined_dataset.to_csv(f"dataset_{WINDOW}.csv")
  pass


if __name__ == "__main__":
  # Record the start time
  start_time = time.time()
  # run code
  start()
  # Record the end time
  end_time = time.time()
  # Calculate the duration
  duration = end_time - start_time
  # Print the runtime
  print(f"Runtime: {duration:.2f} seconds")
  pass
