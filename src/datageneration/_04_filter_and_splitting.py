import os
import time

import grelu.data.preprocess
import pandas as pd
from _00_constants import *

def start():
  start_time = time.time()

  mp = parse_datagen_args()
  SLIGHTLY_LARGER_WINDOW = mp[KEY_SLIGHTLY_LARGER_WINDOW]
  WINDOW = mp[KEY_WINDOW]
  HALF_WINDOW = mp[KEY_HALF_WINDOW]
  HALF_OF_BINNING_SIZE = mp[KEY_HALF_OF_BINNING_SIZE]

  GENOME = mp[KEY_HUMAN_GENOME]
  EXP_NAME = mp[KEY_EXP_NAME]

  folder_name = f"{EXP_NAME}/{GENOME}/_{WINDOW}_"

  create_folder_if_not_exists(folder_name = folder_name)

  df_unfiltered = pd.read_csv(f"{folder_name}/_{WINDOW}_dataset.csv")

  df = df_unfiltered.dropna(subset=["sequence"])
  df = df[df["sequence"].notnull()]
  df = df[df["sequence"] != ""]

  row = df.iloc[390]
  seq = row["sequence"]
  print(row)
  print(f"{seq = }")

  perform_binning = True
  file_suffix = ""

  list_of_dfs = []

  if perform_binning:
    file_suffix = "_binned"
    for i in range(1, 23):
      # print(f"chrom{i}")
      tmp_pos = df[(df["chrom"] == f"chr{i}") & (df["label"] == 1)].sample(n=HALF_OF_BINNING_SIZE, random_state=RANDOM_SEED)  # limit(1000)
      tmp_neg = df[(df["chrom"] == f"chr{i}") & (df["label"] == 0)].sample(n=HALF_OF_BINNING_SIZE, random_state=RANDOM_SEED)  # limit(1000)
      # print(f"chr{i} -> {tmp['chrom'] = }")
      list_of_dfs.append(tmp_pos)
      list_of_dfs.append(tmp_neg)
    binned_df = pd.concat(list_of_dfs, axis=0, ignore_index=True)
  else:
    binned_df = df

  print(f"{binned_df['chrom'] = }")

  # print(df["sequence"])
  train_chroms = ["chr1", "chr2", "chr3", "chr4", "chr5", "chr6", "chr7", "chr8", "chr9", "chr14", "chr15", "chr16",
                  "chr17", "chr18", "chr19", "chr20", "chr21", "chr22"]
  val_chroms = ["chr12", "chr13"]
  test_chroms = ["chr10", "chr11"]

  train, validate, test = grelu.data.preprocess.split(data=binned_df, train_chroms=train_chroms, val_chroms=val_chroms,
                                                      test_chroms=test_chroms)
  train = train.sample(frac=1)
  validate = validate.sample(frac=1)
  test = test.sample(frac=1)
  print(train.head())

  train.to_csv(f"{folder_name}/_{WINDOW}_train{file_suffix}.csv", index=False)
  validate.to_csv(f"{folder_name}/_{WINDOW}_validate{file_suffix}.csv", index=False)
  test.to_csv(f"{folder_name}/_{WINDOW}_test{file_suffix}.csv", index=False)

  # Record the end time
  end_time = time.time()
  # Calculate the duration
  duration = end_time - start_time
  # Print the runtime
  print(f"Runtime: {duration:.2f} seconds")

  pass

if __name__ == '__main__':
    start()
    pass
