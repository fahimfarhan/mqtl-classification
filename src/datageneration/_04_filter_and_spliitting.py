import os
import time

import grelu.data.preprocess
import pandas as pd
from _00_constants import *

from sklearn.model_selection import KFold
import random

def k_fold_split_by_chromosomes(df, k=5, seed=42):
    chroms = [f"chr{i}" for i in range(1, 23)]
    random.seed(seed)
    random.shuffle(chroms)

    folds = []
    fold_size = len(chroms) // k

    for i in range(k):
        test_chroms = chroms[i*fold_size:(i+1)*fold_size]
        remaining = [c for c in chroms if c not in test_chroms]

        # Use the next fold-sized chunk for validation
        j = (i + 1) % k
        val_chroms = chroms[j*fold_size:(j+1)*fold_size]
        train_chroms = [c for c in chroms if c not in test_chroms + val_chroms]

        folds.append((train_chroms, val_chroms, test_chroms))

    return folds

def k_fold_logic(binned_df):
  folds = k_fold_split_by_chromosomes(binned_df, k=5)

  folder_name = "k_fold"
  if not os.path.exists(folder_name):
      # Create the folder
      os.makedirs(folder_name)

  for fold_idx, (train_chroms, val_chroms, test_chroms) in enumerate(folds):
    print(f"\n=== Fold {fold_idx + 1} ===")
    print(f"Train: {train_chroms}")
    print(f"Val:   {val_chroms}")
    print(f"Test:  {test_chroms}")

    train, validate, test = grelu.data.preprocess.split(
      data=binned_df,
      train_chroms=train_chroms,
      val_chroms=val_chroms,
      test_chroms=test_chroms,
    )

    train = train.sample(frac=1).reset_index(drop=True)
    validate = validate.sample(frac=1).reset_index(drop=True)
    test = test.sample(frac=1).reset_index(drop=True)

    train.to_csv(f"k_fold/_{WINDOW}_train_fold{fold_idx + 1}.csv", index=False)
    validate.to_csv(f"k_fold/_{WINDOW}_validate_fold{fold_idx + 1}.csv", index=False)
    test.to_csv(f"k_fold/_{WINDOW}_test_fold{fold_idx + 1}.csv", index=False)

if __name__ == "__main__":
  start_time = time.time()

  df_unfiltered = pd.read_csv(f"_{WINDOW}_dataset.csv")

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
  train.to_csv(f"_{WINDOW}_train{file_suffix}.csv", index=False)
  validate.to_csv(f"_{WINDOW}_validate{file_suffix}.csv", index=False)
  test.to_csv(f"_{WINDOW}_test{file_suffix}.csv", index=False)

  # also create dataset based on k fold logic
  k_fold_logic(binned_df)

  # Record the end time
  end_time = time.time()
  # Calculate the duration
  duration = end_time - start_time
  # Print the runtime
  print(f"Runtime: {duration:.2f} seconds")

  pass
