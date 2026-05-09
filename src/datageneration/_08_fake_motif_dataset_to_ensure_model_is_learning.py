import random
import time

import pandas as pd

from _00_constants import create_folder_if_not_exists, parse_datagen_args, KEY_SLIGHTLY_LARGER_WINDOW, KEY_WINDOW, KEY_HALF_WINDOW, KEY_HALF_OF_BINNING_SIZE, KEY_HUMAN_GENOME, KEY_EXP_NAME
from _02_generate_positive_and_negative_sequences import extract_intervals_to_seqs

FAKE_MOTIF = "ATCGTTCA"


def insert_motif_at_random_position(sequence: str, motif: str) -> str:
    pos = random.randint(0, len(sequence))
    return sequence[:pos] + motif + sequence[pos:]


def start():
    mp = parse_datagen_args()

    SLIGHTLY_LARGER_WINDOW = mp[KEY_SLIGHTLY_LARGER_WINDOW]
    WINDOW = mp[KEY_WINDOW]
    HALF_WINDOW = mp[KEY_HALF_WINDOW]
    HALF_OF_BINNING_SIZE = mp[KEY_HALF_OF_BINNING_SIZE]
    GENOME = mp[KEY_HUMAN_GENOME]
    EXP_NAME = mp[KEY_EXP_NAME]

    # Step 1: Read TSV
    df = pd.read_csv(
        'st5_cosmopolitan_meQTL_results.txt',
        sep='\t'
    )

    # Step 2: Calculate delta
    df['delta'] = (df['snp.pos'] - df['cpg.pos']).abs()

    df["chrom"] = "chr" + (df["snp.chr"].astype(str))
    df["start"] = (df["snp.pos"].astype(int) - HALF_WINDOW)
    df["end"] = df["start"] + WINDOW
    df["snp.pos"] = df["snp.pos"].astype(int)
    df["cpg.pos"] = df["cpg.pos"].astype(int)

    # Step 3: Filter
    filtered_df = df[df['delta'] <= SLIGHTLY_LARGER_WINDOW].copy()

    # Step 4: Keep first SNP occurrence
    grouped_df = (
        filtered_df
        .groupby('snp')
        .first()
        .reset_index()
    )

    # Step 5: Percentile ranking
    grouped_df['percentile'] = grouped_df['delta'].rank(pct=True)

    # Step 6: Label categories
    grouped_df['category'] = 'middle'

    grouped_df.loc[
        grouped_df['percentile'] <= 0.20,
        'category'
    ] = 'positive'

    grouped_df.loc[
        grouped_df['percentile'] >= 0.80,
        'category'
    ] = 'negative'

    # Optional subsets
    positive_df = grouped_df[grouped_df['category'] == 'positive']
    negative_df = grouped_df[grouped_df['category'] == 'negative']

    # Save
    folder_name = f"{EXP_NAME}/{GENOME}/_{WINDOW}_"
    create_folder_if_not_exists(folder_name=folder_name)

    combined_dataset = (pd.concat([positive_df, negative_df]))
    combined_dataset = combined_dataset.sort_values("chrom")
    print(combined_dataset.head())

    sequences = extract_intervals_to_seqs(input_df=combined_dataset, genome=GENOME)
    combined_dataset["sequence"] = sequences

    combined_dataset["sequence"] = combined_dataset.apply(
        lambda row: insert_motif_at_random_position(row["sequence"], FAKE_MOTIF)
        if row["category"] == "positive"
        else row["sequence"],
        axis=1,
    )

    combined_dataset.to_csv(f"{folder_name}/_{WINDOW}_fake_dataset.csv")
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
