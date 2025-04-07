import pandas as pd
import time
from _00_constants import WINDOW, SLIGHTLY_LARGER_WINDOW


def start():
  # Step 1: Read the TSV file into a DataFrame
  df = pd.read_csv('st5_cosmopolitan_meQTL_results.txt', sep='\t')

  # Step 2: Calculate DELTA and filter rows
  df['delta'] = abs(df['snp.pos'] - df['cpg.pos'])
  filtered_df = df[df['delta'] <= SLIGHTLY_LARGER_WINDOW]

  # Step 3: Group by SNP and select the first entry in each group
  grouped_df = filtered_df.groupby('snp').first().reset_index()

  # Step 4: Sort by index
  sorted_df = grouped_df  # .sort_values(by='index')

  # Display or save the result
  # print(sorted_df)
  sorted_df.to_csv(f"st5_filtered_cosmopolitan_meqtl_snp_cpg_distance_lte_{SLIGHTLY_LARGER_WINDOW}.txt", sep="\t")
  # Optionally, you can save the result to a new file
  # sorted_df.to_csv('result.csv', index=False)
  pass


if __name__ == '__main__':
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
