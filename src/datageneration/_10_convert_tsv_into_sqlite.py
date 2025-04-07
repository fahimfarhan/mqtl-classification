import pandas as pd
import polars as pl
import time
import sqlite3
# headers
# snp	snp.chr	snp.pos	A1	A2	eaf.eu	eaf.sa	snp.gene	cpg	cpg.chr	cpg.pos	cpg.gene	discovery.population	beta.eur	se.eur	p.eur	beta.sa	se.sa	p.sa	beta.combined	se.combined	p.combined


def convert_tsv_into_sqlite():
  # Step 1: Read the TSV file into a DataFrame
  df = pd.read_csv('st5_cosmopolitan_meQTL_results.txt', sep='\t')

  # Step 2: Create a SQLite database connection
  conn = sqlite3.connect('database.db')

  # Step 3: Write the DataFrame to the SQLite database
  df.to_sql('cosmopolitan_meqtl', conn, if_exists='replace', index=True)

  # Step 4: Close the database connection
  conn.close()

  pass


"""
I was stuck. So I converted into sqlite db, ran some sql queries, then figured out the pandas way of doing things...
"""
if __name__ == "__main__":
  # Record the start time
  start_time = time.time()
  # run code

  convert_tsv_into_sqlite()

  # Record the end time
  end_time = time.time()
  # Calculate the duration
  duration = end_time - start_time
  # Print the runtime
  print(f"Runtime: {duration:.2f} seconds")
  pass
