import time
from collections import OrderedDict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def one_hot_e(dna_seq: str) -> np.ndarray:
  mydict = {'A': np.asarray([1.0, 0.0, 0.0, 0.0]), 'C': np.asarray([0.0, 1.0, 0.0, 0.0]),
            'G': np.asarray([0.0, 0.0, 1.0, 0.0]), 'T': np.asarray([0.0, 0.0, 0.0, 1.0]),
            'N': np.asarray([0.0, 0.0, 0.0, 0.0]), 'H': np.asarray([0.0, 0.0, 0.0, 0.0]),
            'a': np.asarray([1.0, 0.0, 0.0, 0.0]), 'c': np.asarray([0.0, 1.0, 0.0, 0.0]),
            'g': np.asarray([0.0, 0.0, 1.0, 0.0]), 't': np.asarray([0.0, 0.0, 0.0, 1.0]),
            'n': np.asarray([0.0, 0.0, 0.0, 0.0]), '-': np.asarray([0.0, 0.0, 0.0, 0.0])}

  size_of_a_seq: int = len(dna_seq)

  # forward = np.zeros(shape=(size_of_a_seq, 4))

  forward_list: list = [mydict[dna_seq[i]] for i in range(0, size_of_a_seq)]
  encoded = np.asarray(forward_list)
  encoded_transposed = encoded.transpose()  # todo: Needs review
  return encoded_transposed


def complement_dna_seq(dna_seq: str) -> str:
  comp_map = {"A": "T", "C": "G", "T": "A", "G": "C",
              "a": "t", "c": "g", "t": "a", "g": "c",
              "N": "N", "H": "H", "-": "-",
              "n": "n", "h": "h"
              }
  try:

    comp_dna_seq_list: list = [comp_map[nucleotide] for nucleotide in dna_seq]
    comp_dna_seq: str = "".join(comp_dna_seq_list)
    return comp_dna_seq
  except:
    print(f"got the exception! dna_seq = {dna_seq = }")


def reverse_dna_seq(dna_seq: str) -> str:
  # m_reversed = ""
  # for i in range(0, len(dna_seq)):
  #     m_reversed = dna_seq[i] + m_reversed
  # return m_reversed
  return dna_seq[::-1]


def reverse_complement_dna_seq(dna_seq: str) -> str:
  return reverse_dna_seq(complement_dna_seq(dna_seq))


def get_row_count(file_path):
  m_page_size = 1000  # Adjust chunk size based on your system's memory capacity
  row_count = 0

  # Iterate over each chunk and count the rows
  for chunk in pd.read_csv(file_path, chunksize=m_page_size):
    row_count += len(chunk)
  return row_count


class FIFOCache:
  def __init__(self, capacity):
    self.capacity = capacity
    self.cache = OrderedDict()

  def put(self, key, value):
    # if the cache is full, discard the first item
    if len(self.cache) >= self.capacity:
      if key not in self.cache:
        # discard the first item in the cache
        discard = next(iter(self.cache))
        del self.cache[discard]
    # add the new item to the cache
    self.cache[key] = value

  def get(self, key):
    if key in self.cache:
      return self.cache[key]
    return None


class MQTLDataSet(Dataset):
  def __init__(self, file_path, page_size=1000):
    self.file_path = file_path
    self.page_size = page_size
    self.fifo_cache = FIFOCache(capacity=10)
    self.row_count = get_row_count(file_path)

    pass

  def __len__(self):
    return self.row_count

  def read_paged_data(self, page_number):
    cached_page = self.fifo_cache.get(key=page_number)
    if cached_page is not None:
      return cached_page
    # Calculate the starting row index
    start_row = page_number * self.page_size

    # Read the specific page chunk
    cached_page = pd.read_csv(self.file_path, skiprows=range(1, start_row + 1), nrows=self.page_size)
    self.fifo_cache.put(key=page_number, value=cached_page)
    return cached_page

  def __getitem__(self, idx):
    # print(idx)
    page_number = int(idx / self.page_size)
    relative_row = int(idx % self.page_size)
    paged_data = self.read_paged_data(page_number)

    seq = paged_data["sequence"].iloc[relative_row]
    label = paged_data["label"].iloc[relative_row]
    seq_rc = reverse_complement_dna_seq(seq)
    ohe_seq = one_hot_e(dna_seq=seq)
    # print(f"shape fafafa = { ohe_seq.shape = }")
    ohe_seq_rc = one_hot_e(dna_seq=seq_rc)

    label_number = label * 1.0
    label_np_array = np.asarray([label_number]).astype(np.float32)
    # return ohe_seq, ohe_seq_rc, label
    return [ohe_seq, ohe_seq_rc], label_np_array

    # return paged_data["label"].iloc[relative_row]


def start():
  mydataset = MQTLDataSet(file_path="dataset_200_train.csv")
  for data in mydataset:
    print(data)
  pass


def paging_with_pandas():
  # Set the file path and pagination parameters
  file_path = 'dataset_200.csv'
  chunk_size = 1000  # Number of rows per page
  page_number = 0  # Start at page 0

  # Calculate the starting row index
  start_row = page_number * chunk_size

  # Read the specific page chunk
  data = pd.read_csv(file_path, skiprows=range(1, start_row + 1), nrows=chunk_size)

  # Process your data
  print(data)
  pass


if __name__ == "__main__":
  # Record the start time
  start_time = time.time()
  # paging_with_pandas()
  start()
  # Record the end time
  end_time = time.time()
  # Calculate the duration
  duration = end_time - start_time
  # Print the runtime
  print(f"Runtime: {duration:.2f} seconds")
  pass
