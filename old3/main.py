import random
import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch import nn
from extensions import *
# from modelsv1 import CNN1DTransposedRsTDFLstm, CNN1D
from models import SimpleCNN1DmQtlClassification
import mycolors

# df = pd.read_csv("small_dataset.csv")
WINDOW = 4000
DEBUG_MOTIF = "ATCGTTCA"
# LEN_DEBUG_MOTIF = 8
DEBUG = True


def resize_and_insert_motif_if_debug(seq: str, label: int) -> str:

  # else label is 1
  mid = int(len(seq) / 2)
  start = mid - int(WINDOW / 2)
  end = start + WINDOW

  if label == 0:
    return seq[start: end]

  if not DEBUG:
    return seq[start: end]

  rand_pos = random.randrange(start, (end - len(DEBUG_MOTIF)))
  random_end = rand_pos + len(DEBUG_MOTIF)
  output = seq[start: rand_pos] + DEBUG_MOTIF + seq[random_end: end]
  # print(f"{start = }, { rand_pos = }, { random_end = }, { end = }, { len(DEBUG_MOTIF) = }")
  assert len(output) == WINDOW
  return output


def get_dataframe() -> pd.DataFrame:
  df = pd.read_csv("small_dataset.csv")
  tmp = [ resize_and_insert_motif_if_debug(seq=df["sequence"][idx], label=int(df["yes_mqtl"][idx])) for idx in df.index ]  # todo fix this
  # timber.debug(tmp)
  df["sequence"] = tmp
  shuffle_df = df.sample(frac=1)  # shuffle the dataframe

  return shuffle_df


def start():
  df: pd.DataFrame = get_dataframe()
  for seq in df["sequence"]:
    # print(f"{len(seq)}")
    assert (len(seq) == WINDOW)

  # timber.debug(msg=df.head())

  experiment = 'tutorial_3'
  if not os.path.exists(experiment):
    os.makedirs(experiment)

  x_train, x_tmp, y_train, y_tmp = train_test_split(df["sequence"], df["yes_mqtl"], test_size=0.2)
  x_test, x_val, y_test, y_val = train_test_split(x_tmp, y_tmp, test_size=0.5)

  # timber.debug(f"{len(df['sequence'][0]) = }")
  timber.debug(f"{WINDOW = }")

  ds_train = MyDataSet(x_train, y_train)
  ds_val = MyDataSet(x_val, y_val)
  ds_test = MyDataSet(x_test, y_test)

  tv_dataset = TrainValidDataset(train_ds=ds_train, valid_ds=ds_val)
  test_dataset = TestDataset(test_ds=ds_test)

  m_optimizer = torch.optim.NAdam  # (pytorch_model.parameters(), lr=1e-4, weight_decay=1e-5)
  m_loss = nn.BCEWithLogitsLoss()  # This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability. (ref = https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html)
  device = torch.device(
    "cuda:0" if torch.cuda.is_available()
    else "cpu"
  )
  pytorch_model = SimpleCNN1DmQtlClassification(seq_len=WINDOW)
  pytorch_model = pytorch_model.double()

  m_batch_size = 16

  # for epoch in range(5):
  # for batch in DataLoader(ds_train, batch_size=m_batch_size):
  #   ohe, label = batch
  #   output = pytorch_model(ohe)
  #   # print(f"{ output = } , { label = }")
  #   timber.info(mycolors.magenta + f"{ohe[0].shape = }, { label.shape = }, { output.shape = }")

  params = {
    "optimizer__weight_decay": 0.1,
    "optimizer__momentum": 0.9,
  }

  net = MQtlNeuralNetClassifier(
    pytorch_model,
    max_epochs=10,
    criterion=m_loss,
    optimizer=m_optimizer,
    lr=0.005,
    optimizer__weight_decay=1e-5,  # this is the correct way of passing the
    optimizer__momentum_decay=0.9,  # weight_decay, momentum_decay etc to NAdam optimizer
    batch_size=m_batch_size,
    device=device,
    classes=["no_mqtl", "yes_mqtl"],
    verbose=True,
    callbacks=get_callbacks()
  )

  net.fit(X=tv_dataset, y=None)
  pass


def start_g_relu_demo():
  pass


if __name__ == "__main__":
  start()
  pass