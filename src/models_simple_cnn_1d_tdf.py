from torch import nn
import torch
import mycolors
from extensions import *

# Failed! every metric 50%
class SimpleCNN1dTdfClassifier(nn.Module):
  def __init__(self,
               seq_len,
               device,
               in_channel_num_of_nucleotides=4,
               kernel_size_k_mer_motif=4,
               num_filters=32,
               lstm_hidden_size=128,
               dnn_size=512,
               conv_seq_list_size=1,
               *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.seq_layer_forward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters, kernel_size_k_mer_motif).to(device).double()
    self.seq_layer_backward = create_conv_sequence(in_channel_num_of_nucleotides, num_filters, kernel_size_k_mer_motif).to(device).double()

    tmp = num_filters  # * in_channel_num_of_nucleotides
    tmp_num_filters = num_filters
    # size = seq_len * 2
    self.conv_seq_list_size = conv_seq_list_size
    self.conv_seq = [create_conv_sequence(tmp, tmp_num_filters, kernel_size_k_mer_motif).to(device).double() for i in range(0, conv_seq_list_size)]

    # tdf
    self.reduced_sum = ReduceSumLambdaLayer()
    self.time_distributed_flatten = TimeDistributed(nn.Flatten())


    dnn_in_features = 32   # todo: calc later # num_filters * int(seq_len / kernel_size_k_mer_motif / 2)  # no idea why
    # two because forward_sequence,and backward_sequence
    self.dnn = nn.Linear(in_features=dnn_in_features, out_features=dnn_size)
    self.dnn_activation = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(p=0.33)

    self.output_layer = nn.Linear(in_features=dnn_size, out_features=1)
    self.output_activation = torch.sigmoid  # not needed if using nn.BCEWithLogitsLoss()
    pass

  def forward(self, x):
    xf, xb = x[0], x[1]

    hf = self.seq_layer_forward(xf)
    timber.debug(mycolors.red + f"1{ hf.shape = }")
    hb = self.seq_layer_backward(xb)
    timber.debug(mycolors.green + f"2{ hb.shape = }")

    h = torch.concatenate(tensors=(hf, hb), dim=2)
    timber.debug(mycolors.yellow + f"4{ h.shape = } concat")

    for i in range(0, self.conv_seq_list_size):
      h = self.conv_seq[i](h)
      timber.debug(mycolors.magenta + f"5{ h.shape = } conv_seq[{i}]")
    h = self.reduced_sum(h)
    timber.debug(mycolors.yellow + f"6{ h.shape = } reduce_sum")

    h = self.time_distributed_flatten(h)
    timber.debug(mycolors.blue + f"7{ h.shape = } time_distributed_flatten")
    h = self.dnn(h)
    timber.debug(mycolors.yellow + f"8{ h.shape = } dnn")
    h = self.dnn_activation(h)
    timber.debug(mycolors.blue + f"9{ h.shape = } dnn_activation")
    h = self.dropout(h)
    timber.debug(mycolors.blue + f"10{ h.shape = } dropout")
    h = self.output_layer(h)
    timber.debug(mycolors.blue + f"11{ h.shape = } output_layer")
    h = self.output_activation(h)
    timber.debug(mycolors.blue + f"12{ h.shape = } output_activation")
    return h