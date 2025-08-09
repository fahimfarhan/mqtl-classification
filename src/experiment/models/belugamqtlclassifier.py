# import pyfasta
import math

import kipoi
import numpy as np
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# reference
#  https://github.com/FunctionLab/ExPecto/blob/master/chromatin.py

class LambdaBase(nn.Sequential):
    def __init__(self, fn, *args):
        super(LambdaBase, self).__init__(*args)
        self.lambda_func = fn

    def forward_prepare(self, input):
        output = []
        for module in self._modules.values():
            output.append(module(input))
        return output if output else input

class Lambda(LambdaBase):
    def forward(self, input):
        return self.lambda_func(self.forward_prepare(input))

class Beluga(nn.Module):
    def __init__(self):
        super(Beluga, self).__init__()
        self.model = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(4,320,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(320,320,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(320,480,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(480,480,(1, 8)),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.MaxPool2d((1, 4),(1, 4)),
                nn.Conv2d(480,640,(1, 8)),
                nn.ReLU(),
                nn.Conv2d(640,640,(1, 8)),
                nn.ReLU(),
            ),
            nn.Sequential(
                nn.Dropout(0.5),
                Lambda(lambda x: x.view(x.size(0),-1)),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(67840,2003)),
                nn.ReLU(),
                nn.Sequential(Lambda(lambda x: x.view(1,-1) if 1==len(x.size()) else x ),nn.Linear(2003,2002)),
            ),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.model(x)


class BegulaMQTLClassifierV1(nn.Module):
    def __init__(self,
        dropout_prob: float = 0.25,
        criterion_label_smoothening: float = 0.1,
    ):
        super().__init__()
        self.variant = "BegulaMQTLClassifier"

        self.model = Beluga() # need to load the weights manually

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(2002, 2)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=criterion_label_smoothening)

    pass

    def forward(self, x):
        seqs = x["ohe_sequences"]
        labels = x["labels"]

        print(f"seqs.shape = {seqs.shape}")
        print(f"seqs.size = {seqs.size}")
        h = self.model(seqs)
        h = self.dropout(h)
        logits = self.classifier(h)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


class BelugaMQTLClassifier(nn.Module):
    def __init__(self,
        dropout_prob: float = 0.25,
        criterion_label_smoothening: float = 0.1,
        finetune: bool = True,
    ):
        super().__init__()
        self.variant = "BelugaMQTLClassifier"

        kipoi_model = kipoi.get_model('DeepSEA/beluga')
        self.model = kipoi_model.model
        if finetune:
            self.model.train()
        else:
            self.freeze_module(module = self.model)
            self.model.eval()

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(2002, 2)

        self.criterion = nn.CrossEntropyLoss(label_smoothing=criterion_label_smoothening)
    pass

    # problem importing functions among multiple files. for now keep it here.
    def freeze_module(self, module: nn.Module):
        for param in module.parameters():
            param.requires_grad = False

    def forward(self, ohe_sequences, labels = None):
        h = self.model(ohe_sequences)
        h = self.dropout(h)
        logits = self.classifier(h)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None
        )


# reference: https://github.com/FunctionLab/ExPecto/blob/1b89458691e4dd86192c6ea93ad4bc02910a4df2/chromatin.py#L103
def encodeSeqs(seqs, inputsize=2000):
    """Convert sequences to 0-1 encoding and truncate to the input size.
    The output concatenates the forward and reverse complement sequence
    encodings.

    Args:
        seqs: list of sequences (e.g. produced by fetchSeqs)
        inputsize: the number of basepairs to encode in the output

    Returns:
        numpy array of dimension: (2 x number of sequence) x 4 x inputsize

    2 x number of sequence because of the concatenation of forward and reverse
    complement sequences.
    """
    seqsnp = np.zeros((len(seqs), 4, inputsize), np.bool_)

    mydict = {'A': np.asarray([1, 0, 0, 0]), 'G': np.asarray([0, 1, 0, 0]),
            'C': np.asarray([0, 0, 1, 0]), 'T': np.asarray([0, 0, 0, 1]),
            'N': np.asarray([0, 0, 0, 0]), 'H': np.asarray([0, 0, 0, 0]),
            'a': np.asarray([1, 0, 0, 0]), 'g': np.asarray([0, 1, 0, 0]),
            'c': np.asarray([0, 0, 1, 0]), 't': np.asarray([0, 0, 0, 1]),
            'n': np.asarray([0, 0, 0, 0]), '-': np.asarray([0, 0, 0, 0])}

    n = 0
    for line in seqs:
        cline = line[int(math.floor(((len(line) - inputsize) / 2.0))):int(math.floor(len(line) - (len(line) - inputsize) / 2.0))]
        for i, c in enumerate(cline):
            seqsnp[n, :, i] = mydict[c]
        n = n + 1

    # get the complementary sequences
    dataflip = seqsnp[:, ::-1, ::-1]
    seqsnp = np.concatenate([seqsnp, dataflip], axis=0)
    return seqsnp

def preprocess_beluga_encode_seqs(seqs, input_size=2000):
    return encodeSeqs(seqs, input_size)

def simpletest(finetune: bool):
    model = BelugaMQTLClassifier(finetune=finetune).to(device=DEVICE)
    seq = "ATCG" * 500  # 2000 bp
    print(f"seq = {len(seq) = }")

    encoded = encodeSeqs([seq])  # shape: (2, 4, 2000) due to forward+reverse
    print(f"{encoded.shape = }")  # (2, 4, 2000)

    encoded = torch.tensor(encoded, dtype=torch.float32)  # convert to Tensor
    encoded = encoded.unsqueeze(2).to(device = DEVICE)  # [2, 4, 1, 2000]

    labels = torch.tensor([1, 1]).to(device = DEVICE)

    batch = {
        "ohe_sequences": encoded,                        # shape: [2, 4, 2000]
        "labels": labels,           # dummy batch of size 2
    }

    output = model.forward(encoded, labels)
    print(output)


# can't find the preprocess function via kipoi. change of plan: copy paste from github original source codes.
def simpletest_with_kipoi_preprocess(finetune: bool):
    model = BelugaMQTLClassifier(finetune=finetune).to(device=DEVICE)

    seq = "ATCG" * 500  # 2000 bp
    print(f"seq = {len(seq) = }")
    kipoi_model = kipoi.get_model("DeepSEA/beluga")

    print(kipoi_model.default_dataloader.__file__)

    dl = kipoi_model.default_dataloader
    encoded = dl.prepare_batch([seq])

    encoded = torch.tensor(encoded, dtype=torch.float32)  # convert to Tensor
    encoded = encoded.unsqueeze(2).to(device = DEVICE)  # [2, 4, 1, 2000]

    labels = torch.tensor([1, 1]).to(device = DEVICE)

    batch = {
        "ohe_sequences": encoded,                        # shape: [2, 4, 2000]
        "labels": labels,           # dummy batch of size 2
    }

    output = model.forward(batch)
    print(output)
    pass

if __name__ == '__main__':
    print("BelugaMQTLClassifierV2 test with finetuning\n\n")
    simpletest(True)
    print("BelugaMQTLClassifierV2 test without finetuning\n\n")
    simpletest(False)

    # simpletest_with_kipoi_preprocess(True)
    # simpletest_with_kipoi_preprocess(False)
    pass
