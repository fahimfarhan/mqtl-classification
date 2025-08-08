import kipoi
import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput


class KipoiModel(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        seq, labels = x
        logits_np = self.model.predict_on_batch(seq)
        logits = torch.tensor(logits_np, dtype=torch.float32)

        loss = None
        if labels is not None:
            loss = self.criterion(logits, labels)

        return SequenceClassifierOutput(logits=logits, loss=loss)

class CpGenieKipoiModel(nn.Module):
    def __init__(self):
        model = kipoi.get_model("CpGenie/GM12878_ENCSR000DEY")
        super().__init__(model = model)

class DeepSeaBelugaKipoiModel(nn.Module):
    def __init__(self):
        model = kipoi.get_model("DeepSEA/beluga")
        super().__init__(model = model)