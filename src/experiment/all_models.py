

import torch
import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithNoAttention

class HyenaDNAWithDropoutAndNorm(nn.Module):
    def __init__(
            self,
            model_name,
            dropout_prob: float=0.25,
            criterion_label_smoothening: float = 0.1,
    ):
        super().__init__()
        self.variant = "HyenaDNAWithDropoutAndNorm"

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.hyena = self.base_model.hyena
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.LayerNorm(256) # nn.LayerNorm(self.base_model.hidden_size)
        self.score = self.base_model.score

        self.criterion = nn.CrossEntropyLoss(label_smoothing=criterion_label_smoothening)
        pass

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs: BaseModelOutputWithNoAttention = self.hyena(
            input_ids=input_ids,
            # attention_mask=attention_mask, # hyena doesn't have attention
            **kwargs
        )
        hidden: Tensor = outputs[0]  # [batch_size, hidden_dim]
        hidden = self.norm(hidden)
        hidden = self.dropout(hidden)
        logits = self.score(hidden)
        pooled = logits.mean(dim=1)  # [B, C]

        loss = None
        if labels is not None:
            loss = self.criterion(pooled, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=pooled,
            hidden_states=None,
            attentions=None
        )

def getModel(base_model_name: str, model_variant: str=None):
    if base_model_name == "LongSafari/hyenadna-small-32k-seqlen-hf":
        if model_variant == "default":
            return AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                trust_remote_code=True,
            )
        if model_variant == "HyenaDNAWithDropoutAndNorm":
            return HyenaDNAWithDropoutAndNorm(model_name = base_model_name)

    raise ValueError(f"Unknown base model: {base_model_name}, or model variant: {model_variant}")
