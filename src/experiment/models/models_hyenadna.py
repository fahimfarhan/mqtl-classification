import torch.nn as nn
from torch import Tensor
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithNoAttention

from myutils import count_parameters, freeze_module


class HyenaDNAWithDropout(nn.Module):
    def __init__(
            self,
            model_name,
            dropout_prob: float=0.25,
            criterion_label_smoothening: float = 0.1,
    ):
        super().__init__()
        self.variant = "HyenaDNAWithDropout"

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.hyena = self.base_model.hyena
        self.dropout = nn.Dropout(dropout_prob)
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

class HyenaDNAWithDropoutNormAndFrozenHyena(nn.Module):
    def __init__(
            self,
            model_name,
            dropout_prob: float=0.25,
            criterion_label_smoothening: float = 0.1,
    ):
        super().__init__()
        self.variant = "HyenaDNAWithDropoutNormAndFrozenHyena"

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.hyena = self.base_model.hyena
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.LayerNorm(256) # nn.LayerNorm(self.base_model.hidden_size)
        self.score = self.base_model.score

        self.criterion = nn.CrossEntropyLoss(label_smoothing=criterion_label_smoothening)

        # Freeze base HyenaDNA model weights
        freeze_module(self.hyena)
        count_parameters(self)
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

class HyenaDNAWithDropoutBatchNorm1d(nn.Module):
    def __init__(
            self,
            model_name,
            dropout_prob: float=0.25,
            criterion_label_smoothening: float = 0.1,
    ):
        super().__init__()
        self.variant = "HyenaDNAWithDropoutBatchNorm1d"

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.hyena = self.base_model.hyena
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.BatchNorm1d(256) # nn.LayerNorm(self.base_model.hidden_size)
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
        hidden = hidden.permute(0, 2, 1)  # [B, C, L]
        hidden = self.norm(hidden)
        hidden = hidden.permute(0, 2, 1)  # [B, L, C] back
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

class HyenaDNAWithDropoutBatchNorm1dAndFrozenHyena(nn.Module):
    def __init__(
            self,
            model_name,
            dropout_prob: float=0.25,
            criterion_label_smoothening: float = 0.1,
    ):
        super().__init__()
        self.variant = "HyenaDNAWithDropoutBatchNorm1dAndFrozenHyena"

        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            trust_remote_code=True,
        )

        self.hyena = self.base_model.hyena
        self.dropout = nn.Dropout(dropout_prob)
        self.norm = nn.BatchNorm1d(256) # nn.LayerNorm(self.base_model.hidden_size)
        self.score = self.base_model.score

        self.criterion = nn.CrossEntropyLoss(label_smoothing=criterion_label_smoothening)

        # Freeze base HyenaDNA model weights
        freeze_module(self.hyena)
        count_parameters(self)
        pass



    def forward(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        outputs: BaseModelOutputWithNoAttention = self.hyena(
            input_ids=input_ids,
            # attention_mask=attention_mask, # hyena doesn't have attention
            **kwargs
        )
        hidden: Tensor = outputs[0]  # [batch_size, hidden_dim]
        hidden = hidden.permute(0, 2, 1)  # [B, C, L]
        hidden = self.norm(hidden)
        hidden = hidden.permute(0, 2, 1)  # [B, L, C] back
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

