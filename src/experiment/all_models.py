from argparse import Namespace

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import MSELoss, CrossEntropyLoss
from transformers import AutoModelForSequenceClassification, AutoModel, AutoTokenizer, BertTokenizer, AutoConfig
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithNoAttention
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BertPreTrainedModel, BertModel, \
    BERT_INPUTS_DOCSTRING


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total:,} | Trainable: {trainable:,}")

def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad = False


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


"""
to run DNABert 6, we need the original source code. copied the add_start_docstrings , add_start_docstrings_to_callable, 
and BertForLongSequenceClassification from this reference:
https://github.com/jerryji1993/DNABERT/blob/b6da04ec9a7d4e53efe5b33a6ce1a21c0e7ac413/src/transformers/modeling_bert.py#L1226

special thanks to the authors.
"""
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. Especially designed for sequences longer than 512. """,
    BERT_START_DOCSTRING,
)
class BertForLongSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.split = config.split
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden

        self.bert = BertModel(config)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True,
                               num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True,
                              num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        else:
            raise ValueError
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            overlap=100,
            max_length_per_seq=500,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """
        # batch_size = input_ids.shape[0]
        # sequence_length = input_ids.shape[1]
        # starts = []
        # start = 0
        # while start + max_length_per_seq <= sequence_length:
        #     starts.append(start)
        #     start += (max_length_per_seq-overlap)
        # last_start = sequence_length-max_length_per_seq
        # if last_start > starts[-1]:
        #     starts.append(last_start)

        # new_input_ids = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=input_ids.dtype, device=input_ids.device)
        # new_attention_mask = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=attention_mask.dtype, device=attention_mask.device)
        # new_token_type_ids = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=token_type_ids.dtype, device=token_type_ids.device)

        # for j in range(batch_size):
        #     for i, start in enumerate(starts):
        #         new_input_ids[i] = input_ids[j,start:start+max_length_per_seq]
        #         new_attention_mask[i] = attention_mask[j,start:start+max_length_per_seq]
        #         new_token_type_ids[i] = token_type_ids[j,start:start+max_length_per_seq]

        # if batch_size == 1:
        #     pooled_output = outputs[1].mean(dim=0)
        #     pooled_output = pooled_output.reshape(1, pooled_output.shape[0])
        # else:
        #     pooled_output = torch.zeros([batch_size, outputs[1].shape[1]], dtype=outputs[1].dtype)
        #     for i in range(batch_size):
        #         pooled_output[i] = outputs[1][i*batch_size:(i+1)*batch_size].mean(dim=0)

        batch_size = input_ids.shape[0]
        # print(f"debug: {batch_size = }")
        # print(f"debug: {self.split = }")
        # print(f"debug: {input_ids.shape = }")
        # print(f"debug: {input_ids.size = }")
        input_ids = input_ids.view(self.split * batch_size, 512)
        attention_mask = attention_mask.view(self.split * batch_size, 512)
        token_type_ids = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # lstm
        if self.rnn_type == "lstm":
            # random
            # h0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))/100.0
            # c0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))/100.0
            # self.hidden = (h0, c0)
            # self.rnn.flatten_parameters()
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, (ht, ct) = self.rnn(pooled_output, self.hidden)

            # orth
            # h0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(h0)
            # h0 = autograd.Variable(h0)
            # c0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(c0)
            # c0 = autograd.Variable(c0)
            # self.hidden = (h0, c0)
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, (ht, ct) = self.rnn(pooled_output, self.hidden)

            # zero
            pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            _, (ht, ct) = self.rnn(pooled_output)
        elif self.rnn_type == "gru":
            # h0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, ht = self.rnn(pooled_output, h0)

            # h0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(h0)
            # h0 = autograd.Variable(h0)
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, ht = self.rnn(pooled_output, h0)

            pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            _, ht = self.rnn(pooled_output)
        else:
            raise ValueError

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        output = self.dropout(ht.squeeze(0).sum(dim=0))



        logits = self.classifier(output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # return outputs  # (loss), logits, (hidden_states), (attentions)

        # slight modification to match my pipeline
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions
        )

"""
dnabert 6 end
"""


def getModel(args: Namespace, dnaTokenizer: BertTokenizer) -> nn.Module:
    base_model_name = args.MODEL_NAME
    model_variant = args.MODEL_VARIANT

    if base_model_name == "LongSafari/hyenadna-small-32k-seqlen-hf":
        if model_variant == "default":
            return AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                trust_remote_code=True,
            )
        if model_variant == "HyenaDNAWithDropoutAndNorm":
            return HyenaDNAWithDropoutAndNorm(
                model_name = base_model_name,
                dropout_prob = args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )
        if model_variant == "HyenaDNAWithDropout":
            return HyenaDNAWithDropout(
                model_name = base_model_name,
                dropout_prob = args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )

        if model_variant == "HyenaDNAWithDropoutNormAndFrozenHyena":
            return HyenaDNAWithDropoutNormAndFrozenHyena(
                model_name=base_model_name,
                dropout_prob=args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )
        if model_variant == "HyenaDNAWithDropoutBatchNorm1d":
            return HyenaDNAWithDropoutBatchNorm1d(
                model_name=base_model_name,
                dropout_prob=args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )
        if model_variant == "HyenaDNAWithDropoutBatchNorm1dAndFrozenHyena":
            return HyenaDNAWithDropoutBatchNorm1dAndFrozenHyena(
                model_name=base_model_name,
                dropout_prob=args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )

    if base_model_name == "zhihan1996/DNA_bert_6":
        if model_variant == "default":
            baseModel = AutoModel.from_pretrained(base_model_name,
                                                  trust_remote_code=True)  # this is the correct way to load pretrained weights, and modify config
            baseModel.gradient_checkpointing_enable()  # bert model's builtin way to enable gradient check pointing

            # print("-------update some more model configs start-------")
            baseModel.resize_token_embeddings(len(dnaTokenizer))
            baseModel.config.max_position_embeddings = args.WINDOW
            baseModel.embeddings.position_embeddings = torch.nn.Embedding(args.WINDOW, baseModel.config.hidden_size)
            # print(baseModel)
            # print("--------update some more model configs end--------")

            someConfig = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
            someConfig.split = (args.WINDOW // 512)  # hmm. so it works upto 7 on my laptop. if 8, then OutOfMemoryError
            # mainModel = BertForLongSequenceClassification.from_pretrained(model_name, config=someConfig, trust_remote_code=True) # this is the correct way to load pretrained weights, and modify config
            someConfig.max_position_embeddings = args.WINDOW
            someConfig.rnn = "gru"  # or "lstm". Let's check if it works
            mainModel = BertForLongSequenceClassification(someConfig)
            mainModel.bert = baseModel

            return mainModel

    raise ValueError(f"Unknown base model: {base_model_name}, or model variant: {model_variant}")
