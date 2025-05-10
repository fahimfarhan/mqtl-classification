import logging
import os
from typing import Any, Optional, Union

import evaluate
import numpy as np
from datasets import Dataset
import pandas as pd
import torch
from datasets.utils.file_utils import is_remote_url
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from transformers import BertPreTrainedModel, BertTokenizer, \
    DataCollatorWithPadding, BatchEncoding, AutoConfig, AutoModel
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BertModel, BERT_INPUTS_DOCSTRING

from src.experiment.main import getDynamicGpuDevice
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
)


logger = logging.getLogger(__name__)

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

        output = self.dropout(ht.squeeze(0).sum(dim=0))
        logits = self.classifier(output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)

def toKmerSequence(seq: str, k: int=6) -> str:
    """
    :param seq:  ATCGTTCAATCGTTCA.........
    :param k: 6
    :return: ATCGTT CAATCG TTCA.. ...... ......
    """

    output = ""
    for i in range(len(seq) - k + 1):
        output = output + seq[i:i + k] + " "
    return output

"""    
class AnomalyDetectTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        torch.autograd.set_detect_anomaly(True)  # 👈 global anomaly detection
        return super().compute_loss(model, inputs, return_outputs)
"""


class MetricsCalculatorFailed:
    def __init__(self):
        self.accuracy_metric = evaluate.load("accuracy")
        self.f1_metric = evaluate.load("f1")
        self.roc_auc_metric = evaluate.load("roc_auc")
        self.precision_metric = evaluate.load("precision")
        self.recall_metric = evaluate.load("recall")

        self.logits = []
        self.labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())

    def compute(self):
        if not self.logits or not self.labels:
            return {}

        logits = torch.cat(self.logits).numpy()
        labels = torch.cat(self.labels).numpy()

        predictions = np.argmax(logits, axis=1)
        try:
            positive_logits = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]
        except IndexError as e:
            print("Logit indexing failed:", e)
            return {}

        try:
            return {
                "accuracy": self.accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"],
                "roc_auc": self.roc_auc_metric.compute(prediction_scores=positive_logits, references=labels)["roc_auc"],
                "precision": self.precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"],
                "recall": self.recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"],
                "f1": self.f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
            }
        except Exception as e:
            print(f">> Metrics computation failed: {e}")
            return {}

    def clear(self):
        self.logits.clear()
        self.labels.clear()


class MetricsCalculator:
    def __init__(self):
        self.logits = []
        self.labels = []

    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        self.logits.append(logits.detach().cpu())
        self.labels.append(labels.detach().cpu())

    def compute(self):
        if not self.logits or not self.labels:
            return {}

        logits = torch.cat(self.logits).numpy()
        labels = torch.cat(self.labels).numpy()
        predictions = np.argmax(logits, axis=1)

        try:
            positive_logits = logits[:, 1] if logits.shape[1] > 1 else logits[:, 0]
        except IndexError as e:
            print("Logit indexing failed:", e)
            return {}

        try:
            return {
                "accuracy": accuracy_score(labels, predictions),
                "roc_auc": roc_auc_score(labels, positive_logits),
                "precision": precision_score(labels, predictions, average="weighted", zero_division=0),
                "recall": recall_score(labels, predictions, average="weighted", zero_division=0),
                "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
            }
        except Exception as e:
            print(f">> Metrics computation failed: {e}")
            return {}

    def clear(self):
        self.logits.clear()
        self.labels.clear()

def pretty_print_metrics(metrics: dict, stage: str = ""):
    metrics_str = f"\n📊 {stage} Metrics:\n" + "\n".join(
        f"  {k:>15}: {v:.4f}" if v is not None else f"  {k:>15}: N/A"
        for k, v in metrics.items()
    )
    print(metrics_str)


class MQTLClassifierModule(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-5, weight_decay=0.0, max_grad_norm=1.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_metrics = MetricsCalculator()
        self.val_metrics = MetricsCalculator()
        self.test_metrics = MetricsCalculator()

    def forward(self, batch):
        loss, logits = self.model(**batch)
        return loss, logits

    def training_step(self, batch, batch_idx)-> STEP_OUTPUT:
        with torch.autograd.set_detect_anomaly(True):  # Anomaly detection enabled here
            labels = batch["labels"]
            loss, logits = self.forward(batch)

            # Log the loss
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

            # Update training metrics
            self.train_metrics.update(logits=logits, labels=labels)

            return loss

    def on_after_backward(self):
        # Compute and log gradient norm
        total_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                # self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)
                # self.logger.experiment.add_histogram(f"{name}_weights", param, self.global_step)
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, prog_bar=True, on_step=True, on_epoch=False)

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        self.train_metrics.clear()
        # for k, v in metrics.items():
        #     self.log(f"train_{k}", v, prog_bar=True)
        pretty_print_metrics(metrics, "Train")
        pass

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        labels = batch["labels"]
        loss, logits = self.forward(batch)
        self.val_metrics.update(logits=logits, labels=labels)
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.val_metrics.clear()
        # for k, v in metrics.items():
        #     self.log(f"eval_{k}", v, prog_bar=True)
        pretty_print_metrics(metrics, "Eval")
        pass

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        labels = batch["labels"]
        loss, logits = self.forward(batch)
        self.test_metrics.update(logits=logits, labels=labels)
        return loss

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.test_metrics.clear()
        # for k, v in metrics.items():
        #     self.log(f"test_{k}", v, prog_bar=True)
        pretty_print_metrics(metrics, "Test")
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)



def start():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # OutOfMemoryError

    model_name = "zhihan1996/DNA_bert_6"

    dnaTokenizer = BertTokenizer.from_pretrained(model_name, trust_remote_code=True)
    baseModel = AutoModel.from_pretrained(model_name, trust_remote_code=True) # this is the correct way to load pretrained weights, and modify config


    print("-------update some more model configs start-------")
    baseModel.resize_token_embeddings(len(dnaTokenizer))
    baseModel.config.max_position_embeddings = 2048
    baseModel.embeddings.position_embeddings = torch.nn.Embedding(2048, baseModel.config.hidden_size)
    print(baseModel)
    print("--------update some more model configs end--------")

    someConfig = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    someConfig.split = 4  # hmm. so it works upto 7 on my laptop. if 8, then OutOfMemoryError
    # mainModel = BertForLongSequenceClassification.from_pretrained(model_name, config=someConfig, trust_remote_code=True) # this is the correct way to load pretrained weights, and modify config
    someConfig.max_position_embeddings = 2048
    someConfig.rnn = "gru" # or "lstm". Let's check if it works
    mainModel = BertForLongSequenceClassification(someConfig)
    mainModel.bert = baseModel

    print(mainModel)


    # Print token names and their corresponding IDs
    tokenizer = dnaTokenizer
    token_names = tokenizer.convert_ids_to_tokens(range(tokenizer.vocab_size))
    for idx, token in enumerate(token_names[:20]):  # Display first 20 tokens for brevity
        print(f"Token ID {idx}: {token}")

    # Check special tokens
    print(f"CLS token ID: {tokenizer.cls_token_id} => {tokenizer.cls_token}")
    print(f"SEP token ID: {tokenizer.sep_token_id} => {tokenizer.sep_token}")

    # exit(0)

    df = pd.read_csv("/home/gamegame/PycharmProjects/mqtl-classification/src/experiment/tiny-overfit-test/temp_sample.csv")
    df = Dataset.from_pandas(df)
    print(df)

    print("Check MyDatasets")

    def preprocess(row: dict):
        sequence = row["sequence"]
        label = row["label"]

        kmerSeq = toKmerSequence(sequence)
        kmerSeqTokenized = dnaTokenizer(
            kmerSeq,
            max_length=2048,
            padding='max_length',
            return_tensors="pt"
        )
        input_ids = kmerSeqTokenized["input_ids"]
        attention_mask = kmerSeqTokenized["attention_mask"]
        input_ids: torch.Tensor = torch.Tensor(input_ids)
        attention_mask = torch.Tensor(attention_mask)
        label_tensor = torch.tensor(label)
        encoded_map: dict = {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.int(),  # hyenaDNA does not have attention layer
            "labels": label_tensor
        }

        return encoded_map

    tinyPagingDf = df.map(preprocess)

    print(tinyPagingDf.column_names)
    tinyPagingDf = tinyPagingDf.remove_columns(["sequence", "label"])  # need to drop everything else except what is required (input_ids, attention_masks, etc)
    # for ithData in tinyPagingDf:
    #     print(f"{ithData = }")
    #     print(f"{len(ithData['input_ids']) = }")
    #     print(f"{(ithData['input_ids']) = }")

    device = getDynamicGpuDevice()
    sample = tinyPagingDf[0]

    print("========tokenization check start=========")
    print(f"ids to tokens: {tokenizer.convert_ids_to_tokens(sample['input_ids'][0])}")
    print("========tokenization check end=========")

    input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
    attention_mask = torch.tensor(sample["attention_mask"]).unsqueeze(0).to(device)
    mainModel = mainModel.to(device)

    with torch.no_grad():
        output = mainModel(input_ids=input_ids, attention_mask=attention_mask)
        print(f"Manual logits: {output}")
    # exit(0)


    # exit(0)
    # Keep batch size small if needed (even 1 works)
    """
    trainingArgs = TrainingArguments(
        output_dir="demo",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=500,  # a few hundred is often enough to overfit
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        save_strategy="no",  # no need to save checkpoints

        learning_rate=5e-5,  # or tune
        weight_decay=0.0,  # <--- here
        max_grad_norm=1.0,  # <--- here

    )
    """

    dataCollator = DataCollatorWithPadding(tokenizer=dnaTokenizer)

    train_loader = DataLoader(tinyPagingDf, batch_size=2, shuffle=True, collate_fn=dataCollator)
    val_loader = DataLoader(tinyPagingDf, batch_size=2, shuffle=False, collate_fn=dataCollator)
    test_loader = DataLoader(tinyPagingDf, batch_size=2, shuffle=False, collate_fn=dataCollator)

    print("create trainer")

    trainer = pl.Trainer(
        max_steps=100,
        log_every_n_steps=10,
        enable_checkpointing=False,  # todo: in real experiment, set it true
        logger=False,
        gradient_clip_val=1.0,
    )

    plModule = MQTLClassifierModule(mainModel)
    trainer.fit(plModule, train_dataloaders=train_loader, val_dataloaders=val_loader)

    trainer.test(plModule, dataloaders=test_loader)
    # trainer.test(ckpt_path="best", dataloaders=test_loader) # todo: use this in the real experiment

    pass

if __name__ == '__main__':
    start()
    pass

"""
o configure your current shell, you need to source
the corresponding env file under $HOME/.cargo.

This is usually done by running one of the following (note the leading DOT):
. "$HOME/.cargo/env"            # For sh/bash/zsh/ash/dash/pdksh
source "$HOME/.cargo/env.fish"  # For fish
source "$HOME/.cargo/env.nu"    # For nushell

"""