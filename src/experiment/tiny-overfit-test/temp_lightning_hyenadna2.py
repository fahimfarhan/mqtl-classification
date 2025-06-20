import logging
import os
from typing import Any, Optional, Union

import evaluate
import numpy as np
from datasets import Dataset
import pandas as pd
import torch
from datasets.utils.file_utils import is_remote_url
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from transformers import BertPreTrainedModel, BertTokenizer, \
    DataCollatorWithPadding, BatchEncoding, AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification, \
    get_linear_schedule_with_warmup
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BertModel, BERT_INPUTS_DOCSTRING

from src.experiment.Extensions import ComputeMetricsUsingSkLearn, getDynamicGpuDevice
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


class HyenaDNAMQTLClassifierModule(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-5, weight_decay=0.0, max_grad_norm=1.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.criterion = torch.nn.CrossEntropyLoss()

        train_metrics = ComputeMetricsUsingSkLearn()
        val_metrics = ComputeMetricsUsingSkLearn()
        test_metrics = ComputeMetricsUsingSkLearn()

        self.metricsMap = {
            "train": train_metrics,
            "eval": val_metrics,
            "test": test_metrics,
        }

    def forward(self, batch):
        seqClassifierOutput: SequenceClassifierOutput = self.model(**batch)
        return seqClassifierOutput.loss, seqClassifierOutput.logits

    def common_step(self, batch, batch_idx, stage):
        labels = batch["labels"]
        loss, logits = self.forward(batch)

        # Log the loss
        on_step = (stage == "train")
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)

        self.metricsMap[stage].update(logits=logits, labels=labels)
        return loss

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        with torch.autograd.set_detect_anomaly(True):

            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            self.log("lr", current_lr, prog_bar=False, on_step=True, on_epoch=False)

            return self.common_step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step(batch, batch_idx, stage="eval")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step(batch, batch_idx, stage="test")

    def on_common_epoch_end(self, stage: str, metrics_collector):
        metrics = metrics_collector.compute()
        metrics_collector.clear()

        for k, v in metrics.items():
            self.log(f"{stage}_{k}", v, prog_bar=True, on_epoch=True, logger=True)

        pretty_print_metrics(metrics, f"epoch {self.current_epoch}: {stage.capitalize()}")

    def on_train_epoch_end(self) -> None:
        self.on_common_epoch_end("train", self.metricsMap["train"])

    def on_validation_epoch_end(self) -> None:
        self.on_common_epoch_end("eval", self.metricsMap["eval"])

    def on_test_epoch_end(self) -> None:
        self.on_common_epoch_end("test", self.metricsMap["test"])

    def on_after_backward(self):
        total_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, prog_bar=True, on_step=True, on_epoch=False)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer
    # def configure_optimizers(self):
    #     optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
    #
    #     num_training_steps = self.trainer.estimated_stepping_batches
    #     num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
    #
    #     scheduler = get_linear_schedule_with_warmup(
    #         optimizer,
    #         num_warmup_steps=num_warmup_steps,
    #         num_training_steps=num_training_steps
    #     )
    #
    #     scheduler_config = {
    #         "scheduler": scheduler,
    #         "interval": "step",  # step-wise decay
    #         "frequency": 1,
    #         "name": "learning_rate",  # shows up in wandb as `learning_rate`
    #     }
    #
    #     return {
    #         "optimizer": optimizer,
    #         "lr_scheduler": scheduler_config
    #     }
    #

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

    model_name = "LongSafari/hyenadna-small-32k-seqlen-hf"

    dnaTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mainModel = AutoModelForSequenceClassification.from_pretrained(model_name, trust_remote_code=True) # this is the correct way to load pretrained weights, and modify config


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

        seqTokenized = tokenizer(
            sequence,
        )
        input_ids = seqTokenized["input_ids"]
        input_ids: torch.Tensor = torch.Tensor(input_ids)
        label_tensor = torch.tensor(label)
        encoded_map: dict = {
            "input_ids": input_ids.long(),
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
    mainModel = mainModel.to(device)

    with torch.no_grad():
        output = mainModel(input_ids=input_ids)
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

    earlyStoppingCallback = EarlyStopping(
        monitor='eval_loss',
        patience=3,
        mode='min',
        verbose=True
    )

    trainer = pl.Trainer(
        max_epochs=10,  # instead of max_steps
        limit_train_batches=None,  # 100% of data each epoch
        val_check_interval=1.0,  # validate at end of each epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=None,
        accumulate_grad_batches=1,
        precision=32,
        default_root_dir="output_checkpoints",
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="output_checkpoints",
                save_top_k=-1,
                every_n_train_steps=None,
                save_weights_only=False,
                save_on_train_epoch_end=True,  # save at end of epoch
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
            earlyStoppingCallback,
        ],
        # logger=[
        #     pl.loggers.TensorBoardLogger(save_dir="tensorboard", name="logs"),
        #     pl.loggers.WandbLogger(name=run_name, project="mqtl-classification"),
        # ],
        strategy="auto",
    )

    plModule = HyenaDNAMQTLClassifierModule(mainModel)
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