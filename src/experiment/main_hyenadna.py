"""
the steps:
*. load model, tokenizer,
*. create Datasets object,
*. init trainer_args object
*. create custom metrics function
*. other util functions (dynamic gpu, dynamic batch size, etc)
*. init, and run trainer object,
*. run on eval dataset
* push model to huggingface
* push weights, & biases to wandb
* save the kaggle notebook result into github
"""
import argparse
import uuid

import wandb
from transformers import get_linear_schedule_with_warmup

""" import dependencies """
from datetime import datetime
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
from transformers.modeling_outputs import SequenceClassifierOutput
from datetime import datetime
import uuid

try:
    from Extensions import *
except ImportError as ie:
    print(ie)

class HyenaDNAPagingMQTLDataset(PagingMQTLDataset):
    def preprocess(self, row: dict):
        sequence = row["sequence"]
        label = row["label"]

        tokenizedSequence = self.dnaSeqTokenizer(sequence)
        input_ids = tokenizedSequence["input_ids"]

        input_ids_tensor: torch.Tensor = torch.tensor(input_ids).long() # need to convert to long
        label_tensor: torch.Tensor = torch.tensor(label)

        encoded_map: dict = {
            "input_ids": input_ids_tensor,
            "labels": label_tensor,
        }
        return encoded_map


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

        num_training_steps = self.trainer.estimated_stepping_batches
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )

        scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",  # step-wise decay
            "frequency": 1,
            "name": "learning_rate",  # shows up in wandb as `learning_rate`
        }

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config
        }

    def configure_gradient_clipping(
            self,
            optimizer: Optimizer,
            gradient_clip_val: Optional[Union[int, float]] = None,
            gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)


def createSingleHyenaDnaPagingDatasets(
        split,
        tokenizer,
        window,
) -> HyenaDNAPagingMQTLDataset:  # I can't come up with creative names

    dataset_map = load_dataset("fahimfarhan/mqtl-classification-datasets", streaming=True)
    dataset_len = get_dataset_length(dataset_name="fahimfarhan/mqtl-classification-datasets", split=split)

    someDataset = dataset_map[split]
    print(f"{split = } ==> {dataset_len = }")
    return HyenaDNAPagingMQTLDataset(
        someDataset=someDataset,
        dnaSeqTokenizer=tokenizer,
        seqLength=window,
        toKmer=False,
        datasetLen = dataset_len
    )

def createHyenaDnaPagingTrainValTestDatasets(
        tokenizer: PreTrainedTokenizer,
        window: int,
) -> (HyenaDNAPagingMQTLDataset, HyenaDNAPagingMQTLDataset, HyenaDNAPagingMQTLDataset):


    # not sure if this is a good idea. if anything goes wrong, revert back to previous code of this function
    train_dataset = createSingleHyenaDnaPagingDatasets(
        split = f"train_binned_{window}",
        tokenizer=tokenizer,
        window=window,
    )

    val_dataset = createSingleHyenaDnaPagingDatasets(
        split = f"validate_binned_{window}",
        tokenizer=tokenizer,
        window=window,
    )

    test_dataset = createSingleHyenaDnaPagingDatasets(
        split = f"test_binned_{window}",
        tokenizer = tokenizer,
        window = window,
    )
    return train_dataset, val_dataset, test_dataset

def save_fine_tuned_model(
    mainModel,
    repository,
    commit_message,
):
    # save the model in huggingface repository, and local storage
    mainModel.save_pretrained(save_directory=repository, safe_serialization=False)
    # push to the hub
    is_my_laptop = isMyLaptop()

    if is_my_laptop:  # no need to save
        return

    mainModel.push_to_hub(
        repo_id=repository,
        # subfolder=f"my-awesome-model-{WINDOW}", subfolder didn't work :/
        commit_message=commit_message,
        safe_serialization=False
    )
    pass



def get_run_name_suffix():
    date = datetime.now().strftime("%y%b%d")  # e.g. 25Jun20
    rand = str(uuid.uuid4())[:4]              # e.g. a9f1
    return f"{date}-{rand}"


def parse_args():
    # ------------------------
    # Default Config Values
    # ------------------------
    DEFAULT_MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
    DEFAULT_RUN_NAME_PREFIX = "hyena-dna-mqtl-classifier"
    DEFAULT_WINDOW = 1024
    DEFAULT_NUM_EPOCHS = 10
    DEFAULT_PER_DEVICE_BATCH_SIZE = None  # dynamic
    DEFAULT_NUM_GPUS = None  # dynamic
    DEFAULT_ENABLE_LOGGING = False
    DEFAULT_RUN_NAME_SUFFIX = None
    DEFAULT_SAVE_MODEL_IN_LOCAL_DIRECTORY = None
    DEFAULT_SAVE_MODEL_IN_REMOTE_REPOSITORY = None

    # ------------------------
    # Argument Parsing
    # ------------------------
    parser = argparse.ArgumentParser(description="Train or fine-tune HyenaDNA or DNA_BERT")

    parser.add_argument("--MODEL_NAME", type=str, default=DEFAULT_MODEL_NAME,
                        help="Pretrained model name or path")
    parser.add_argument("--run_name_prefix", type=str, default=DEFAULT_RUN_NAME_PREFIX,
                        help="Prefix for naming this run")
    parser.add_argument("--WINDOW", type=int, default=DEFAULT_WINDOW,
                        help="Sliding window size for input sequences")
    parser.add_argument("--NUM_EPOCHS", type=int, default=DEFAULT_NUM_EPOCHS,
                        help="Total number of training epochs")
    parser.add_argument("--PER_DEVICE_BATCH_SIZE", type=int, default=DEFAULT_PER_DEVICE_BATCH_SIZE,
                        help="Batch size per device. If not set, it will be determined dynamically.")
    parser.add_argument("--NUM_GPUS", type=int, default=DEFAULT_NUM_GPUS,
                        help="Number of GPUs to use. If not set, auto-detected via torch.")
    parser.add_argument("--ENABLE_LOGGING", action="store_true", default=DEFAULT_ENABLE_LOGGING,
                        help="Enable logging with tools like W&B")

    # Optional: override naming
    parser.add_argument("--run_name_suffix", type=str, default=DEFAULT_RUN_NAME_SUFFIX,
                        help="Override automatic run name suffix")
    parser.add_argument("--SAVE_MODEL_IN_LOCAL_DIRECTORY", type=str, default=DEFAULT_SAVE_MODEL_IN_LOCAL_DIRECTORY,
                        help="Custom local directory to save model")
    parser.add_argument("--SAVE_MODEL_IN_REMOTE_REPOSITORY", type=str, default=DEFAULT_SAVE_MODEL_IN_REMOTE_REPOSITORY,
                        help="Custom HuggingFace repo name to push model")

    return parser.parse_args()

def start():
    args = parse_args()

    run_name_suffix = args.run_name_suffix or get_run_name_suffix()
    convert_to_kmer = (args.MODEL_NAME == "zhihan1996/DNA_bert_6")

    run_name = f"{args.run_name_prefix}-{args.WINDOW}" # "-{run_name_suffix}"
    save_model_in_local_directory = args.SAVE_MODEL_IN_LOCAL_DIRECTORY or f"fine-tuned-{run_name}"
    save_model_in_remote_repository = args.SAVE_MODEL_IN_REMOTE_REPOSITORY or f"fahimfarhan/{run_name}"

    per_device_batch_size = args.PER_DEVICE_BATCH_SIZE or getDynamicBatchSize()
    num_gpus = args.NUM_GPUS or max(torch.cuda.device_count(), 1)

    commit_msg_and_wandb_run_name = run_name + run_name_suffix

    print("=" * 60)
    print(f"RUN_NAME: {run_name}")
    print(f"MODEL_NAME: {args.MODEL_NAME}")
    print(f"WINDOW: {args.WINDOW}")
    print(f"NUM_EPOCHS: {args.NUM_EPOCHS}")
    print(f"PER_DEVICE_BATCH_SIZE: {per_device_batch_size}")
    print(f"NUM_GPUS: {num_gpus}")
    print(f"ENABLE_LOGGING: {args.ENABLE_LOGGING}")
    print(f"CONVERT_TO_KMER: {convert_to_kmer}")
    print(f"SAVE_MODEL_IN_LOCAL_DIRECTORY: {save_model_in_local_directory}")
    print(f"SAVE_MODEL_IN_REMOTE_REPOSITORY: {save_model_in_remote_repository}")
    print(f"COMMIT_MESSAGE: {commit_msg_and_wandb_run_name}")
    print("=" * 60)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # to prevent out of memory error
    disableAnnoyingWarnings()

    if args.ENABLE_LOGGING:
        signInToHuggingFaceAndWandbToUploadModelWeightsAndBiases()
    else:
        wandb.init("offline")

    model_name = args.MODEL_NAME

    dnaTokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    mainModel = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    train_dataset, val_dataset, test_dataset = createHyenaDnaPagingTrainValTestDatasets(
        tokenizer=dnaTokenizer,
        window=args.WINDOW,
    )

    print(mainModel)

    dataCollator = DataCollatorWithPadding(tokenizer=dnaTokenizer)
    train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=False, collate_fn=dataCollator) # Can't shuffle the paging/streaming datasets
    val_loader = DataLoader(val_dataset, batch_size=per_device_batch_size, shuffle=False, collate_fn=dataCollator)
    test_loader = DataLoader(test_dataset, batch_size=per_device_batch_size, shuffle=False, collate_fn=dataCollator)

    earlyStoppingCallback = EarlyStopping(
        monitor='eval_loss',
        patience=3,
        mode='min',
        verbose=True
    )
    print("create trainer")

    trainer = pl.Trainer(
        max_epochs=args.NUM_EPOCHS,  # instead of max_steps
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
        logger=[
            pl.loggers.TensorBoardLogger(save_dir="tensorboard", name="logs"),
            pl.loggers.WandbLogger(name=run_name, project="mqtl-classification"),
        ],
        strategy="auto",
    )

    plModule = HyenaDNAMQTLClassifierModule(mainModel)
    try:
        trainer.fit(plModule, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as x:
        timber.error(f"Error during training/evaluating: {x}")
    finally:
        if args.ENABLE_LOGGING:
            try:
                save_fine_tuned_model(
                    mainModel=mainModel,
                    repository=save_model_in_remote_repository,
                    commit_message=commit_msg_and_wandb_run_name,

                )
            except Exception as x:
                timber.error(f"Error during fine-tuning: {x}")
            pass

    try:
        trainer.test(plModule, dataloaders=test_loader)
    except Exception as e:
        timber.error(f"Error during testing: {e}")
    pass


def main():
    start_time = datetime.now()

    start()

    end_time = datetime.now()
    execution_time = end_time - start_time
    total_seconds = execution_time.total_seconds()

    # Convert total seconds into hours, minutes, and seconds
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"Execution time: {int(hours)}h {int(minutes)}m {seconds:.2f}s")
    pass

if __name__ == '__main__':
    main()
    pass
