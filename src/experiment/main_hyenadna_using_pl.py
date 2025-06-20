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
from pytorch_lightning.callbacks import EarlyStopping
from transformers.modeling_outputs import SequenceClassifierOutput

""" import dependencies """
from datetime import datetime
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader

try:
    from src.experiment.Extensions import *
except ImportError as ie:
    print(ie)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # hyena dna requires this
print("import dependencies completed")

""" dynamic section. may be some consts,  changes based on model, etc. Try to keep it as small as possible """
""" THIS IS THE MOST IMPORTANT PART """

MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
run_name_prefix = "hyena-dna-mqtl-classifier"

run_name_suffix = datetime.now().strftime("%Y-%m-%d-%H-%M")
# run_platform="laptop"

CONVERT_TO_KMER = (MODEL_NAME == "zhihan1996/DNA_bert_6")
WINDOW = 1024  # use small window on your laptop gpu (eg nvidia rtx 2k), and large window on datacenter gpu (T4, P100, etc)
RUN_NAME = f"{run_name_prefix}-{WINDOW}-{run_name_suffix}"
SAVE_MODEL_IN_LOCAL_DIRECTORY= f"fine-tuned-{RUN_NAME}"
SAVE_MODEL_IN_REMOTE_REPOSITORY = f"fahimfarhan/{RUN_NAME}"

NUM_EPOCHS = 50
PER_DEVICE_BATCH_SIZE = getDynamicBatchSize()
NUM_GPUS = max(torch.cuda.device_count(), 1)  # fallback to 1 if no GPU
ENABLE_LOGGING = True

# use it for step based implementation (huggingface trainer library)
# NUM_ROWS = 2_000    # hardcoded value
# EPOCHS = 1
# effective_batch_size = PER_DEVICE_BATCH_SIZE * NUM_GPUS
# STEPS_PER_EPOCH = NUM_ROWS // effective_batch_size
# MAX_STEPS = EPOCHS * STEPS_PER_EPOCH

print("init arguments completed")

""" Common codes """
class HyenaDnaPagingMQTLDataset(PagingMQTLDataset):
    def preprocess(self, row: dict):
        sequence = row["sequence"]
        label = row["label"]

        seqTokenized = self.dnaSeqTokenizer(  # for hyena dna not a bert tokenizer. eg, no attention etc. misleading name
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


""" main """
class HyenaDnaMQTLClassifierModule(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-5, weight_decay=0.0, max_grad_norm=1.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        # self.criterion = torch.nn.CrossEntropyLoss()

        self.train_metrics = ComputeMetricsUsingSkLearn()
        self.val_metrics = ComputeMetricsUsingSkLearn()
        self.test_metrics = ComputeMetricsUsingSkLearn()

    def forward(self, batch):
        seqClassifierOutput: SequenceClassifierOutput = self.model(**batch)
        return seqClassifierOutput.loss, seqClassifierOutput.logits

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

        for k, v in metrics.items():
            self.log(f"train_{k}", v, prog_bar=True, on_epoch=True, logger=True)

        pretty_print_metrics(metrics, f"epoch {self.current_epoch}: Train")
        pass

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        labels = batch["labels"]
        loss, logits = self.forward(batch)

        # Log the loss
        self.log("eval_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        self.val_metrics.update(logits=logits, labels=labels)
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.val_metrics.clear()

        for k, v in metrics.items():
            self.log(f"eval_{k}", v, prog_bar=True, on_epoch=True, logger=True)

        pretty_print_metrics(metrics, f"epoch {self.current_epoch}: Eval")
        pass

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        labels = batch["labels"]
        loss, logits = self.forward(batch)

        # Log the loss
        self.log("test_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        self.test_metrics.update(logits=logits, labels=labels)
        return loss

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.test_metrics.clear()

        for k, v in metrics.items():
            self.log(f"test_{k}", v, prog_bar=True, on_epoch=True, logger=True)

        pretty_print_metrics(metrics, f"epoch {self.current_epoch}: Test")
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

def createSingleHyenaDnaPagingDatasets(
        data_files,
        split,
        tokenizer,
        window,
        splitSequenceRequired
) -> HyenaDnaPagingMQTLDataset:  # I can't come up with creative names
    is_my_laptop = isMyLaptop()
    if is_my_laptop:
        dataset_map = load_dataset("csv", data_files=data_files, streaming=True)
        dataset_len = get_dataset_length(local_path=data_files[split], split=split)
    else:
        dataset_map = load_dataset("fahimfarhan/mqtl-classification-datasets", streaming=True)
        dataset_len = get_dataset_length(dataset_name="fahimfarhan/mqtl-classification-datasets", split=split)

    someDataset = dataset_map[split]
    print(f"{split = } ==> {dataset_len = }")
    return HyenaDnaPagingMQTLDataset(
        someDataset=someDataset,
        dnaSeqTokenizer=tokenizer,
        seqLength=window,
        toKmer=splitSequenceRequired,
        datasetLen = dataset_len
    )

def createHyenaDnaPagingTrainValTestDatasets(tokenizer, window, toKmer) -> (HyenaDnaPagingMQTLDataset, HyenaDnaPagingMQTLDataset, HyenaDnaPagingMQTLDataset):
    prefix = "/home/gamegame/PycharmProjects/mqtl-classification/src/datageneration/"

    data_files = {
        # small
        "train_binned_1027": f"{prefix}_1027_train_binned.csv",
        "validate_binned_1027": f"{prefix}_1027_validate_binned.csv",
        "test_binned_1027": f"{prefix}_1027_train_binned.csv",

        # medium
        "train_binned_2051": f"{prefix}_2051_train_binned.csv",
        "validate_binned_2051": f"{prefix}_2051_validate_binned.csv",
        "test_binned_2051": f"{prefix}_2051_test_binned.csv",

        # large
        "train_binned_4099": f"{prefix}_4099_train_binned.csv",
        "validate_binned_4099": f"{prefix}_4099_validate_binned.csv",
        "test_binned_4099": f"{prefix}_4099_test_binned.csv",
    }

    # not sure if this is a good idea. if anything goes wrong, revert back to previous code of this function
    train_dataset = createSingleHyenaDnaPagingDatasets(data_files, f"train_binned_{window}", tokenizer, window, toKmer)

    val_dataset =createSingleHyenaDnaPagingDatasets(data_files, f"validate_binned_{window}", tokenizer, window, toKmer)

    test_dataset = createSingleHyenaDnaPagingDatasets(data_files, f"test_binned_{window}", tokenizer, window, toKmer)

    return train_dataset, val_dataset, test_dataset


def save_fine_tuned_model(mainModel):
    # save the model in huggingface repository, and local storage
    mainModel.save_pretrained(save_directory=SAVE_MODEL_IN_LOCAL_DIRECTORY, safe_serialization=False)
    # push to the hub
    is_my_laptop = isMyLaptop()

    if is_my_laptop:  # no need to save
        return

    # commit_message = f":tada: Push {RUN_NAME} model for window size {WINDOW} from my laptop"
    commit_message = f":tada: Push {RUN_NAME} model for window size {WINDOW} into huggingface hub"
    mainModel.push_to_hub(
        repo_id=SAVE_MODEL_IN_REMOTE_REPOSITORY,
        # subfolder=f"my-awesome-model-{WINDOW}", subfolder didn't work :/
        commit_message=commit_message,
        safe_serialization=False
    )

def start():
    timber.info(green)
    timber.info("---Inside start function---")
    timber.info(f"{PER_DEVICE_BATCH_SIZE = }")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    disableAnnoyingWarnings()

    if isMyLaptop() or not ENABLE_LOGGING:
        wandb.init(mode="offline")  # Logs only locally
    else:
        # datacenter eg huggingface or kaggle.
        signInToHuggingFaceAndWandbToUploadModelWeightsAndBiases()


    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # OutOfMemoryError

    model_name = MODEL_NAME

    dnaTokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    mainModel = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                   trust_remote_code=True)  # this is the correct way to load pretrained weights, and modify config

    print(mainModel)

    dataCollator = DataCollatorWithPadding(tokenizer=dnaTokenizer)

    # L = T + k - 3 [for dna bert 6, we have 2 extra tokens, cls, and sep]
    rawSequenceLength = WINDOW + 6 - 3
    train_dataset, val_dataset, test_dataset = createHyenaDnaPagingTrainValTestDatasets(tokenizer=dnaTokenizer, window=rawSequenceLength, toKmer=CONVERT_TO_KMER)

    train_loader = DataLoader(train_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, collate_fn=dataCollator) # Can't shuffle the paging/streaming datasets
    val_loader = DataLoader(val_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, collate_fn=dataCollator)
    test_loader = DataLoader(test_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, collate_fn=dataCollator)

    earlyStoppingCallback = EarlyStopping(
        monitor='eval_loss',
        patience=3,
        mode='min',
        verbose=True
    )
    print("create trainer")

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,  # instead of max_steps
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
            pl.loggers.WandbLogger(name=RUN_NAME, project="mqtl-classification"),
        ],
        strategy="auto",
    )

    plModule = HyenaDnaMQTLClassifierModule(mainModel)

    try:
        trainer.fit(plModule, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as x:
        timber.error(f"Error during training/evaluating: {x}")
    finally:
        if ENABLE_LOGGING:
            try:
                save_fine_tuned_model(mainModel=mainModel)
            except Exception as x:
                timber.error(f"Error during fine-tuning: {x}")
            pass

    try:
        trainer.test(plModule, dataloaders=test_loader)
    except Exception as e:
        timber.error(f"Error during testing: {e}")

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
    # for some reason, the variables in the main function act like global variables in python
    # hence other functions get confused with the "global" variables. easiest solution, write everything
    # in another function (say, start(), or main()), and call it inside the main
    main()
    pass
