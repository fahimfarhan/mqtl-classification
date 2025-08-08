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

""" import dependencies """
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers.modeling_outputs import SequenceClassifierOutput
from Extensions import *
from all_models import getModel, MQTLStreamingDataset
# try:
#     from Extensions import *
# except ImportError as ie:
#     print(ie)

class MQTLClassifierModule(pl.LightningModule):
    def __init__(
        self,
         model,
         learning_rate:float,
         weight_decay:float,
         optimizer_name: str,
         l1_lambda:float=1.0,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.l1_lambda = l1_lambda
        self.optimizer_name = optimizer_name
        # self.criterion = torch.nn.CrossEntropyLoss()

        train_metrics = ComputeMetricsUsingSkLearn()
        val_metrics = ComputeMetricsUsingSkLearn()
        test_metrics = ComputeMetricsUsingSkLearn()

        self.metricsMap = {
            "train": train_metrics,
            "eval": val_metrics,
            "test": test_metrics,
        }

    def configure_optimizers(self):
        optimizer = get_optimizer(name=self.optimizer_name, parameters=self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer


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

            loss = self.common_step(batch, batch_idx, stage="train")
            if self.l1_lambda > 0:
                raw_l1 = sum(p.abs().sum() for p in self.parameters())
                l1_penalty = self.l1_lambda * raw_l1
                loss = loss + l1_penalty
                self.log("l1_penalty", l1_penalty, prog_bar=True, on_step=True)

            return loss

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step(batch, batch_idx, stage="eval")

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        return self.common_step(batch, batch_idx, stage="test")

    def on_common_epoch_end(self, stage: str, metrics_collector):
        metrics = metrics_collector.compute()
        metrics_collector.clear()

        # Log only scalar metrics
        for k, v in metrics.items():
            if isinstance(v, (int, float, torch.Tensor)):  # Scalars are safe / ignore the confusion matrix
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

    # Manual gradient clipping
    # def configure_gradient_clipping(
    #         self,
    #         optimizer: Optimizer,
    #         gradient_clip_val: Optional[Union[int, float]] = None,
    #         gradient_clip_algorithm: Optional[str] = None,
    # ) -> None:
    #     if self.max_grad_norm is not None:
    #         torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)


def createSingleStreamingDatasets(
        inputArgs: Namespace,
        split,
        tokenizer,
        window,
        dataset_map: DatasetDict,
) -> MQTLStreamingDataset:  # I can't come up with creative names

    # dataset_len = get_dataset_length(dataset_name="fahimfarhan/mqtl-classification-datasets", split=split)

    someDataset = dataset_map[split]
    # print(f"{split = } ==> {dataset_len = }")
    return MQTLStreamingDataset(
        inputArgs=inputArgs,
        someDataset=someDataset,
        dnaSeqTokenizer=tokenizer,
        seqLength=window,
        datasetLen = 0 # dataset_len
    )

def createStreamingTrainValTestDatasets(
        inputArgs: Namespace,
        tokenizer: PreTrainedTokenizer,
        window: int,
) -> (MQTLStreamingDataset, MQTLStreamingDataset, MQTLStreamingDataset):

    dataset_map: DatasetDict = load_dataset(f"fahimfarhan/mqtl-classification-datasets-{inputArgs.GENOME}", streaming=True)

    # not sure if this is a good idea. if anything goes wrong, revert back to previous code of this function
    train_dataset = createSingleStreamingDatasets(
        inputArgs=inputArgs,
        split = f"train_binned_{window}",
        tokenizer=tokenizer,
        window=window,
        dataset_map=dataset_map,
    )

    val_dataset = createSingleStreamingDatasets(
        inputArgs=inputArgs,
        split = f"validate_binned_{window}",
        tokenizer=tokenizer,
        window=window,
        dataset_map=dataset_map,
    )

    test_dataset = createSingleStreamingDatasets(
        inputArgs=inputArgs,
        split = f"test_binned_{window}",
        tokenizer = tokenizer,
        window = window,
        dataset_map=dataset_map,
    )
    return train_dataset, val_dataset, test_dataset


def start():
    args = parse_args()

    run_name_suffix = args.RUN_NAME_SUFFIX or get_run_name_suffix()
    convert_to_kmer = (args.MODEL_NAME == "zhihan1996/DNA_bert_6")

    run_name = f"{args.RUN_NAME_PREFIX}-{args.WINDOW}" # "-{run_name_suffix}"
    save_model_in_local_directory = args.SAVE_MODEL_IN_LOCAL_DIRECTORY or f"fine-tuned-{run_name}"
    save_model_in_remote_repository = args.SAVE_MODEL_IN_REMOTE_REPOSITORY or f"fahimfarhan/{run_name}"

    per_device_batch_size = args.PER_DEVICE_BATCH_SIZE or getDynamicBatchSize()
    num_gpus = args.NUM_GPUS or max(torch.cuda.device_count(), 1)

    commit_msg_and_wandb_run_name = run_name + run_name_suffix

    print("=" * 60)
    print(f"RUN_NAME: {run_name}")
    print(f"MODEL_NAME: {args.MODEL_NAME}")
    print(f"MODEL_NAME: {args.MODEL_VARIANT}")
    print(f"WINDOW: {args.WINDOW}")
    print(f"NUM_EPOCHS: {args.NUM_EPOCHS}")
    print(f"PER_DEVICE_BATCH_SIZE: {per_device_batch_size}")
    print(f"NUM_GPUS: {num_gpus}")
    print(f"ENABLE_LOGGING: {args.ENABLE_LOGGING}")
    print(f"CONVERT_TO_KMER: {convert_to_kmer}")
    print(f"SAVE_MODEL_IN_LOCAL_DIRECTORY: {save_model_in_local_directory}")
    print(f"SAVE_MODEL_IN_REMOTE_REPOSITORY: {save_model_in_remote_repository}")
    print(f"COMMIT_MESSAGE: {commit_msg_and_wandb_run_name}")
    print(f"LEARNING_RATE: {args.LEARNING_RATE}")
    print(f"L1_LAMBDA_WEIGHT: {args.L1_LAMBDA_WEIGHT}")
    print(f"WEIGHT_DECAY: {args.WEIGHT_DECAY}")
    print(f"GRADIENT_CLIP: {args.GRADIENT_CLIP}")
    print(f"DROP_OUT_PROBABILITY: {args.DROP_OUT_PROBABILITY}")
    print(f"CRITERION_LABEL_SMOOTHENING: {args.CRITERION_LABEL_SMOOTHENING}")
    print(f"OPTIMIZER: {args.OPTIMIZER}")
    print(f"EARLY_STOPPING: {args.EARLY_STOPPING}")
    print(f"KIPOI_FINE_TUNE: {args.KIPOI_FINE_TUNE}")
    print(f"GENOME: {args.GENOME}")
    print("=" * 60)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # to prevent out of memory error
    disableAnnoyingWarnings()

    if args.ENABLE_LOGGING:
        signInToHuggingFaceAndWandbToUploadModelWeightsAndBiases()
    else:
        wandb.init(mode="offline")  # Logs only locally

    model_name = args.MODEL_NAME

    dnaTokenizer = None

    if model_name in ["zhihan1996/DNA_bert_6", "LongSafari/hyenadna-small-32k-seqlen-hf"]:
        dnaTokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
        )
    mainModel = getModel(args = args, dnaTokenizer=dnaTokenizer)


    rawSequenceLength = args.WINDOW
    if model_name == "zhihan1996/DNA_bert_6":
        rawSequenceLength = args.WINDOW + 6 - 3


    train_dataset, val_dataset, test_dataset = createStreamingTrainValTestDatasets(
        inputArgs=args,
        tokenizer=dnaTokenizer,
        window=rawSequenceLength,
    )

    print(mainModel)

    dataCollator = DataCollatorWithPadding(tokenizer=dnaTokenizer)
    train_loader = DataLoader(train_dataset, batch_size=per_device_batch_size, shuffle=False, collate_fn=dataCollator) # Can't shuffle the paging/streaming datasets
    val_loader = DataLoader(val_dataset, batch_size=per_device_batch_size, shuffle=False, collate_fn=dataCollator)
    test_loader = DataLoader(test_dataset, batch_size=per_device_batch_size, shuffle=False, collate_fn=dataCollator)

    plCallBacks = [
            pl.callbacks.ModelCheckpoint(
                dirpath="output_checkpoints",
                save_top_k=1,  # todo: -1 means no checkpoint is saved. 1 means top 1 checkpoint is saved.
                every_n_train_steps=None,
                save_weights_only=False,
                save_on_train_epoch_end=True,  # save at end of epoch
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),

        ]
    if args.EARLY_STOPPING:
        earlyStoppingCallback = EarlyStopping(
            monitor='eval_loss',
            patience=3,
            mode='min',
            verbose=True
        )
        plCallBacks.append(earlyStoppingCallback)
    print("create trainer")

    trainer = pl.Trainer(
        max_epochs=args.NUM_EPOCHS,  # instead of max_steps
        limit_train_batches=None,  # 100% of data each epoch
        val_check_interval=1.0,  # validate at end of each epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=args.GRADIENT_CLIP,
        accumulate_grad_batches=1,
        precision=32,
        default_root_dir="output_checkpoints",
        enable_checkpointing=True,
        callbacks=plCallBacks,
        logger=[
            pl.loggers.TensorBoardLogger(save_dir=f"tensorboard-{run_name}", name="logs"),
            pl.loggers.WandbLogger(name=run_name, project="mqtl-classification"),
        ],
        strategy="auto",
    )

    plModule = MQTLClassifierModule(
        model=mainModel,
        learning_rate=args.LEARNING_RATE,
        weight_decay=args.WEIGHT_DECAY,
        optimizer_name=args.OPTIMIZER,
        l1_lambda=args.L1_LAMBDA_WEIGHT,
    )
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
