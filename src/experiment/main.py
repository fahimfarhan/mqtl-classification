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
from datetime import datetime

import logging
import os

import evaluate
import huggingface_hub
from huggingface_hub import HfApi
import numpy as np
import wandb
from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from torch.utils.data import IterableDataset, get_worker_info
from transformers import BertTokenizer, BatchEncoding, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding
import torch
import warnings

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # hyena dna requires this
print("import dependencies completed")

""" Common codes """
# some colors for visual convenience
red = "\u001b[31m"
green = "\u001b[32m"
yellow = "\u001b[33m"
blue = "\u001b[34m"

timber = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)  # change to level=logging.DEBUG to print more logs...


def getDynamicGpuDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")  # For NVIDIA GPUs
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For Apple Silicon Macs
    else:
        return torch.device("cpu")   # Fallback to CPU

def getDynamicBatchSize():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        vramGiB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB

        if "a100" in gpu_name:   # A100 (40GB+ VRAM)
            batch_size = 128
        elif "v100" in gpu_name:  # V100 (16GB/32GB VRAM)
            batch_size = 64 if vramGiB >= 32 else 32
        elif "p100" in gpu_name:  # P100 (16GB VRAM)
            batch_size = 32
        elif "t4" in gpu_name:    # Tesla T4 (16GB VRAM, common in Colab/Kaggle)
            batch_size = 32  # Maybe try 64 if no OOM
        elif "rtx 3090" in gpu_name or vramGiB >= 24:  # RTX 3090 (24GB VRAM)
            batch_size = 64
        elif vramGiB >= 16:   # Any other 16GB+ VRAM GPUs
            batch_size = 32
        elif vramGiB >= 8:    # 8GB VRAM GPUs (e.g., RTX 2080, 3060, etc.)
            batch_size = 16
        elif vramGiB >= 6:    # 6GB VRAM GPUs (e.g., RTX 2060)
            batch_size = 8
        else:
            batch_size = 4  # Safe fallback for smaller GPUs
    else:
        batch_size = 4  # CPU mode, keep it small

    return batch_size

def getGpuName():
    gpu_name = torch.cuda.get_device_name(0).lower()
    return gpu_name

# for hyenaDna. its tokenizer can process longer sequences...
def sequenceEncodePlusForHyenaDna(
    tokenizer: BertTokenizer,
    sequence: str,
    label: int
) -> BatchEncoding:
    input_ids = tokenizer(sequence)["input_ids"]
    input_ids: torch.Tensor = torch.Tensor(input_ids)
    label_tensor = torch.tensor(label)
    encoded_map: dict = {
        "input_ids": input_ids.long(),
        # "attention_mask": attention_mask.int(),    # hyenaDNA does not have attention layer
        "labels": label_tensor
    }

    batchEncodingDict: BatchEncoding = BatchEncoding(encoded_map)
    return batchEncodingDict

# for dnaBert. it cannot process longer sequences...
def sequenceEncodePlusWithSplitting(
        tokenizer: BertTokenizer,
        sequence: str,
        label: int
) -> BatchEncoding:
    max_size = 512

    tempMap: BatchEncoding = tokenizer.encode_plus(
        sequence,
        add_special_tokens=False,  # we'll add the special tokens manually in the for loop below
        return_attention_mask=True,
        return_tensors="pt"
    )

    someInputIds1xN = tempMap["input_ids"]  # shape = 1xN , N = sequence length
    someMasks1xN = tempMap["attention_mask"]
    inputIdsList = list(someInputIds1xN[0].split(510))
    masksList = list(someMasks1xN[0].split(510))

    tmpLength: int = len(inputIdsList)

    for i in range(0, tmpLength):
        cls: torch.Tensor = torch.Tensor([101])
        sep: torch.Tensor = torch.Tensor([102])

        isTokenUnitTensor = torch.Tensor([1])

        inputIdsList[i]: torch.Tensor = torch.cat([
            cls,
            inputIdsList[i],
            sep
        ])

        masksList[i] = torch.cat([
            isTokenUnitTensor,
            masksList[i],
            isTokenUnitTensor
        ])


        pad_len: int = max_size - inputIdsList[i].shape[0]
        if pad_len > 0:
            pad: torch.Tensor = torch.Tensor([0] * pad_len)

            inputIdsList[i]: torch.Tensor = torch.cat([
                inputIdsList[i],
                pad
            ])

            masksList[i]: torch.Tensor = torch.cat([
                masksList[i],
                pad
            ])


    # so each item len = 512, and the last one may have some padding
    input_ids: torch.Tensor = torch.stack(inputIdsList).squeeze()  # what's with this squeeze / unsqueeze thing? o.O
    attention_mask: torch.Tensor = torch.stack(masksList)
    label_tensor = torch.tensor(label)

    # print(f"{input_ids.shape = }")

    encoded_map: dict = {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.int(),
        "labels": label_tensor
    }

    batchEncodingDict: BatchEncoding = BatchEncoding(encoded_map)
    return batchEncodingDict

def sequenceEncodePlusCompact(
        splitSequence: bool,
        tokenizer: BertTokenizer,
        sequence: str,
        label: int
) -> BatchEncoding:
    if splitSequence:
        return sequenceEncodePlusWithSplitting(tokenizer, sequence, label)
    else:
        return sequenceEncodePlusForHyenaDna(tokenizer, sequence, label)


class PagingMQTLDataset(IterableDataset):
    def __init__(
            self,
            someDataset: Dataset,
            bertTokenizer: BertTokenizer,
            seqLength: int,
            splitSequenceRequired: bool,
            datasetLen: int
        ):
        self.someDataset = someDataset
        self.bertTokenizer = bertTokenizer
        self.seqLength = seqLength
        self.splitSequenceRequired = splitSequenceRequired
        self.datasetLen=datasetLen
        pass

    def __len__(self):
        return self.datasetLen

    def createShardDatasetForMultipleGpus(self):
        worker_info = get_worker_info()

        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

        # Total shards = world_size * num_workers (or 1 if no workers)
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        timber.info(f"{num_workers = }, {worker_id = }, {rank = }, {world_size=}")

        total_shards = world_size * num_workers
        shard_index = rank * num_workers + worker_id
        # Shard the dataset accordingly
        shard_dataset = self.someDataset.shard(num_shards=total_shards, index=shard_index)
        return shard_dataset

    def __iter__(self):
        shardDataset = self.createShardDatasetForMultipleGpus()

        for row in shardDataset:
            processed = self.preprocess(row)
            if processed is not None:
                yield processed

    def preprocess(self, row: dict):
        sequence = row["sequence"]
        label = row["label"]

        if len(sequence) != self.seqLength:
            return None  # skip a few problematic rows

        return sequenceEncodePlusCompact(self.splitSequenceRequired, self.bertTokenizer, sequence, label)

def isMyLaptop() -> bool:
    is_my_laptop = os.path.isfile("/home/gamegame/PycharmProjects/mqtl-classification/src/datageneration/dataset_4000_train_binned.csv")
    return is_my_laptop


def signInToHuggingFaceAndWandbToUploadModelWeightsAndBiases():
    # Load the .env file, but don't crash if it's not found (e.g., in Hugging Face Space)
    try:
        load_dotenv()  # Only useful on your laptop if .env exists
        print(".env file loaded successfully.")
    except Exception as e:
        print(f"Warning: Could not load .env file. Exception: {e}")

    # Try to get the token from environment variables
    try:
        token = os.getenv("HF_TOKEN")

        if not token:
            raise ValueError("HF_TOKEN not found. Make sure to set it in the environment variables or .env file.")

        # Log in to Hugging Face Hub
        huggingface_hub.login(token)
        print("Logged in to Hugging Face Hub successfully.")

    except Exception as e:
        print(f"Error during Hugging Face login: {e}")
        # Handle the error appropriately (e.g., exit or retry)

    # wand db login
    try:
        api_key = os.getenv("WAND_DB_API_KEY")
        timber.info(f"{api_key = }")

        if not api_key:
            raise ValueError(
                "WAND_DB_API_KEY not found. Make sure to set it in the environment variables or .env file.")

        # Log in to Hugging Face Hub
        wandb.login(key=api_key)
        print("Logged in to wand db successfully.")

    except Exception as e:
        print(f"Error during wand db Face login: {e}")
    pass

def get_dataset_length(dataset_name=None, split=None, local_path=None):
    try:
        if local_path:
            dataset = load_dataset("csv", data_files={split: local_path}, split=split, streaming=False)
        else:
            dataset = load_dataset(dataset_name, split=split, streaming=False)
        return len(dataset)
    except Exception as e:
        print(f"Error while loading length for {split}: {e}")
        return None

def createSinglePagingDatasets(
        data_files,
        split,
        tokenizer,
        window,
        splitSequenceRequired
) -> PagingMQTLDataset:  # I can't come up with creative names
    is_my_laptop = isMyLaptop()
    if is_my_laptop:
        dataset_map = load_dataset("csv", data_files=data_files, streaming=True)
        dataset_len = get_dataset_length(local_path=data_files[split], split=split)
    else:
        dataset_map = load_dataset("fahimfarhan/mqtl-classification-datasets", streaming=True)
        dataset_len = get_dataset_length(dataset_name="fahimfarhan/mqtl-classification-datasets", split=split)

    print(f"{split = } ==> {dataset_len = }")
    return PagingMQTLDataset(
        someDataset=dataset_map[f"train_binned_{window}"],
        bertTokenizer=tokenizer,
        seqLength=window,
        splitSequenceRequired=splitSequenceRequired,
        datasetLen = dataset_len
    )


def createPagingTrainValTestDatasets(tokenizer, window, splitSequenceRequired) -> (PagingMQTLDataset, PagingMQTLDataset, PagingMQTLDataset):
    prefix = "/home/gamegame/PycharmProjects/mqtl-classification/"
    data_files = {
        # small samples
        "train_binned_200": f"{prefix}src/datageneration/dataset_200_train_binned.csv",
        "validate_binned_200": f"{prefix}src/datageneration/dataset_200_validate_binned.csv",
        "test_binned_200": f"{prefix}src/datageneration/dataset_200_test_binned.csv",
        # medium samples
        "train_binned_1000": f"{prefix}src/datageneration/dataset_1000_train_binned.csv",
        "validate_binned_1000": f"{prefix}src/datageneration/dataset_1000_validate_binned.csv",
        "test_binned_1000": f"{prefix}src/datageneration/dataset_1000_test_binned.csv",
        # medium samples
        "train_binned_2000": f"{prefix}src/datageneration/dataset_2000_train_binned.csv",
        "validate_binned_2000": f"{prefix}src/datageneration/dataset_2000_validate_binned.csv",
        "test_binned_2000": f"{prefix}src/datageneration/dataset_2000_test_binned.csv",
        # large samples
        "train_binned_4000": f"{prefix}src/datageneration/dataset_4000_train_binned.csv",
        "validate_binned_4000": f"{prefix}src/datageneration/dataset_4000_validate_binned.csv",
        "test_binned_4000": f"{prefix}src/datageneration/dataset_4000_test_binned.csv",
    }

    # not sure if this is a good idea. if anything goes wrong, revert back to previous code of this function
    train_dataset = createSinglePagingDatasets(data_files, f"train_binned_{window}", tokenizer, window, splitSequenceRequired)

    val_dataset =createSinglePagingDatasets(data_files, f"validate_binned_{window}", tokenizer, window, splitSequenceRequired)

    test_dataset = createSinglePagingDatasets(data_files, f"test_binned_{window}", tokenizer, window, splitSequenceRequired)

    return train_dataset, val_dataset, test_dataset


# Load metrics
# global variables. bad practice...
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
roc_auc_metric = evaluate.load("roc_auc")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def computeMetricsUsingTorchEvaluate(args):
    logits, labels = args
    predictions = np.argmax(logits, axis=1)  # Get predicted class

    positive_logits = logits[:, 1]  # Get positive class logits

    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    roc_auc = roc_auc_metric.compute(prediction_scores=positive_logits, references=labels)["roc_auc"]  # using positive_logits repairs the error
    precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
    recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# use sklearn cz torchmetrics.classification gave array index out of bound exception :/ (whatever it is called in python)
def computeMetricsUsingSkLearn(args):
    #try:
    logits, labels = args
    # Get predicted class labels
    predictions = np.argmax(logits, axis=1)

    # Get predicted probabilities for the positive class
    positive_logits = logits[:, 1]  # Assuming binary classification and 2 output classes

    accuracy = accuracy_score(y_true=labels, y_pred=predictions)
    recall = recall_score(y_true=labels, y_pred=predictions)
    precision = precision_score(y_true=labels, y_pred=predictions)
    f1 = f1_score(y_true=labels, y_pred=predictions)
    roc_auc = roc_auc_score(y_true=labels, y_score=positive_logits)

    return {
      "accuracy": accuracy,
      "roc_auc": roc_auc,
      "precision": precision,
      "recall": recall,
      "f1": f1
    }
    #except Exception as x:
    #    timber.error(f"compute_metrics_using_sklearn failed with exception: {x}")
    #    return {"accuracy": 0, "roc_auc": 0, "precision": 0, "recall": 0, "f1": 0}

def disableAnnoyingWarnings():
    # Caution! if anything goes wrong, enable it. make sure this warning related issue ain't the culprit!
    warnings.filterwarnings(
        "ignore",
        message="Length of IterableDataset",
        category=UserWarning,
        module="torch.utils.data.dataloader"
    )


""" dynamic section. may be some consts,  changes based on model, etc. Try to keep it as small as possible """

MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
run_name_prefix = "hyena-dna-mqtl-classifier"
# MODEL_NAME =  "zhihan1996/DNA_bert_6"
# run_name_prefix = "dna-bert-6-mqtl-classifier"

run_name_suffix = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
run_platform="laptop"

RUN_NAME = f"{run_platform}-{run_name_prefix}-{run_name_suffix}"
SPLIT_SEQUENCE_REQUIRED= (MODEL_NAME != "LongSafari/hyenadna-small-32k-seqlen-hf")
WINDOW = 200  # use 200 on your local pc.

SAVE_MODEL_IN_LOCAL_DIRECTORY= f"fine-tuned-{RUN_NAME}-{WINDOW}"
SAVE_MODEL_IN_REMOTE_REPOSITORY = f"fahimfarhan/{RUN_NAME}-{WINDOW}"


NUM_ROWS = 1_000    # hardcoded value
PER_DEVICE_BATCH_SIZE = getDynamicBatchSize()
EPOCHS = 2
NUM_GPUS = max(torch.cuda.device_count(), 1)  # fallback to 1 if no GPU

effective_batch_size = PER_DEVICE_BATCH_SIZE * NUM_GPUS
STEPS_PER_EPOCH = NUM_ROWS // effective_batch_size
MAX_STEPS = EPOCHS * STEPS_PER_EPOCH

""" main """
def start():
    timber.info(green)
    timber.info("---Inside start function---")
    timber.info(f"{PER_DEVICE_BATCH_SIZE = }")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    disableAnnoyingWarnings()

    if isMyLaptop():
        wandb.init(mode="offline")  # Logs only locally
    else:
        # datacenter eg huggingface or kaggle.
        signInToHuggingFaceAndWandbToUploadModelWeightsAndBiases()

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model architecture:", config.architectures)

    mainTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mainModel = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True, num_labels=2)

    isGpuAvailable = torch.cuda.is_available()
    if isGpuAvailable:
        mainModel = mainModel.to("cuda")  # not sure if it is necessary in the kaggle / huggingface virtual-machine


    train_dataset, val_dataset, test_dataset = createPagingTrainValTestDatasets(tokenizer=mainTokenizer, window=WINDOW, splitSequenceRequired=SPLIT_SEQUENCE_REQUIRED)


    trainingArgs = TrainingArguments(
        run_name=RUN_NAME,
        output_dir="output_checkpoints",
        eval_strategy="steps",
        save_strategy="steps",
        logging_strategy="steps",
        eval_steps=STEPS_PER_EPOCH,
        save_steps=500,
        logging_steps=1,  # ← more frequent logs
        logging_first_step=True,  # ← log even the very first step
        log_level="info",  # ← control log verbosity
        log_level_replica="warning",   # ← useful if using multiple GPUs
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        max_steps=MAX_STEPS,
        weight_decay=0.01,
        learning_rate=1e-3,
        logging_dir="./logs",
        save_safetensors=False,
        gradient_checkpointing=True,  # to prevent out of memory error
        fp16=True,  # to train faster
        report_to="wandb"
    )

    dataCollator = DataCollatorWithPadding(tokenizer=mainTokenizer)


    print("create trainer")
    trainer = Trainer(
        model=mainModel,
        args=trainingArgs,
        train_dataset=train_dataset,  # train
        eval_dataset=val_dataset,  # validate
        data_collator=dataCollator,
        compute_metrics=computeMetricsUsingTorchEvaluate
    )

    try:
        # train, and validate
        result = trainer.train()
        print("-------Training completed. Results--------\n")
        print(result)
    except Exception as x:
        print(f"{x = }")
    finally:
        # in case sth goes wrong, upload the partially trained model so that I can work with something...
        mainModel.save_pretrained(save_directory=SAVE_MODEL_IN_LOCAL_DIRECTORY, safe_serialization=False)
        # push to the hub
        is_my_laptop = isMyLaptop()

        commit_message = f":tada: Push {RUN_NAME} model for window size {WINDOW} from huggingface space"
        if is_my_laptop:
            commit_message = f":tada: Push {RUN_NAME} model for window size {WINDOW} from my laptop"

        mainModel.push_to_hub(
            repo_id=SAVE_MODEL_IN_REMOTE_REPOSITORY,
            # subfolder=f"my-awesome-model-{WINDOW}", subfolder didn't work :/
            commit_message=commit_message,  # f":tada: Push model for window size {WINDOW}"
            safe_serialization=False
        )

    test_results = trainer.evaluate(eval_dataset=test_dataset)
    try:
        print("-------Test completed. Results--------\n")
        print(test_results)
    except Exception as x:
        print(f"{x = }")


    pass

if __name__ == "__main__":
    # for some reason, the variables in the main function act like global variables in python
    # hence other functions get confused with the "global" variables. easiest solution, write everything
    # in another function (say, start()), and call it inside the main
    start()
    pass
