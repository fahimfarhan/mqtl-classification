# Extensions.py
import argparse
import logging
import os
import uuid
from argparse import Namespace
from datetime import datetime

import evaluate
import huggingface_hub
from huggingface_hub import HfApi
import numpy as np
from torch.optim import AdamW, Adam
from lion_pytorch import Lion
from adan_pytorch import Adan
import wandb
from datasets import load_dataset, Dataset, DatasetDict
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score, confusion_matrix
from torch.utils.data import IterableDataset, get_worker_info
from transformers import BertTokenizer, BatchEncoding, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding, \
    PreTrainedTokenizer
import torch
import warnings
import torch

timber = logging.getLogger()
# logging.basicConfig(level=logging.DEBUG)
logging.basicConfig(level=logging.INFO)  # change to level=logging.DEBUG to print more logs...

# some colors for visual convenience
red = "\u001b[31m"
green = "\u001b[32m"
yellow = "\u001b[33m"
blue = "\u001b[34m"


def getDynamicGpuDevice():
    if torch.cuda.is_available():
        return torch.device("cuda")  # For NVIDIA GPUs
    elif torch.backends.mps.is_available():
        return torch.device("mps")  # For Apple Silicon Macs
    else:
        return torch.device("cpu")  # Fallback to CPU


def getDynamicBatchSize():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        vramGiB = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # Convert to GB

        if "a100" in gpu_name:  # A100 (40GB+ VRAM)
            batch_size = 128
        elif "v100" in gpu_name:  # V100 (16GB/32GB VRAM)
            batch_size = 64 if vramGiB >= 32 else 32
        elif "p100" in gpu_name:  # P100 (16GB VRAM)
            batch_size = 32
        elif "t4" in gpu_name:  # Tesla T4 (16GB VRAM, common in Colab/Kaggle)
            batch_size = 32  # Maybe try 64 if no OOM
        elif "rtx 3090" in gpu_name or vramGiB >= 24:  # RTX 3090 (24GB VRAM)
            batch_size = 64
        elif vramGiB >= 16:  # Any other 16GB+ VRAM GPUs
            batch_size = 32
        elif vramGiB >= 8:  # 8GB VRAM GPUs (e.g., RTX 2080, 3060, etc.)
            batch_size = 16
        elif vramGiB >= 6:  # 6GB VRAM GPUs (e.g., RTX 2060)
            batch_size = 8
        else:
            batch_size = 4  # Safe fallback for smaller GPUs
    else:
        batch_size = 4  # CPU mode, keep it small

    return batch_size


def getGpuName():
    gpu_name = torch.cuda.get_device_name(0).lower()
    return gpu_name


def toKmerSequence(seq: str, k: int = 6) -> str:
    """
    :param seq:  ATCGTTCAATCGTTCA.........
    :param k: 6
    :return: ATCGTT CAATCG TTCA.. ...... ......
    """

    output = ""
    for i in range(len(seq) - k + 1):
        output = output + seq[i:i + k] + " "
    return output


def pretty_print_metrics(metrics: dict, stage: str = ""):
    print(f"\n📊 {stage} Metrics:")
    for k, v in metrics.items():
        if k == "confusion_matrix":
            print(f"  {k:>15}:")
            # If v is a list of lists, print row by row
            if isinstance(v, (list, np.ndarray)):
                for row in v:
                    print(" " * 18, row)
            else:
                print(" " * 18, v)
        elif v is not None:
            print(f"  {k:>15}: {v:.4f}")
        else:
            print(f"  {k:>15}: N/A")


class PagingMQTLDataset(IterableDataset):
    def __init__(
            self,
            inputArgs: Namespace,
            someDataset: Dataset,
            dnaSeqTokenizer: PreTrainedTokenizer,
            seqLength: int,
            datasetLen: int
    ):
        self.inputArgs = inputArgs
        self.someDataset = someDataset
        self.dnaSeqTokenizer = dnaSeqTokenizer
        self.seqLength = seqLength
        self.datasetLen = datasetLen
        pass

    """
    # if you're using lightning ai, don't define the __len__. 
    # But in Huggingface transformer, setting the __len__ was kinda convenient

    def __len__(self):
        return self.datasetLen
    """

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
        raise Exception("Please override this function")


def isMyLaptop() -> bool:
    is_my_laptop = os.path.isdir("/home/gamegame/")
    return is_my_laptop


def signInToHuggingFaceAndWandbToUploadModelWeightsAndBiases():
    # Try to import kaggle_secrets only if available (i.e., on Kaggle)
    try:
        from kaggle_secrets import UserSecretsClient
        IS_KAGGLE = True
    except ImportError:
        IS_KAGGLE = False

    if IS_KAGGLE:
        print("Running on Kaggle. Using Kaggle secrets...")
        secrets = UserSecretsClient()
        hf_token = secrets.get_secret("HF_TOKEN")
        wandb_token = secrets.get_secret("WAND_DB_API_KEY")
    else:
        print("Running locally. Trying to load from .env...")
        try:
            load_dotenv()
            print(".env file loaded successfully.")
        except Exception as e:
            print(f"Warning: Could not load .env file. Exception: {e}")

        hf_token = os.getenv("HF_TOKEN")
        wandb_token = os.getenv("WAND_DB_API_KEY")

    # Hugging Face login
    try:
        if not hf_token:
            raise ValueError("Hugging Face token not found.")
        huggingface_hub.login(hf_token)
        print("Logged in to Hugging Face Hub successfully.")
    except Exception as e:
        print(f"Error during Hugging Face login: {e}")

    # Weights & Biases login
    try:
        if not wandb_token:
            raise ValueError("W&B token not found.")
        wandb.login(key=wandb_token)
        print("Logged in to Weights & Biases successfully.")
    except Exception as e:
        print(f"Error during W&B login: {e}")
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


def disableAnnoyingWarnings():
    # Caution! if anything goes wrong, enable it. make sure this warning related issue ain't the culprit!
    warnings.filterwarnings(
        "ignore",
        message="Length of IterableDataset",
        category=UserWarning,
        module="torch.utils.data.dataloader"
    )


class ComputeMetricsUsingSkLearn:
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
                "f1": f1_score(labels, predictions, average="weighted", zero_division=0),
                "confusion_matrix": confusion_matrix(labels, predictions).tolist(),  # convert to list for easier logging
            }
        except Exception as e:
            print(f">> Metrics computation failed: {e}")
            return {}

    def clear(self):
        self.logits.clear()
        self.labels.clear()

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


def parse_args() -> Namespace:
    # ------------------------
    # Default Config Values
    # ------------------------
    DEFAULT_MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
    DEFAULT_MODEL_VARIANT = "default"
    DEFAULT_RUN_NAME_PREFIX = "hyena-dna-mqtl-classifier"
    DEFAULT_WINDOW = 1024
    DEFAULT_NUM_EPOCHS = 10
    DEFAULT_PER_DEVICE_BATCH_SIZE = None  # dynamic
    DEFAULT_NUM_GPUS = None  # dynamic
    DEFAULT_ENABLE_LOGGING = False
    DEFAULT_EARLY_STOPPING = False
    DEFAULT_RUN_NAME_SUFFIX = None
    DEFAULT_SAVE_MODEL_IN_LOCAL_DIRECTORY = None
    DEFAULT_SAVE_MODEL_IN_REMOTE_REPOSITORY = None
    DEFAULT_LEARNING_RATE = 5e-5
    DEFAULT_L1_LAMBDA_WEIGHT = -1 # l1 regularization
    DEFAULT_WEIGHT_DECAY = 0.0 # L2 regularization
    DEFAULT_GRADIENT_CLIP = 1.0
    DEFAULT_OPTIMIZER = "adam"
    DEFAULT_DROP_OUT_PROBABILITY = 0.25
    DEFAULT_CRITERION_LABEL_SMOOTHENING = 0.1 # criterion_label_smoothening
    # ------------------------
    # Argument Parsing
    # ------------------------
    parser = argparse.ArgumentParser(description="Train or fine-tune HyenaDNA or DNA_BERT")

    parser.add_argument("--MODEL_NAME", type=str, default=DEFAULT_MODEL_NAME,
                        help="Pretrained model name or path")
    parser.add_argument("--MODEL_VARIANT", type=str, default=DEFAULT_MODEL_VARIANT,
                        help="Model variant")
    parser.add_argument("--RUN_NAME_PREFIX", type=str, default=DEFAULT_RUN_NAME_PREFIX,
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
    parser.add_argument("--EARLY_STOPPING", action="store_true", default=DEFAULT_EARLY_STOPPING,
                        help="Stop early if eval scores aren't updating.")

    # Optional: override naming
    parser.add_argument("--RUN_NAME_SUFFIX", type=str, default=DEFAULT_RUN_NAME_SUFFIX,
                        help="Override automatic run name suffix")
    parser.add_argument("--SAVE_MODEL_IN_LOCAL_DIRECTORY", type=str, default=DEFAULT_SAVE_MODEL_IN_LOCAL_DIRECTORY,
                        help="Custom local directory to save model")
    parser.add_argument("--SAVE_MODEL_IN_REMOTE_REPOSITORY", type=str, default=DEFAULT_SAVE_MODEL_IN_REMOTE_REPOSITORY,
                        help="Custom HuggingFace repo name to push model")

    parser.add_argument("--LEARNING_RATE", type=float, default=DEFAULT_LEARNING_RATE,
                        help="Set the learning rate")
    parser.add_argument("--L1_LAMBDA_WEIGHT", type=float, default=DEFAULT_L1_LAMBDA_WEIGHT,
                        help="Set the L1 regularization lambda weight")
    parser.add_argument("--WEIGHT_DECAY", type=float, default=DEFAULT_WEIGHT_DECAY,
                        help="Set the L2 regularization weight decay")
    parser.add_argument("--GRADIENT_CLIP", type=float, default=DEFAULT_GRADIENT_CLIP,
                        help="Set the gradient clipping")
    parser.add_argument("--DROP_OUT_PROBABILITY", type=float, default=DEFAULT_DROP_OUT_PROBABILITY,
                        help="Set the dropout probability")
    parser.add_argument("--CRITERION_LABEL_SMOOTHENING", type=float, default=DEFAULT_CRITERION_LABEL_SMOOTHENING,
                        help="Set the criterion label smoothening")
    parser.add_argument("--OPTIMIZER", type=str, default=DEFAULT_OPTIMIZER,
                        help="Set the optimizer")
    return parser.parse_args()

def get_optimizer(name, parameters, lr, weight_decay):
    if name == "adamw":
        return AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "lion":
        return Lion(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "adan":
        return Adan(parameters, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return Adam(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unsupported optimizer: {name}")


class MyArgs:
    MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
    MODEL_VARIANT = "default"
    RUN_NAME_PREFIX = "hyena-dna-mqtl-classifier"
    WINDOW = 1024
    NUM_EPOCHS = 10
    PER_DEVICE_BATCH_SIZE = None  # dynamic
    NUM_GPUS = None  # dynamic
    ENABLE_LOGGING = False
    RUN_NAME_SUFFIX = None
    SAVE_MODEL_IN_LOCAL_DIRECTORY = None
    SAVE_MODEL_IN_REMOTE_REPOSITORY = None
    LEARNING_RATE = 5e-5
    L1_LAMBDA_WEIGHT = -1  # l1 regularization
    WEIGHT_DECAY = 0.0  # L2 regularization
    GRADIENT_CLIP = 1.0
    OPTIMIZER = "adam"
    DROP_OUT_PROBABILITY = 0.25
    CRITERION_LABEL_SMOOTHENING = 0.1
    DEFAULT_EARLY_STOPPING = False