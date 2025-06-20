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
    metrics_str = f"\n📊 {stage} Metrics:\n" + "\n".join(
        f"  {k:>15}: {v:.4f}" if v is not None else f"  {k:>15}: N/A"
        for k, v in metrics.items()
    )
    print(metrics_str)


class PagingMQTLDataset(IterableDataset):
    def __init__(
            self,
            someDataset: Dataset,
            dnaSeqTokenizer: PreTrainedTokenizer,
            seqLength: int,
            toKmer: bool,
            datasetLen: int
    ):
        self.someDataset = someDataset
        self.dnaSeqTokenizer = dnaSeqTokenizer
        self.seqLength = seqLength
        self.toKmer = toKmer
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
                "f1": f1_score(labels, predictions, average="weighted", zero_division=0)
            }
        except Exception as e:
            print(f">> Metrics computation failed: {e}")
            return {}

    def clear(self):
        self.logits.clear()
        self.labels.clear()
