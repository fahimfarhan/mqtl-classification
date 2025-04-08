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
import logging
import os
import wandb
import huggingface_hub
from dotenv import load_dotenv

from torch.utils.data import DataLoader, IterableDataset

""" import dependencies """
from transformers import BertTokenizer, BatchEncoding
import torch


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
def sequenceEncodePlusDefault(
    tokenizer: BertTokenizer,
    sequence: str,
    label: int
) -> BatchEncoding:
    tempMap: BatchEncoding = tokenizer.encode_plus(
        sequence,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt"
    )

    someInputIds1xN = tempMap["input_ids"]  # shape = 1xN , N = sequence length
    someMasks1xN = tempMap["attention_mask"]
    input_ids: torch.Tensor = torch.Tensor(someInputIds1xN)
    attention_mask: torch.Tensor = torch.Tensor(someMasks1xN)

    label_tensor = torch.tensor(label)

    encoded_map: dict = {
        "input_ids": input_ids.long(),
        "attention_mask": attention_mask.int(),
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
    input_ids: torch.Tensor = torch.stack(inputIdsList)
    attention_mask: torch.Tensor = torch.stack(masksList)
    label_tensor = torch.tensor(label)

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
        return sequenceEncodePlusDefault(tokenizer, sequence, label)


class PagingMQTLDataset(IterableDataset):
    def __init__(
            self,
            someDataset,  # todo: set type
            bertTokenizer: BertTokenizer,
            seqLength: int,
            splitSequenceRequired: bool
        ):
        self.someDataset = someDataset
        self.bertTokenizer = bertTokenizer
        self.seqLength = seqLength
        self.splitSequenceRequired = splitSequenceRequired
        pass

    def __iter__(self):
        for row in self.someDataset:
            processed = self.preprocess(row)
            if processed is not None:
                yield processed

    def preprocess(self, row: dict):
        sequence = row["sequence"]
        label = row["label"]

        if len(sequence) != self.seqLength:
            return None  # skip a few problematic rows

        return sequenceEncodePlusCompact(self.splitSequenceRequired, self.bertTokenizer, sequence, label)


def signin_to_huggingface_and_wandb_to_upload_model_weights_and_biases():
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


""" dynamic section. may be some consts,  changes based on model, etc. Try to keep it as small as possible """

MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
WINDOW = 4000
BATCH_SIZE = getDynamicBatchSize()


""" main """
def start():
    timber.info(green)
    timber.info("---Inside start function---")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    signin_to_huggingface_and_wandb_to_upload_model_weights_and_biases()

    pass

if __name__ == "__main__":
    # for some reason, the variables in the main function act like global variables in python
    # hence other functions get confused with the "global" variables. easiest solution, write everything
    # in another function (say, start()), and call it inside the main
    start()
    pass
