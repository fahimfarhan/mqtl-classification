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
import logging
import os

import evaluate
import huggingface_hub
import numpy as np
import wandb
from datasets import load_dataset, Dataset
from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
from torch.utils.data import IterableDataset
from transformers import BertTokenizer, BatchEncoding, AutoTokenizer, \
    AutoModelForSequenceClassification, AutoConfig, TrainingArguments, Trainer, DataCollatorWithPadding
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
    # someMasks1xN = tempMap["attention_mask"]   # does not exist for hyena dna :/
    input_ids: torch.Tensor = torch.Tensor(someInputIds1xN)
    # attention_mask: torch.Tensor = torch.Tensor(someMasks1xN)

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
            someDataset: Dataset,
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

def createPagingTrainValTestDatasets(tokenizer, window, splitSequenceRequired) -> (PagingMQTLDataset, PagingMQTLDataset, PagingMQTLDataset):
    prefix = "/home/gamegame/PycharmProjects/mqtl-classification/"
    data_files = {
        # small samples
        "train_binned_200": f"{prefix}src/datageneration/dataset_200_train_binned.csv",
        "validate_binned_200": f"{prefix}src/datageneration/dataset_200_validate_binned.csv",
        "test_binned_200": f"{prefix}src/datageneration/dataset_200_test_binned.csv",
        # medium samples
        "train_binned_1000": f"{prefix}src/datageneration/dataset_1000_train_binned.csv",
        "validate_binned_1000": f"{prefix}src/datageneration/dataset_1000_train_binned.csv",
        "test_binned_1000": f"{prefix}src/datageneration/dataset_1000_train_binned.csv",

        # large samples
        "train_binned_4000": f"{prefix}src/datageneration/dataset_4000_train_binned.csv",
        "validate_binned_4000": f"{prefix}src/datageneration/dataset_4000_train_binned.csv",
        "test_binned_4000": f"{prefix}src/datageneration/dataset_4000_train_binned.csv",
    }

    dataset_map = None
    is_my_laptop = os.path.isfile("/home/gamegame/PycharmProjects/mqtl-classification/src/datageneration/dataset_4000_train_binned.csv")
    if is_my_laptop:
        dataset_map = load_dataset("csv", data_files=data_files, streaming=True)
    else:
        dataset_map = load_dataset("fahimfarhan/mqtl-classification-datasets", streaming=True)

    train_dataset = PagingMQTLDataset(someDataset=dataset_map[f"train_binned_{window}"],
                                    bertTokenizer=tokenizer,
                                    seqLength=window,
                                    splitSequenceRequired=splitSequenceRequired
                                    )
    val_dataset = PagingMQTLDataset(dataset_map[f"validate_binned_{window}"],
                                  bertTokenizer=tokenizer,
                                  seqLength=window,
                                  splitSequenceRequired=splitSequenceRequired
                                  )
    test_dataset = PagingMQTLDataset(dataset_map[f"test_binned_{window}"],
                                   bertTokenizer=tokenizer,
                                   seqLength=window,
                                   splitSequenceRequired=splitSequenceRequired
                                   )
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


""" dynamic section. may be some consts,  changes based on model, etc. Try to keep it as small as possible """

RUN_NAME = "hyena-dna-mqtl-classifier"
MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
SPLIT_SEQUENCE_REQUIRED=False
WINDOW = 200

SAVE_MODEL_IN_LOCAL_DIRECTORY= f"fine-tuned-{RUN_NAME}-{WINDOW}"
SAVE_MODEL_IN_REMOTE_REPOSITORY = f"fahimfarhan/{RUN_NAME}-{WINDOW}"


NUM_ROWS = 20 # 20_000    # hardcoded value
PER_DEVICE_BATCH_SIZE = getDynamicBatchSize()
EPOCHS = 3
NUM_GPUS = max(torch.cuda.device_count(), 1)  # fallback to 1 if no GPU

effective_batch_size = PER_DEVICE_BATCH_SIZE * NUM_GPUS
STEPS_PER_EPOCH = NUM_ROWS // effective_batch_size
MAX_STEPS = EPOCHS * STEPS_PER_EPOCH

""" main """
def start():
    timber.info(green)
    timber.info("---Inside start function---")

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # signInToHuggingFaceAndWandbToUploadModelWeightsAndBiases()
    wandb.init(mode="offline")  # Logs only locally


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
        save_steps=STEPS_PER_EPOCH,
        logging_steps=STEPS_PER_EPOCH,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        max_steps=MAX_STEPS,
        weight_decay=0.01,
        learning_rate=1e-3,
        logging_dir="./logs"
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


    # train, and validate
    result = trainer.train()
    try:
        print("-------Training completed. Results--------\n")
        print(result)
    except Exception as x:
        print(f"{x = }")

    test_results = trainer.evaluate(eval_dataset=test_dataset)
    try:
        print("-------Test completed. Results--------\n")
        print(test_results)
    except Exception as x:
        print(f"{x = }")

    mainModel.save_pretrained(save_directory=SAVE_MODEL_IN_LOCAL_DIRECTORY, safe_serialization=False)
    # push to the hub
    is_my_laptop = os.path.isfile("/home/gamegame/PycharmProjects/mqtl-classification/src/datageneration/dataset_4000_train_binned.csv")

    commit_message = f":tada: Push model for window size {WINDOW} from huggingface space"
    if is_my_laptop:
      commit_message = f":tada: Push model for window size {WINDOW} from my laptop"

    mainModel.push_to_hub(
      repo_id=SAVE_MODEL_IN_REMOTE_REPOSITORY,
      # subfolder=f"my-awesome-model-{WINDOW}", subfolder didn't work :/
      commit_message=commit_message,  # f":tada: Push model for window size {WINDOW}"
      safe_serialization=False
    )
    pass

if __name__ == "__main__":
    # for some reason, the variables in the main function act like global variables in python
    # hence other functions get confused with the "global" variables. easiest solution, write everything
    # in another function (say, start()), and call it inside the main
    start()
    pass
