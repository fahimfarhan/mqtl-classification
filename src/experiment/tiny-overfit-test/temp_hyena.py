import evaluate
import numpy as np
import pandas as pd
import wandb
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer, logging
import torch


# Load metrics
# global variables. bad practice...
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")
roc_auc_metric = evaluate.load("roc_auc")
precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")

def computeMetricsTest(args):
    logits, labels = args

    print(">> logits:", logits)
    print(">> labels:", labels)

    predictions = np.argmax(logits, axis=1)
    positive_logits = logits[:, 1]  # <-- Might crash here if logits shape is (N, 1) instead of (N, 2)

    try:
        accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
        f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
        roc_auc = roc_auc_metric.compute(prediction_scores=positive_logits, references=labels)["roc_auc"]
        precision = precision_metric.compute(predictions=predictions, references=labels, average="weighted")["precision"]
        recall = recall_metric.compute(predictions=predictions, references=labels, average="weighted")["recall"]
    except Exception as e:
        print(f">> Metrics computation failed: {e}")
        return {}

    return {
        "accuracy": accuracy,
        "roc_auc": roc_auc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def start():
    # instantiate pretrained model
    checkpoint = 'LongSafari/hyenadna-medium-160k-seqlen-hf'

    wandb.init(mode="offline")  # Logs only locally

    # bfloat16 for better speed and reduced memory usage
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)

    # Generate some random sequence and labels
    # If you're copying this code, replace the sequences and labels
    # here with your own data!

    # tokenized = tokenizer(sequence)["input_ids"]


    df = pd.read_csv("/home/gamegame/PycharmProjects/mqtl-classification/src/experiment/tiny-overfit-test/temp_sample.csv")
    df = Dataset.from_pandas(df)


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


    # Create a dataset for training
    ds = df.map(preprocess)
    ds.set_format("pt")

    # Initialize Trainer
    # Note that we're using extremely small batch sizes to maximize
    # our ability to fit long sequences in memory!
    args = {
        "output_dir": "tmp",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "gradient_checkpointing": False,
        "learning_rate": 2e-5,
        # "safe_serialization": False,
    }

    training_args = TrainingArguments(
        output_dir="demo",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=100,  # a few hundred is often enough to overfit
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        save_strategy="no",  # no need to save checkpoints

        learning_rate=5e-5,  # or tune
        weight_decay=0.0,  # <--- here
        max_grad_norm=1.0,  # <--- here

    )
    dataCollator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds,
        eval_dataset = ds,
        data_collator = dataCollator,
        compute_metrics = computeMetricsTest
    )

    try:
        # train, and validate
        trainingResult = trainer.train()
        print("-------Training completed. Results--------\n")
        print(f"{trainingResult = }")
    except Exception as x:
        print(f"{x = }")

    try:
        print("\n-------Test Results--------\n")
        # eval_dataset = Dataset.from_dict(inputs)
        for item in ds:
            print(f"item = {item}")
        test_results = trainer.evaluate(eval_dataset=ds)
        print(f"{test_results = }")
    except Exception as x:
        print(f"{x = }")

if __name__ == '__main__':
    start()
    pass
