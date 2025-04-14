from datetime import datetime

import grelu
import pandas as pd
import torch
from datasets import Dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, \
    DataCollatorWithPadding, Trainer, BertTokenizer, BatchEncoding

from src.experiment.main import computeMetricsUsingTorchEvaluate, PagingMQTLDataset, WINDOW, MODEL_NAME, \
    SPLIT_SEQUENCE_REQUIRED


# def checkIfLabelsAreOk() # from manual inspection, ok.

# is preprocessing working correctly?
def getSmallDataset():
    pathPrefix = "/home/gamegame/PycharmProjects/mqtl-classification/"

    df_unfiltered = pd.read_csv(f"{pathPrefix}src/datageneration/dataset_{WINDOW}.csv")

    df = df_unfiltered.dropna(subset=["sequence"])
    df = df[df["sequence"].notnull()]
    df = df[df["sequence"] != ""]

    row = df.iloc[390]
    seq = row["sequence"]
    print(row)
    print(f"{seq = }")

    perform_binning = True
    file_suffix = ""

    list_of_dfs = []
    tmp_pos = df[(df["chrom"] == f"chr1") & (df["label"] == 1)].head(10)
    tmp_neg = df[(df["chrom"] == f"chr1") & (df["label"] == 0)].head(10)
    # print(f"chr{i} -> {tmp['chrom'] = }")
    list_of_dfs.append(tmp_pos)
    list_of_dfs.append(tmp_neg)

    binned_df = pd.concat(list_of_dfs, axis=0, ignore_index=True)
    print(f"{binned_df['chrom'] = }")
    print(f"{binned_df = }")

    tiny_df = binned_df.sample(frac=1, random_state=42).reset_index(drop=True)

    return tiny_df

# overfit 10 samples
def overfit10SamplesCheck():
    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    print("Model architecture:", config.architectures)

    mainTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    mainModel = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, trust_remote_code=True, num_labels=2)

    print(f"{type(config) = }")
    print(f"{type(mainModel.config) = }")

    assert type(config) == type(mainModel.config), f"Config type mismatch: {type(config)} != {type(mainModel.config)}"

    isGpuAvailable = torch.cuda.is_available()
    if isGpuAvailable:
        mainModel = mainModel.to("cuda")  # not sure if it is necessary in the kaggle / huggingface virtual-machine

    tiny_df = getSmallDataset()
    tiny_ds = Dataset.from_pandas(tiny_df)

    tinyPagingDf = PagingMQTLDataset(
        someDataset=tiny_ds,
        bertTokenizer=mainTokenizer,
        seqLength=WINDOW,
        splitSequenceRequired=SPLIT_SEQUENCE_REQUIRED,
        datasetLen=20
    )

    # Keep batch size small if needed (even 1 works)
    trainingArgs = TrainingArguments(
        output_dir="tiny-overfit-test",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        max_steps=1000,  # a few hundred is often enough to overfit
        learning_rate=1e-3,
        logging_dir="./logs",
        logging_steps=10,
        report_to="none",
        save_strategy="no",  # no need to save checkpoints
    )

    dataCollator = DataCollatorWithPadding(tokenizer=mainTokenizer)


    print("create trainer")
    trainer = Trainer(
        model=mainModel,
        args=trainingArgs,
        train_dataset=tinyPagingDf,  # train
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

    print("--------- Evaluation start ----------")

    for item in tinyPagingDf:
        print(item)

    test_results = trainer.evaluate(eval_dataset=tinyPagingDf)
    print(f"{test_results = }")
    pass


def kmers(seq, k=6):
    return [seq[i:i+k] for i in range(len(seq)-k+1)]

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


def sequenceEncodePlusWithSplitting(
        tokenizer: BertTokenizer,
        seq: str,
        label: int
) -> BatchEncoding:
    max_size = 512
    kmerSeq = toKmerSequence(seq, k=6)

    tempMap: BatchEncoding = tokenizer.encode_plus(
        kmerSeq,
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



def tokenizedKmers(mainTokenizer: BertTokenizer, seq, k=6):
    kmer_str = toKmerSequence(seq, k)
    tokenized_kmer_list = mainTokenizer.tokenize(kmer_str)
    print(f"{seq = }")
    print(f"{tokenized_kmer_list = }")
    print(f"{len(tokenized_kmer_list) = }")
    tokenizedIdsList = mainTokenizer.convert_tokens_to_ids(tokenized_kmer_list)
    print(f"{tokenizedIdsList = }")
    print(f"{len(tokenizedIdsList) = }")

    tempMap: BatchEncoding = mainTokenizer.encode_plus(
        kmer_str,
        add_special_tokens=False,  # we'll add the special tokens manually in the for loop below
        return_attention_mask=True,
        return_tensors="pt"
    )

    print(f"{tempMap = }")

    someInputIds1xN = tempMap["input_ids"]  # shape = 1xN , N = sequence length
    someMasks1xN = tempMap["attention_mask"]

    print(f"{someInputIds1xN = }")
    print(f"{someMasks1xN = }")
    pass


def howIsTokenizerWorking():
    mainTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    df = getSmallDataset()
    seq = df["sequence"][0]
    tokenizedKmers(mainTokenizer, seq, 6)

    encoded = sequenceEncodePlusWithSplitting(mainTokenizer, seq, 1)
    print(f"{encoded = }")
    pass

"""
without kmerSeq
    basically cls, 1, sep, and everything else was 0 definitely looks wrong
"""

"""
with kmerSeq. looks ok
encoded = {'input_ids': tensor([[ 101,  506, 2009,  ..., 1663, 2542,  102],
        [ 101, 1963, 3743,  ...,  282, 1113,  102],
        [ 101,  342, 1353,  ..., 1355, 1310,  102],
        [ 101, 1130,  410,  ...,    0,    0,    0]]), 'attention_mask': tensor([[1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 1, 1, 1],
        [1, 1, 1,  ..., 0, 0, 0]], dtype=torch.int32), 'labels': tensor(1)}
"""

def howIsTokenizerWorkingV1():
    mainTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    seq = "ATCGCC"
    tokenized = mainTokenizer.tokenize(seq)
    print(f"{type(tokenized) = }")
    print(f"{mainTokenizer.tokenize('ATATATAT') = }")
    print(f"{mainTokenizer.tokenize('CGCGCG ATATAT ') = }")
    print(f"{mainTokenizer.tokenize('AAAA AAA') = }")
    ids = mainTokenizer.convert_tokens_to_ids(mainTokenizer.tokenize('CGCGCG ATATAT '))
    print(f"{ids = }")
    pass


def preprocessingForDnaBert6():
    cls: torch.Tensor = torch.Tensor([101])
    sep: torch.Tensor = torch.Tensor([102])
    pad: torch.Tensor = torch.Tensor([0])

    mainTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

    df = getSmallDataset()
    seq = df["sequence"][0]
    print(f"{seq = }")
    mainTokenizer.tokenize(seq)
    kmerList: list[str] = kmers(seq, 6)
    kmerSize = len(kmerList) # 2000 --> 1995
    print(f"{kmerSize = }")

    remainder = kmerSize % 510  # 465
    q = kmerSize // 510

    if remainder != 0:
        paddingSize = 510 - remainder  # 45




    pass

if __name__ == '__main__':
    overfit10SamplesCheck()
    # preprocessingForDnaBert6()
    # howIsTokenizerWorking()
    pass

"""
DNA Bert 6

 0%|          | 0/1000 [00:00<?, ?it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
  1%|          | 10/1000 [00:02<03:38,  4.54it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 2.1787, 'grad_norm': 10.88424301147461, 'learning_rate': 0.000991, 'epoch': 1.0}
  2%|▏         | 20/1000 [00:04<03:34,  4.56it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 0.7694, 'grad_norm': 5.420258522033691, 'learning_rate': 0.000981, 'epoch': 2.0}

... ... ...

100%|██████████| 1000/1000 [03:51<00:00,  4.32it/s]
{'loss': 0.7423, 'grad_norm': 4.62071418762207, 'learning_rate': 1e-06, 'epoch': 100.0}
{'train_runtime': 231.3038, 'train_samples_per_second': 8.647, 'train_steps_per_second': 4.323, 'train_loss': 0.7349368896484375, 'epoch': 100.0}
-------Training completed. Results--------

TrainOutput(global_step=1000, training_loss=0.7349368896484375, metrics={'train_runtime': 231.3038, 'train_samples_per_second': 8.647, 'train_steps_per_second': 4.323, 'total_flos': 526222110720000.0, 'train_loss': 0.7349368896484375, 'epoch': 100.0})
--------- Evaluation start ----------
INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
100%|██████████| 10/10 [00:00<00:00, 15.28it/s]/home/gamegame/PycharmProjects/mqtl-classification/.venv/lib/python3.12/site-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
100%|██████████| 10/10 [00:00<00:00, 15.18it/s]
test_results = {'eval_loss': 0.6933879852294922, 'eval_accuracy': 0.5, 'eval_roc_auc': 0.5, 'eval_precision': 0.25, 'eval_recall': 0.5, 'eval_f1': 0.3333333333333333, 'eval_runtime': 0.725, 'eval_samples_per_second': 27.587, 'eval_steps_per_second': 13.793, 'epoch': 100.0}

"""

"""
hyenaDNA

  0%|          | 0/1000 [00:00<?, ?it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
  1%|          | 10/1000 [00:01<01:56,  8.49it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 0.6986, 'grad_norm': 1.1621005535125732, 'learning_rate': 0.000991, 'epoch': 1.0}
  2%|▏         | 20/1000 [00:02<01:52,  8.72it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 0.712, 'grad_norm': 0.0909879133105278, 'learning_rate': 0.000981, 'epoch': 2.0}

... ... 
{'loss': 0.0001, 'grad_norm': 0.0022754576057195663, 'learning_rate': 0.0005110000000000001, 'epoch': 49.0}
 50%|█████     | 500/1000 [00:59<00:58,  8.49it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 0.0001, 'grad_norm': 0.0021976681891828775, 'learning_rate': 0.000501, 'epoch': 50.0}
 51%|█████     | 510/1000 [01:00<00:57,  8.51it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 0.0001, 'grad_norm': 0.0021256727632135153, 'learning_rate': 0.000491, 'epoch': 51.0}
 52%|█████▏    | 520/1000 [01:01<00:56,  8.51it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 0.0001, 'grad_norm': 0.0020591081120073795, 'learning_rate': 0.000481, 'epoch': 52.0}

... ... ...

{'loss': 0.0001, 'grad_norm': 0.0011076482478529215, 'learning_rate': 2.1000000000000002e-05, 'epoch': 98.0}
 99%|█████████▉| 990/1000 [01:59<00:01,  8.31it/s]INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 0.0001, 'grad_norm': 0.0011071667540818453, 'learning_rate': 1.1e-05, 'epoch': 99.0}
100%|██████████| 1000/1000 [02:00<00:00,  8.30it/s]
INFO:root:num_workers = 1, worker_id = 0, rank = 0, world_size=1
{'loss': 0.0001, 'grad_norm': 0.0011069742031395435, 'learning_rate': 1e-06, 'epoch': 100.0}
{'train_runtime': 120.4288, 'train_samples_per_second': 16.607, 'train_steps_per_second': 8.304, 'train_loss': 0.1322111348991748, 'epoch': 100.0}
-------Training completed. Results--------

TrainOutput(global_step=1000, training_loss=0.1322111348991748, metrics={'train_runtime': 120.4288, 'train_samples_per_second': 16.607, 'train_steps_per_second': 8.304, 'total_flos': 157190519808000.0, 'train_loss': 0.1322111348991748, 'epoch': 100.0})
--------- Evaluation start ----------
100%|██████████| 10/10 [00:00<00:00, 24.71it/s]
test_results = {'eval_loss': 5.518041507457383e-05, 'eval_accuracy': 1.0, 'eval_roc_auc': 1.0, 'eval_precision': 1.0, 'eval_recall': 1.0, 'eval_f1': 1.0, 'eval_runtime': 0.4566, 'eval_samples_per_second': 43.801, 'eval_steps_per_second': 21.901, 'epoch': 100.0}

Process finished with exit code 0

"""