from argparse import Namespace

import torch
from transformers import AutoModel, BertTokenizer, AutoConfig

from models.models_hyenadna import *
from models.models_dnabert import *
from models.belugamqtlclassifier import *
from Extensions import PagingMQTLDataset, toKmerSequence

class MQTLStreamingDataset(PagingMQTLDataset):
    def preprocess(self, row: dict):
        if self.inputArgs.MODEL_NAME == "LongSafari/hyenadna-small-32k-seqlen-hf":
            return self.preprocess_hyena_dna(row = row)
        elif  self.inputArgs.MODEL_NAME == "zhihan1996/DNA_bert_6":
            return self.preprocess_dna_bert_6(row = row)
        elif self.inputArgs.MODEL_NAME == "DeepSEA/beluga":
            return self.preprocess_beluga(row = row)
        else:
            raise Exception(f"unknown model name {self.inputArgs.model_name}")

    def preprocess_beluga(self, row: dict):
        sequence = row["sequence"]
        label = row["label"]

        encoded = preprocess_beluga_encode_seqs(seqs = [sequence], input_size=self.inputArgs.WINDOW)
        encoded_tensor = torch.tensor(encoded, dtype=torch.float32)  # convert to Tensor
        label_tensor: torch.Tensor = torch.tensor(label)

        encoded_map: dict = {
            "ohe_sequences": encoded_tensor,
            "labels": label_tensor,
        }
        return encoded_map

    def preprocess_hyena_dna(self, row: dict):
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

    def preprocess_dna_bert_6(self, row: dict):
        sequence = row["sequence"]
        label = row["label"]

        kmerSeq = toKmerSequence(sequence)
        kmerSeqTokenized = self.dnaSeqTokenizer(
            kmerSeq,
            max_length=self.inputArgs.WINDOW,
            # self.seqLength, # I messed up with passing seqLength somewhere. For now, set the global variable WINDOW
            padding='max_length',
            return_tensors="pt"
        )
        input_ids = kmerSeqTokenized["input_ids"]
        attention_mask = kmerSeqTokenized["attention_mask"]
        input_ids: torch.Tensor = torch.Tensor(input_ids)
        attention_mask = torch.Tensor(attention_mask)
        label_tensor = torch.tensor(label)
        encoded_map: dict = {
            "input_ids": input_ids.long(),
            "attention_mask": attention_mask.int(),  # hyenaDNA does not have attention layer
            "labels": label_tensor
        }

        return encoded_map


def getModel(args: Namespace, dnaTokenizer: BertTokenizer) -> nn.Module:
    base_model_name = args.MODEL_NAME
    model_variant = args.MODEL_VARIANT

    if base_model_name == "LongSafari/hyenadna-small-32k-seqlen-hf":
        if model_variant == "default":
            return AutoModelForSequenceClassification.from_pretrained(
                base_model_name,
                trust_remote_code=True,
            )
        if model_variant == "HyenaDNAWithDropoutAndNorm":
            return HyenaDNAWithDropoutAndNorm(
                model_name = base_model_name,
                dropout_prob = args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )
        if model_variant == "HyenaDNAWithDropout":
            return HyenaDNAWithDropout(
                model_name = base_model_name,
                dropout_prob = args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )

        if model_variant == "HyenaDNAWithDropoutNormAndFrozenHyena":
            return HyenaDNAWithDropoutNormAndFrozenHyena(
                model_name=base_model_name,
                dropout_prob=args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )
        if model_variant == "HyenaDNAWithDropoutBatchNorm1d":
            return HyenaDNAWithDropoutBatchNorm1d(
                model_name=base_model_name,
                dropout_prob=args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )
        if model_variant == "HyenaDNAWithDropoutBatchNorm1dAndFrozenHyena":
            return HyenaDNAWithDropoutBatchNorm1dAndFrozenHyena(
                model_name=base_model_name,
                dropout_prob=args.DROP_OUT_PROBABILITY,
                criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            )

    if base_model_name == "zhihan1996/DNA_bert_6":
        if model_variant == "default":
            baseModel = AutoModel.from_pretrained(base_model_name,
                                                  trust_remote_code=True)  # this is the correct way to load pretrained weights, and modify config
            baseModel.gradient_checkpointing_enable()  # bert model's builtin way to enable gradient check pointing

            # print("-------update some more model configs start-------")
            baseModel.resize_token_embeddings(len(dnaTokenizer))
            baseModel.config.max_position_embeddings = args.WINDOW
            baseModel.embeddings.position_embeddings = torch.nn.Embedding(args.WINDOW, baseModel.config.hidden_size)
            # print(baseModel)
            # print("--------update some more model configs end--------")

            someConfig = AutoConfig.from_pretrained(base_model_name, trust_remote_code=True)
            someConfig.split = (args.WINDOW // 512)  # hmm. so it works upto 7 on my laptop. if 8, then OutOfMemoryError
            # mainModel = BertForLongSequenceClassification.from_pretrained(model_name, config=someConfig, trust_remote_code=True) # this is the correct way to load pretrained weights, and modify config
            someConfig.max_position_embeddings = args.WINDOW
            someConfig.rnn = "gru"  # or "lstm". Let's check if it works
            mainModel = BertForLongSequenceClassification(someConfig)
            mainModel.bert = baseModel

            return mainModel

    if base_model_name == "":
        return BelugaMQTLClassifier(
            dropout_prob=args.DROP_OUT_PROBABILITY,
            criterion_label_smoothening=args.CRITERION_LABEL_SMOOTHENING,
            finetune=args.KIPOI_FINE_TUNE,
        )
    raise ValueError(f"Unknown base model: {base_model_name}, or model variant: {model_variant}")
