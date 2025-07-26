from argparse import Namespace

import torch
from transformers import AutoModel, BertTokenizer, AutoConfig

from models.models_hyenadna import *
from models.models_dnabert import *


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

    raise ValueError(f"Unknown base model: {base_model_name}, or model variant: {model_variant}")
