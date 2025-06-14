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
from typing import Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
from torch.nn import MSELoss, CrossEntropyLoss
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import BertPreTrainedModel, AutoModel, AutoConfig, AutoTokenizer
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BertModel, BERT_INPUTS_DOCSTRING

try:
    from src.experiment.Extensions import *
except ImportError as ie:
    print(ie)

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # hyena dna requires this
print("import dependencies completed")

""" dynamic section. may be some consts,  changes based on model, etc. Try to keep it as small as possible """
""" THIS IS THE MOST IMPORTANT PART """

# MODEL_NAME = "LongSafari/hyenadna-small-32k-seqlen-hf"
# run_name_prefix = "hyena-dna-mqtl-classifier"
MODEL_NAME =  "zhihan1996/DNA_bert_6"
run_name_prefix = "dna-bert-6-mqtl-classifier"

run_name_suffix = datetime.now().strftime("%Y-%m-%d-%H-%M")
# run_platform="laptop"

CONVERT_TO_KMER= (MODEL_NAME == "zhihan1996/DNA_bert_6")
WINDOW = 1024  # use small window on your laptop gpu (eg nvidia rtx 2k), and large window on datacenter gpu (T4, P100, etc)
RUN_NAME = f"{run_name_prefix}-{WINDOW}-{run_name_suffix}"
SAVE_MODEL_IN_LOCAL_DIRECTORY= f"fine-tuned-{RUN_NAME}"
SAVE_MODEL_IN_REMOTE_REPOSITORY = f"fahimfarhan/{RUN_NAME}"

NUM_EPOCHS = 1
PER_DEVICE_BATCH_SIZE = getDynamicBatchSize()
NUM_GPUS = max(torch.cuda.device_count(), 1)  # fallback to 1 if no GPU

# use it for step based implementation (huggingface trainer library)
# NUM_ROWS = 2_000    # hardcoded value
# EPOCHS = 1
# effective_batch_size = PER_DEVICE_BATCH_SIZE * NUM_GPUS
# STEPS_PER_EPOCH = NUM_ROWS // effective_batch_size
# MAX_STEPS = EPOCHS * STEPS_PER_EPOCH

print("init arguments completed")

""" Common codes """
class DNaBert6PagingMQTLDataset(PagingMQTLDataset):
    def preprocess(self, row: dict):
        sequence = row["sequence"]
        label = row["label"]

        kmerSeq = toKmerSequence(sequence)
        kmerSeqTokenized = self.bertTokenizer(
            kmerSeq,
            max_length=self.seqLength, # 2048,
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


""" main """


def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator

def add_start_docstrings_to_callable(*docstr):
    def docstring_decorator(fn):
        class_name = ":class:`~transformers.{}`".format(fn.__qualname__.split(".")[0])
        intro = "   The {} forward method, overrides the :func:`__call__` special method.".format(class_name)
        note = r"""

    .. note::
        Although the recipe for forward pass needs to be defined within
        this function, one should call the :class:`Module` instance afterwards
        instead of this since the former takes care of running the
        pre and post processing steps while the latter silently ignores them.
        """
        fn.__doc__ = intro + note + "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


@add_start_docstrings(
    """Bert Model transformer with a sequence classification/regression head on top (a linear layer on top of
    the pooled output) e.g. for GLUE tasks. Especially designed for sequences longer than 512. """,
    BERT_START_DOCSTRING,
)
class BertForLongSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.split = config.split
        self.rnn_type = config.rnn
        self.num_rnn_layer = config.num_rnn_layer
        self.hidden_size = config.hidden_size
        self.rnn_dropout = config.rnn_dropout
        self.rnn_hidden = config.rnn_hidden

        self.bert = BertModel(config)
        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True,
                               num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(input_size=self.hidden_size, hidden_size=self.hidden_size, bidirectional=True,
                              num_layers=self.num_rnn_layer, batch_first=True, dropout=self.rnn_dropout)
        else:
            raise ValueError
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.hidden_size, self.config.num_labels)

        self.init_weights()

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            overlap=100,
            max_length_per_seq=500,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
            Classification (or regression if config.num_labels==1) loss.
        logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
            Classification (or regression if config.num_labels==1) scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)

        loss, logits = outputs[:2]

        """
        # batch_size = input_ids.shape[0]
        # sequence_length = input_ids.shape[1]
        # starts = []
        # start = 0
        # while start + max_length_per_seq <= sequence_length:
        #     starts.append(start)
        #     start += (max_length_per_seq-overlap)
        # last_start = sequence_length-max_length_per_seq
        # if last_start > starts[-1]:
        #     starts.append(last_start)

        # new_input_ids = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=input_ids.dtype, device=input_ids.device)
        # new_attention_mask = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=attention_mask.dtype, device=attention_mask.device)
        # new_token_type_ids = torch.zeros([len(starts)*batch_size, max_length_per_seq], dtype=token_type_ids.dtype, device=token_type_ids.device)

        # for j in range(batch_size):
        #     for i, start in enumerate(starts):
        #         new_input_ids[i] = input_ids[j,start:start+max_length_per_seq]
        #         new_attention_mask[i] = attention_mask[j,start:start+max_length_per_seq]
        #         new_token_type_ids[i] = token_type_ids[j,start:start+max_length_per_seq]

        # if batch_size == 1:
        #     pooled_output = outputs[1].mean(dim=0)
        #     pooled_output = pooled_output.reshape(1, pooled_output.shape[0])
        # else:
        #     pooled_output = torch.zeros([batch_size, outputs[1].shape[1]], dtype=outputs[1].dtype)
        #     for i in range(batch_size):
        #         pooled_output[i] = outputs[1][i*batch_size:(i+1)*batch_size].mean(dim=0)

        batch_size = input_ids.shape[0]
        # print(f"debug: {batch_size = }")
        # print(f"debug: {self.split = }")
        # print(f"debug: {input_ids.shape = }")
        # print(f"debug: {input_ids.size = }")
        input_ids = input_ids.view(self.split * batch_size, 512)
        attention_mask = attention_mask.view(self.split * batch_size, 512)
        token_type_ids = None

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        # lstm
        if self.rnn_type == "lstm":
            # random
            # h0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))/100.0
            # c0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))/100.0
            # self.hidden = (h0, c0)
            # self.rnn.flatten_parameters()
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, (ht, ct) = self.rnn(pooled_output, self.hidden)

            # orth
            # h0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(h0)
            # h0 = autograd.Variable(h0)
            # c0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(c0)
            # c0 = autograd.Variable(c0)
            # self.hidden = (h0, c0)
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, (ht, ct) = self.rnn(pooled_output, self.hidden)

            # zero
            pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            _, (ht, ct) = self.rnn(pooled_output)
        elif self.rnn_type == "gru":
            # h0 = autograd.Variable(torch.randn([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device))
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, ht = self.rnn(pooled_output, h0)

            # h0 = torch.empty([2*self.num_rnn_layer, batch_size, self.hidden_size], device=input_ids.device)
            # nn.init.orthogonal_(h0)
            # h0 = autograd.Variable(h0)
            # pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            # _, ht = self.rnn(pooled_output, h0)

            pooled_output = outputs[1].view(batch_size, self.split, self.hidden_size)
            _, ht = self.rnn(pooled_output)
        else:
            raise ValueError

        output = self.dropout(ht.squeeze(0).sum(dim=0))
        logits = self.classifier(output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class MQTLClassifierModule(pl.LightningModule):
    def __init__(self, model, learning_rate=5e-5, weight_decay=0.0, max_grad_norm=1.0):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.max_grad_norm = max_grad_norm
        self.criterion = torch.nn.CrossEntropyLoss()

        self.train_metrics = ComputeMetricsUsingSkLearn()
        self.val_metrics = ComputeMetricsUsingSkLearn()
        self.test_metrics = ComputeMetricsUsingSkLearn()

    def forward(self, batch):
        loss, logits = self.model(**batch)
        return loss, logits

    def training_step(self, batch, batch_idx)-> STEP_OUTPUT:
        with torch.autograd.set_detect_anomaly(True):  # Anomaly detection enabled here
            labels = batch["labels"]
            loss, logits = self.forward(batch)

            # Log the loss
            self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

            # Update training metrics
            self.train_metrics.update(logits=logits, labels=labels)

            return loss

    def on_after_backward(self):
        # Compute and log gradient norm
        total_norm = 0.0
        for name, param in self.named_parameters():
            if param.grad is not None:
                # self.logger.experiment.add_histogram(f"{name}_grad", param.grad, self.global_step)
                # self.logger.experiment.add_histogram(f"{name}_weights", param, self.global_step)
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        self.log("grad_norm", total_norm, prog_bar=True, on_step=True, on_epoch=False)

    def on_train_epoch_end(self) -> None:
        metrics = self.train_metrics.compute()
        self.train_metrics.clear()
        # for k, v in metrics.items():
        #     self.log(f"train_{k}", v, prog_bar=True)
        pretty_print_metrics(metrics, "Train")
        pass

    def validation_step(self, batch, batch_idx) -> STEP_OUTPUT:
        labels = batch["labels"]
        loss, logits = self.forward(batch)
        self.val_metrics.update(logits=logits, labels=labels)
        return loss

    def on_validation_epoch_end(self) -> None:
        metrics = self.val_metrics.compute()
        self.val_metrics.clear()
        # for k, v in metrics.items():
        #     self.log(f"eval_{k}", v, prog_bar=True)
        pretty_print_metrics(metrics, "Eval")
        pass

    def test_step(self, batch, batch_idx) -> STEP_OUTPUT:
        labels = batch["labels"]
        loss, logits = self.forward(batch)
        self.test_metrics.update(logits=logits, labels=labels)
        return loss

    def on_test_epoch_end(self) -> None:
        metrics = self.test_metrics.compute()
        self.test_metrics.clear()
        # for k, v in metrics.items():
        #     self.log(f"test_{k}", v, prog_bar=True)
        pretty_print_metrics(metrics, "Test")
        pass

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        return optimizer

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)

class DnaBert6PagingMQTLDataset(PagingMQTLDataset):
    def preprocess(self, row: dict):
        sequence = row["sequence"]
        label = row["label"]

        kmerSeq = toKmerSequence(sequence)
        kmerSeqTokenized = self.bertTokenizer(
            kmerSeq,
            max_length=WINDOW,
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


def createSingleDnaBert6PagingDatasets(
        data_files,
        split,
        tokenizer,
        window,
        splitSequenceRequired
) -> DnaBert6PagingMQTLDataset:  # I can't come up with creative names
    is_my_laptop = isMyLaptop()
    if not is_my_laptop:
        dataset_map = load_dataset("csv", data_files=data_files, streaming=True)
        dataset_len = get_dataset_length(local_path=data_files[split], split=split)
    else:
        dataset_map = load_dataset("fahimfarhan/mqtl-classification-datasets", streaming=True)
        dataset_len = get_dataset_length(dataset_name="fahimfarhan/mqtl-classification-datasets", split=split)

    print(f"{split = } ==> {dataset_len = }")
    return DnaBert6PagingMQTLDataset(
        someDataset=dataset_map[f"train_binned_{window}"],
        bertTokenizer=tokenizer,
        seqLength=window,
        toKmer=splitSequenceRequired,
        datasetLen = dataset_len
    )

def createDnaBert6PagingTrainValTestDatasets(tokenizer, window, toKmer) -> (DnaBert6PagingMQTLDataset, DnaBert6PagingMQTLDataset, DnaBert6PagingMQTLDataset):
    prefix = "/home/gamegame/PycharmProjects/mqtl-classification/src/datageneration/"

    data_files = {
        # small
        "train_binned_1027": f"{prefix}_1027_train_binned.csv",
        "validate_binned_1027": f"{prefix}_1027_validate_binned.csv",
        "test_binned_1027": f"{prefix}_1027_train_binned.csv",

        # medium
        "train_binned_2051": f"{prefix}_2051_train_binned.csv",
        "validate_binned_2051": f"{prefix}_2051_validate_binned.csv",
        "test_binned_2051": f"{prefix}_2051_test_binned.csv",

        # large
        "train_binned_4099": f"{prefix}_4099_train_binned.csv",
        "validate_binned_4099": f"{prefix}_4099_validate_binned.csv",
        "test_binned_4099": f"{prefix}_4099_test_binned.csv",
    }

    # not sure if this is a good idea. if anything goes wrong, revert back to previous code of this function
    train_dataset = createSingleDnaBert6PagingDatasets(data_files, f"train_binned_{window}", tokenizer, window, toKmer)

    val_dataset =createSingleDnaBert6PagingDatasets(data_files, f"validate_binned_{window}", tokenizer, window, toKmer)

    test_dataset = createSingleDnaBert6PagingDatasets(data_files, f"test_binned_{window}", tokenizer, window, toKmer)

    return train_dataset, val_dataset, test_dataset


def save_fine_tuned_model(mainModel):
    # save the model in huggingface repository, and local storage
    mainModel.save_pretrained(save_directory=SAVE_MODEL_IN_LOCAL_DIRECTORY, safe_serialization=False)
    # push to the hub
    is_my_laptop = isMyLaptop()

    if is_my_laptop:  # no need to save
        return

    # commit_message = f":tada: Push {RUN_NAME} model for window size {WINDOW} from my laptop"
    commit_message = f":tada: Push {RUN_NAME} model for window size {WINDOW} into huggingface hub"
    mainModel.push_to_hub(
        repo_id=SAVE_MODEL_IN_REMOTE_REPOSITORY,
        # subfolder=f"my-awesome-model-{WINDOW}", subfolder didn't work :/
        commit_message=commit_message,
        safe_serialization=False
    )

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


    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"  # OutOfMemoryError

    model_name = MODEL_NAME

    dnaTokenizer = BertTokenizer.from_pretrained(model_name, trust_remote_code=True)
    baseModel = AutoModel.from_pretrained(model_name, trust_remote_code=True) # this is the correct way to load pretrained weights, and modify config
    baseModel.gradient_checkpointing_enable()  #  bert model's builtin way to enable gradient check pointing

    print("-------update some more model configs start-------")
    baseModel.resize_token_embeddings(len(dnaTokenizer))
    baseModel.config.max_position_embeddings = WINDOW
    baseModel.embeddings.position_embeddings = torch.nn.Embedding(WINDOW, baseModel.config.hidden_size)
    print(baseModel)
    print("--------update some more model configs end--------")

    someConfig = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    someConfig.split = (WINDOW // 512)  # hmm. so it works upto 7 on my laptop. if 8, then OutOfMemoryError
    # mainModel = BertForLongSequenceClassification.from_pretrained(model_name, config=someConfig, trust_remote_code=True) # this is the correct way to load pretrained weights, and modify config
    someConfig.max_position_embeddings = WINDOW
    someConfig.rnn = "gru" # or "lstm". Let's check if it works
    mainModel = BertForLongSequenceClassification(someConfig)
    mainModel.bert = baseModel

    print(mainModel)

    dataCollator = DataCollatorWithPadding(tokenizer=dnaTokenizer)

    # L = T + k - 3 [for dna bert 6, we have 2 extra tokens, cls, and sep]
    rawSequenceLength = WINDOW + 6 - 3
    train_dataset, val_dataset, test_dataset = createDnaBert6PagingTrainValTestDatasets(tokenizer=dnaTokenizer, window=rawSequenceLength, toKmer=CONVERT_TO_KMER)


    train_loader = DataLoader(train_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, collate_fn=dataCollator) # Can't shuffle the paging/streaming datasets
    val_loader = DataLoader(val_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, collate_fn=dataCollator)
    test_loader = DataLoader(test_dataset, batch_size=PER_DEVICE_BATCH_SIZE, shuffle=False, collate_fn=dataCollator)

    print("create trainer")

    trainer = pl.Trainer(
        max_epochs=NUM_EPOCHS,  # instead of max_steps
        limit_train_batches=None,  # 100% of data each epoch
        val_check_interval=1.0,  # validate at end of each epoch
        enable_progress_bar=True,
        enable_model_summary=True,
        gradient_clip_val=None,
        accumulate_grad_batches=1,
        precision=32,
        default_root_dir="output_checkpoints",
        enable_checkpointing=True,
        callbacks=[
            pl.callbacks.ModelCheckpoint(
                dirpath="output_checkpoints",
                save_top_k=-1,
                every_n_train_steps=None,
                save_weights_only=False,
                save_on_train_epoch_end=True,  # save at end of epoch
            ),
            pl.callbacks.LearningRateMonitor(logging_interval='epoch'),
        ],
        logger=[
            pl.loggers.TensorBoardLogger(save_dir="tensorboard", name="logs"),
            pl.loggers.WandbLogger(name=RUN_NAME, project="mqtl-classification"),
        ],
        strategy="auto",
    )

    plModule = MQTLClassifierModule(mainModel)

    try:
        trainer.fit(plModule, train_dataloaders=train_loader, val_dataloaders=val_loader)
    except Exception as x:
        timber.error(f"Error during training/evaluating: {x}")
    finally:
        try:
            save_fine_tuned_model(mainModel=mainModel)
        except Exception as x:
            timber.error(f"Error during fine-tuning: {x}")
        pass

    try:
        trainer.test(plModule, dataloaders=test_loader)
    except Exception as e:
        timber.error(f"Error during testing: {e}")

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
    # for some reason, the variables in the main function act like global variables in python
    # hence other functions get confused with the "global" variables. easiest solution, write everything
    # in another function (say, start(), or main()), and call it inside the main
    main()
    pass
