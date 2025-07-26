import torch.nn as nn
from torch.nn import MSELoss, CrossEntropyLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import BERT_START_DOCSTRING, BertPreTrainedModel, BertModel, \
    BERT_INPUTS_DOCSTRING

"""
to run DNABert 6, we need the original source code. copied the add_start_docstrings , add_start_docstrings_to_callable, 
and BertForLongSequenceClassification from this reference:
https://github.com/jerryji1993/DNABERT/blob/b6da04ec9a7d4e53efe5b33a6ce1a21c0e7ac413/src/transformers/modeling_bert.py#L1226

special thanks to the authors.
"""
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

        hidden_states = outputs.hidden_states
        attentions = outputs.attentions

        output = self.dropout(ht.squeeze(0).sum(dim=0))



        logits = self.classifier(output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        loss = None
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # return outputs  # (loss), logits, (hidden_states), (attentions)

        # slight modification to match my pipeline
        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=attentions
        )

"""
dnabert 6 end
"""
