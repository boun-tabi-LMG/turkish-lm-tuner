from torch import nn
from transformers import T5EncoderModel, T5Tokenizer
import torch
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.t5.modeling_t5 import T5PreTrainedModel, T5ClassificationHead
from transformers import T5Config
from transformers.modeling_outputs import (
    SequenceClassifierOutput
)

class T5ForClassification(T5PreTrainedModel):   # nn.Module
    def __init__(self, pretrained_model_name, config, num_labels, problem_type, dropout_prob=0.1):
        #super(T5ForSequenceClassification, self).__init__()
        super().__init__(config)

        self.encoder = T5EncoderModel.from_pretrained(pretrained_model_name)
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(self.encoder.config.d_model, num_labels)
        self.config = self.encoder.config
        self.config.num_labels = num_labels
        self.config.problem_type = problem_type
        self.config.dropout_prob = dropout_prob
        
        # Initialize weights and apply final processing
        self.post_init()

        self.model_parallel = False

    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
       
        # Compute mean representation
        if attention_mask is None:
            sequence_output = encoder_output.last_hidden_state.mean(dim=1)
        else:
            sum_repr = (encoder_output.last_hidden_state * attention_mask.unsqueeze(-1)).sum(dim=1)
            sequence_output = sum_repr / attention_mask.sum(dim=1, keepdim=True)

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None

        if labels is not None:
            if self.config.problem_type in ["single_label_classification"]:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                #loss = loss_fct(logits.view(-1, self.config.num_labels), labels)
                loss = loss_fct(logits.squeeze(), labels.squeeze())

        # print(loss)

        # if not return_dict:
        #     output = (logits,) + encoder_output
        #     return ((loss,) + output) if loss is not None else output

        return {
            "loss": loss,
            "logits": logits,
              # hidden_states=encoder_output.hidden_states,
              # attentions=encoder_output.attentions
        }

class T5EncoderForTokenClassification(T5ForClassification):
    def forward(self, input_ids, attention_mask=None, labels=None):
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = encoder_output.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        return {
            "loss": loss,
            "logits": logits,
        }
    
class T5EncoderForQuestionAnswering(T5ForClassification):
    def forward(self, input_ids, attention_mask=None, start_positions=None, end_positions=None):
        encoder_output = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = encoder_output.last_hidden_state
        logits = self.classifier(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        return {
            "loss": total_loss,
            "logits":  (start_logits, end_logits),
        }
