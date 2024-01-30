from torch import nn
from transformers import T5EncoderModel
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.models.t5.modeling_t5 import T5PreTrainedModel


class T5ForClassification(T5PreTrainedModel):   # nn.Module
    """
    T5 encoder adapted for classification
    Args:
        pretrained_model_name: Pretrained model name or path
        config: T5Config
        num_labels: Number of labels
        problem_type: Problem type. It can be either 'single_label_classification', 'multi_label_classification', 'token_classification' or 'regression'
        dropout_prob: Dropout probability
    """
    def __init__(self, pretrained_model_name, config, num_labels, problem_type, dropout_prob=0.1):
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
        if self.config.problem_type == "token_classification":
            sequence_output = encoder_output.last_hidden_state
        else:
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
            if self.config.problem_type in ["single_label_classification", "token_classification"]:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
            elif self.config.problem_type == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.squeeze(), labels.squeeze())

        return {
            "loss": loss,
            "logits": logits,
        }
