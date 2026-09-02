"""A token-classification head with two parallel linear classifiers
(negation + speculation) on top of any HuggingFace `AutoModel` encoder.

This replaces the notebook's vendored MultiHeadBertForTokenClassification /
MultiHeadRobertaForTokenClassification / MultiHeadXLNetForTokenClassification
classes with a single architecture-agnostic implementation.
"""
from torch import nn
from transformers import AutoConfig, AutoModel


class MultiHeadTokenClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super().__init__()
        config = AutoConfig.from_pretrained(model_name)
        self.num_labels = num_labels
        self.encoder = AutoModel.from_pretrained(model_name, config=config)

        hidden_size = getattr(config, "hidden_size", None) or config.d_model
        dropout_prob = getattr(config, "hidden_dropout_prob", None)
        if dropout_prob is None:
            dropout_prob = getattr(config, "dropout", 0.1)

        self.dropout = nn.Dropout(dropout_prob)
        self.classifier_neg = nn.Linear(hidden_size, num_labels)
        self.classifier_spec = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.encoder(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = self.dropout(outputs[0])
        logits_neg = self.classifier_neg(sequence_output)
        logits_spec = self.classifier_spec(sequence_output)
        return (logits_neg, logits_spec),
