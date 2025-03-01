import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class FCLayer(nn.Module):
    """Fully Connected Layer with optional activation"""
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dropout_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.dropout(x)
        if self.use_activation:
            x = self.tanh(x)
        return self.linear(x)


class BERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained BERT
        self.num_labels = config.num_labels

        # Fully connected layer for classification
        self.classifier = FCLayer(config.hidden_size, config.num_labels, args.dropout_rate, use_activation=False)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        """
        Forward pass for standard BERT classification.
        Uses only [CLS] token representation for classification.
        """
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)

        pooled_output = outputs[1]  # Use [CLS] token representation

        # Classification
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # Add hidden states and attention if available

        # Compute loss if labels are provided
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)
