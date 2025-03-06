import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class FCLayer(nn.Module):
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


class RBERT(BertPreTrainedModel):
    def __init__(self, config, args):
        super(RBERT, self).__init__(config)
        self.bert = BertModel(config=config)  # Load pretrained bert

        self.num_labels = config.num_labels
        self.is_binary = hasattr(args, 'binary_mode') and args.binary_mode

        self.cls_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)
        self.entity_fc_layer = FCLayer(config.hidden_size, config.hidden_size, args.dropout_rate)

        # For binary mode, we need a binary classifier (output dim = 2)
        if self.is_binary:
            self.label_classifier = FCLayer(
                config.hidden_size * 3,
                2,  # Binary classification: related or not
                args.dropout_rate,
                use_activation=False,
            )
        else:
            # Standard multi-class relation classifier
            self.label_classifier = FCLayer(
                config.hidden_size * 3,
                config.num_labels,
                args.dropout_rate,
                use_activation=False,
            )

    @staticmethod
    def entity_average(hidden_output, e_mask):
        """
        Average the entity hidden state vectors (H_i ~ H_j)
        :param hidden_output: [batch_size, j-i+1, dim]
        :param e_mask: [batch_size, max_seq_len]
                e.g. e_mask[0] == [0, 0, 0, 1, 1, 1, 0, 0, ... 0]
        :return: [batch_size, dim]
        """
        e_mask_unsqueeze = e_mask.unsqueeze(1)  # [b, 1, j-i+1]
        length_tensor = (e_mask != 0).sum(dim=1).unsqueeze(1)  # [batch_size, 1]

        # [b, 1, j-i+1] * [b, j-i+1, dim] = [b, 1, dim] -> [b, dim]
        sum_vector = torch.bmm(e_mask_unsqueeze.float(), hidden_output).squeeze(1)
        avg_vector = sum_vector.float() / length_tensor.float()  # broadcasting
        return avg_vector

    def forward(self, input_ids, attention_mask, token_type_ids, labels, e1_mask, e2_mask):
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # Average
        e1_h = self.entity_average(sequence_output, e1_mask)
        e2_h = self.entity_average(sequence_output, e2_mask)

        # Dropout -> tanh -> fc_layer (Share FC layer for e1 and e2)
        pooled_output = self.cls_fc_layer(pooled_output)
        e1_h = self.entity_fc_layer(e1_h)
        e2_h = self.entity_fc_layer(e2_h)

        # Concat -> fc_layer
        concat_h = torch.cat([pooled_output, e1_h, e2_h], dim=-1)
        logits = self.label_classifier(concat_h)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        # Softmax
        if labels is not None:
            if self.num_labels == 1:
                loss_fct = nn.MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels.view(-1))

            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class DuoClassifier:
    """
    A wrapper class that combines binary and relation classifiers for the duo-classifier approach.
    """

    def __init__(self, binary_model, relation_model, device, binary_threshold=0.5):
        self.binary_model = binary_model
        self.relation_model = relation_model
        self.device = device
        self.binary_threshold = binary_threshold

        # Set both models to evaluation mode
        self.binary_model.eval()
        self.relation_model.eval()

    def predict(self, inputs):
        """
        Perform two-stage prediction:
        1. Use binary model to determine if the sentence has a relation
        2. If relation exists, use relation model to classify the specific relation

        Args:
            inputs: Dictionary containing input tensors (input_ids, attention_mask, etc.)

        Returns:
            Dictionary containing prediction results
        """
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]
        e1_mask = inputs["e1_mask"]
        e2_mask = inputs["e2_mask"]

        # First stage: binary classification
        with torch.no_grad():
            binary_outputs = self.binary_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                labels=None,
                e1_mask=e1_mask,
                e2_mask=e2_mask
            )
            binary_logits = binary_outputs[0]
            binary_probs = torch.softmax(binary_logits, dim=1)
            has_relation = binary_probs[:, 1] > self.binary_threshold

        # Second stage: relation classification (only for samples predicted to have relations)
        relation_preds = torch.zeros(input_ids.size(0), device=self.device).long()

        # If any samples are predicted to have relations
        if has_relation.any():
            # Get indices of samples with relations
            relation_indices = has_relation.nonzero(as_tuple=True)[0]

            # Extract only the samples with relations for the second model
            rel_inputs = {
                "input_ids": input_ids[relation_indices],
                "attention_mask": attention_mask[relation_indices],
                "token_type_ids": token_type_ids[relation_indices],
                "e1_mask": e1_mask[relation_indices],
                "e2_mask": e2_mask[relation_indices],
                "labels": None
            }

            # Get relation predictions
            with torch.no_grad():
                rel_outputs = self.relation_model(**rel_inputs)
                rel_logits = rel_outputs[0]
                rel_preds = torch.argmax(rel_logits, dim=1)

            # Assign predictions to the original tensor
            for i, idx in enumerate(relation_indices):
                relation_preds[idx] = rel_preds[i]

        return {
            "binary_probs": binary_probs,
            "has_relation": has_relation,
            "relation_preds": relation_preds
        }