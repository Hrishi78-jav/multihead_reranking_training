import torch
from transformers import AutoModel, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class Multi_Head_Reranker_Model(BertPreTrainedModel):
    def __init__(self, config, model_name, num_head_labels=3):
        super().__init__(config)
        self.num_head_labels = num_head_labels
        self.seq_classifier = torch.nn.Linear(self.config.hidden_size, 1)
        self.foundation_model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)
        self.classifier_head1 = torch.nn.Linear(self.config.hidden_size, num_head_labels)
        self.classifier_head2 = torch.nn.Linear(self.config.hidden_size, num_head_labels)
        self.classifier_head3 = torch.nn.Linear(self.config.hidden_size, num_head_labels)
        self.classifier_head4 = torch.nn.Linear(self.config.hidden_size, num_head_labels)
        self.classifier_head5 = torch.nn.Linear(self.config.hidden_size, num_head_labels)
        self.init_weights()

    def head_output(self, logits, labels=None):
        loss_fct = torch.nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_head_labels), labels.view(-1))
        else:
            loss = None
        output = SequenceClassifierOutput(loss=loss, logits=logits)
        return output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            head1_labels=None,
            head2_labels=None,
            head3_labels=None,
            head4_labels=None,
            head5_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        foundation_output = self.foundation_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = foundation_output[1]
        pooled_output = self.dropout(pooled_output)

        sequence_classifier_output = self.seq_classifier(pooled_output)

        logits1 = self.classifier_head1(pooled_output)
        logits2 = self.classifier_head2(pooled_output)
        logits3 = self.classifier_head3(pooled_output)
        logits4 = self.classifier_head4(pooled_output)
        logits5 = self.classifier_head5(pooled_output)

        head1_output = self.head_output(logits1, head1_labels)
        head2_output = self.head_output(logits2, head2_labels)
        head3_output = self.head_output(logits3, head3_labels)
        head4_output = self.head_output(logits4, head4_labels)
        head5_output = self.head_output(logits5, head5_labels)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            main_loss = loss_fct(sequence_classifier_output.view(-1), labels.float())
            sequence_classifier_output = SequenceClassifierOutput(loss=main_loss, logits=sequence_classifier_output)

        return head1_output, head2_output, head3_output, head4_output, head5_output, sequence_classifier_output


class Multi_Head_Reranker_Model2(BertPreTrainedModel):
    def __init__(self, config, model_name, num_head_labels=3):
        super().__init__(config)
        self.num_head_labels = num_head_labels
        self.seq_classifier = torch.nn.Linear(int(self.config.hidden_size / 2), 1)
        self.seq_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.foundation_model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)

        self.classifier_head1 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head1_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.classifier_head2 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head2_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.classifier_head3 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head3_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.classifier_head4 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head4_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.classifier_head5 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head5_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.init_weights()

    def head_output(self, logits, labels=None):
        loss_fct = torch.nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_head_labels), labels.view(-1))
        else:
            loss = None
        output = SequenceClassifierOutput(loss=loss, logits=logits)
        return output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            head1_labels=None,
            head2_labels=None,
            head3_labels=None,
            head4_labels=None,
            head5_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        foundation_output = self.foundation_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = foundation_output[1]
        pooled_output = self.dropout(pooled_output)

        sequence_classifier_output = self.seq_classifier(self.seq_hidden_layer(pooled_output))

        logits1 = self.classifier_head1(self.head1_hidden_layer(pooled_output))
        logits2 = self.classifier_head2(self.head2_hidden_layer(pooled_output))
        logits3 = self.classifier_head3(self.head3_hidden_layer(pooled_output))
        logits4 = self.classifier_head4(self.head4_hidden_layer(pooled_output))
        logits5 = self.classifier_head5(self.head5_hidden_layer(pooled_output))

        head1_output = self.head_output(logits1, head1_labels)
        head2_output = self.head_output(logits2, head2_labels)
        head3_output = self.head_output(logits3, head3_labels)
        head4_output = self.head_output(logits4, head4_labels)
        head5_output = self.head_output(logits5, head5_labels)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            main_loss = loss_fct(sequence_classifier_output.view(-1), labels.float())
            sequence_classifier_output = SequenceClassifierOutput(loss=main_loss, logits=sequence_classifier_output)

        return head1_output, head2_output, head3_output, head4_output, head5_output, sequence_classifier_output


class Multi_Head_Reranker_Model3(BertPreTrainedModel):
    def __init__(self, config, model_name, num_head_labels=3):
        super().__init__(config)
        self.num_head_labels = num_head_labels
        self.seq_classifier = torch.nn.Linear(int(self.config.hidden_size / 2), 1)
        self.seq_hidden_layer = torch.nn.Linear(int(3.5 * self.config.hidden_size), int(self.config.hidden_size / 2))
        self.foundation_model = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = torch.nn.Dropout(self.config.hidden_dropout_prob)

        self.classifier_head1 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head1_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.classifier_head2 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head2_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.classifier_head3 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head3_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.classifier_head4 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head4_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.classifier_head5 = torch.nn.Linear(int(self.config.hidden_size / 2), num_head_labels)
        self.head5_hidden_layer = torch.nn.Linear(self.config.hidden_size, int(self.config.hidden_size / 2))
        self.init_weights()

    def head_output(self, logits, labels=None):
        loss_fct = torch.nn.CrossEntropyLoss()
        if labels is not None:
            loss = loss_fct(logits.view(-1, self.num_head_labels), labels.view(-1))
        else:
            loss = None
        output = SequenceClassifierOutput(loss=loss, logits=logits)
        return output

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            head1_labels=None,
            head2_labels=None,
            head3_labels=None,
            head4_labels=None,
            head5_labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        foundation_output = self.foundation_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = foundation_output[1]
        pooled_output = self.dropout(pooled_output)

        head1_hidden_layer = self.head1_hidden_layer(pooled_output)
        logits1 = self.classifier_head1(head1_hidden_layer)
        head2_hidden_layer = self.head1_hidden_layer(pooled_output)
        logits2 = self.classifier_head2(head2_hidden_layer)
        head3_hidden_layer = self.head1_hidden_layer(pooled_output)
        logits3 = self.classifier_head3(head3_hidden_layer)
        head4_hidden_layer = self.head1_hidden_layer(pooled_output)
        logits4 = self.classifier_head4(head4_hidden_layer)
        head5_hidden_layer = self.head1_hidden_layer(pooled_output)
        logits5 = self.classifier_head5(head5_hidden_layer)

        hidden_layers_combined = torch.concat(
            [pooled_output, head1_hidden_layer, head2_hidden_layer, head3_hidden_layer, head4_hidden_layer,
             head5_hidden_layer], dim=1)
        sequence_classifier_output = self.seq_classifier(self.seq_hidden_layer(hidden_layers_combined))

        head1_output = self.head_output(logits1, head1_labels)
        head2_output = self.head_output(logits2, head2_labels)
        head3_output = self.head_output(logits3, head3_labels)
        head4_output = self.head_output(logits4, head4_labels)
        head5_output = self.head_output(logits5, head5_labels)

        if labels is not None:
            loss_fct = torch.nn.BCEWithLogitsLoss()
            main_loss = loss_fct(sequence_classifier_output.view(-1), labels.float())
            sequence_classifier_output = SequenceClassifierOutput(loss=main_loss, logits=sequence_classifier_output)

        return head1_output, head2_output, head3_output, head4_output, head5_output, sequence_classifier_output
