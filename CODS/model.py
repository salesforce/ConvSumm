
'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import BertModel
import debugger

global inf
inf = 1e10


class TypedModel(nn.Module):
    """
    Two types of dialogs: useful information(marked as 'function_dialogs' in data) / useless chitchats
    Predict these two types
    """

    def __init__(self, bert_path='bert-base-uncased', num_labels=2):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, evaluate=False):
        document = data['document'].to(self.device)
        label = data['label'].to(self.device)

        output = self.model(document)
        cls_vector = output[1]

        prediction_logits = self.classifier(cls_vector)
        loss = F.cross_entropy(prediction_logits, label)
        if evaluate:
            return loss, prediction_logits
        else:
            return loss


class IntentPredictor(nn.Module):
    """
    Given a complete functional dialogs: [CLS]+dialog1+[SEP]+dialog2+[SEP]+...+dialog_n + [SEP]
    predict module index (WHY = 0, WHAT = 1, WHERE = 2, WHEN = 3, CONFIRM = 4, ABSTAIN = -1)
    """

    def __init__(self, bert_path='bert-base-uncased', num_labels=6):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, evaluate=False):
        document = data['document'].to(self.device)
        module_index = data['module_index'].to(self.device)
        output = self.model(document)
        all_seq_hs = output[0]  # batch_size, seq_len, hd_dim

        sent_repr_mat = []
        turn_nums = [(item == 101).sum().cpu().item() for item in document]
        max_turn_num = max(turn_nums)
        for i in range(all_seq_hs.size(0)):
            sent_repr = all_seq_hs[i][document[i] == 101]  # [num_of_turns, hd_dim]
            sent_repr = torch.cat(
                [sent_repr, torch.zeros(max_turn_num - turn_nums[i], sent_repr.size(1)).to(self.device)], 0)
            sent_repr_mat.append(sent_repr)
            module_index[i][turn_nums[i]:] = -1
        sent_repr_mat = torch.stack(sent_repr_mat, 0)  # [batch_size, max_turn_num, hd_dim]

        module_index = module_index[:, :max_turn_num]
        prediction_logits = self.classifier(sent_repr_mat)
        loss = F.cross_entropy(prediction_logits.reshape(-1, 6), module_index.reshape(-1), ignore_index=-1,
                               reduction='mean')

        # For prediction_logits
        if evaluate:
            batch_size = prediction_logits.size(0)
            eval_prediction_logits = []
            for i in range(batch_size):
                eval_prediction_logits.append(prediction_logits[i][:turn_nums[i], :])
            eval_prediction_logits = torch.cat(eval_prediction_logits, 0)
            return loss, eval_prediction_logits, module_index
        else:
            return loss


class ModuleFunction(nn.Module):
    """
    Module Functions of WHY, WHAT, WHERE, WHEN, CONFIRM, ABSTAIN
    """

    def __init__(self, args, bert_path='bert-base-uncased'):
        super().__init__()
        self.encoder = BertModel.from_pretrained(bert_path)
        hidden_dim = self.encoder.config.hidden_size
        
        # Need an empty function to plug in pad position
        self.function_names = ['ABSTAIN', 'WHY', 'WHAT', 'WHERE', 'WHEN', 'CONFIRM', 'EMPTY']
        self.index_to_function_name = {0: 'ABSTAIN', 1: 'WHY', 2: 'WHAT', 3: 'WHERE', 4: 'WHEN', 5: 'CONFIRM',
                                       -1: 'EMPTY'}

        self.module_net = nn.ParameterDict()
        for function_name in self.function_names:
            self.module_net[function_name] = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        self.span_module_net = nn.ParameterDict()
        for function_name in self.function_names:
            self.span_module_net[function_name] = nn.Parameter(torch.randn(hidden_dim, 2))  # max_seq_len
        
        self.dropout_layer = nn.Dropout(args.dropout)
        
        self.loss = nn.CrossEntropyLoss(ignore_index=-1)

    @property
    def device(self):
        return next(self.parameters()).device

    def forward_one_step_span_pred(self, context_hidden, sequence_hidden, sequence_mask, key_phrase_label, module_index,
                                   use_dependency, evaluate):
        # weights: for QA heads
        weights = []
        for index in module_index:
            function_name = self.index_to_function_name[index.tolist()]
            function_net = self.span_module_net[function_name]
            weights.append(function_net)
        weights = torch.stack(weights, dim=0)
        assert weights.shape[0] == len(module_index)
        # bilinear_w: for combining context with sequence
        bilinear_w = []
        for index in module_index:
            function_name = self.index_to_function_name[index.tolist()]
            function_net = self.module_net[function_name]
            bilinear_w.append(function_net)
        bilinear_w = torch.stack(bilinear_w, dim=0)
        assert bilinear_w.shape[0] == len(module_index)
        # -1 in module_index means there is on turn
        module_index_mask = (module_index != -1)

        context_sequence_hidden = torch.einsum('bh,bhz,blz->blz', context_hidden, bilinear_w, sequence_hidden)
        context_sequence_hidden = self.dropout_layer(context_sequence_hidden)
        logits = torch.einsum('blh,bhz->blz', context_sequence_hidden, weights)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        start_positions, end_positions = key_phrase_label[:, 0], key_phrase_label[:, 1]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            loss_fct = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss+end_loss) / 2

        total_loss = total_loss * module_index_mask  # mask out module index = -1

        if not evaluate:
            return total_loss, None
        else:
            return total_loss, None, (start_logits, end_logits)

    def forward(self, data, use_dependency=False, evaluate=False):
        document = data['document'].to(self.device)

        sequence_mask = (document > 0).bool()
        # [bsz, #(turns), max_single_turn_length] span [bsz, #(turns), (#start_in_doc, #end_in_doc)]
        turns = data['turns'].to(self.device)
        key_phrase_target = data['key_phrase_label_documents'].to(self.device)  # [bsz, #(turns), max_seq_length]
        key_phrase_span = data['key_phrase_span'].to(self.device)
        module_index = data['module_index'].to(self.device)  # [bsz, #(turns)]
        assert turns.shape[1] == key_phrase_target.shape[1]
        sequence_output, _ = self.encoder(document)  # [bsz, seq_len, hidden_dim]
        # the padding token is -1
        max_turns = (turns[:, :, 0] != -1).bool().sum(-1).max().item()
        turns_mask = turns[:, :, 0] != -1  # [bsz, #(turns)]
        turns_mask = turns_mask[:, :max_turns]  # [bsz, max_turns]

        loss_tracker = []
        predict_logit_tracker = []
        for step in range(max_turns):
            # start / end: [bsz]
            start, end = turns[:, step, 0], turns[:, step, 1]

            if step > 0 and use_dependency:
                context_hidden_step = reweighted_context_hidden_step
            else:
                context_hidden_step = sequence_output[torch.arange(start.size(0)), start]  # [bsz, hidden]

            key_phrase_target_step = key_phrase_span[:, step]
            module_index_step = module_index[:, step]

            # sequence_output can be updated
            if not evaluate:
                loss_step, reweighted_context_hidden_step = self.forward_one_step_span_pred(context_hidden_step, sequence_output,
                                                                                  sequence_mask, key_phrase_target_step,
                                                                                  module_index_step, use_dependency,
                                                                                  evaluate)  # loss_step.shape = [bsz]
            else:
                loss_step, reweighted_context_hidden_step, logit_step = self.forward_one_step_span_pred(context_hidden_step,
                                                                                              sequence_output,
                                                                                              sequence_mask,
                                                                                              key_phrase_target_step,
                                                                                              module_index_step,
                                                                                              use_dependency,
                                                                                              evaluate)  # loss_step.shape = [bsz]
                predict_logit_tracker.append(logit_step)
            loss_tracker.append(loss_step)

        loss = torch.stack(loss_tracker, dim=-1)
        loss = (loss * turns_mask).sum() / turns_mask.sum()
        if not evaluate:
            return loss
        else:
            return loss, predict_logit_tracker


class SegmentPredictor(nn.Module):

    def __init__(self, bert_path='bert-base-uncased', num_labels=2):
        super().__init__()
        self.model = BertModel.from_pretrained(bert_path)
        self.classifier = nn.Linear(self.model.config.hidden_size, 2)
        self.num_labels = num_labels

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, data, evaluate=False):
        document = data['document'].to(self.device)
        segment_label = data['segment_label'].to(self.device)
        output = self.model(document)
        all_seq_hs = output[0]  # batch_size, seq_len, hd_dim

        sent_repr_mat = []
        turn_nums = [(item == 101).sum().cpu().item() for item in document]
        max_turn_num = max(turn_nums)
        for i in range(all_seq_hs.size(0)):
            sent_repr = all_seq_hs[i][document[i] == 101]  # [num_of_turns, hd_dim]
            sent_repr = torch.cat(
                [sent_repr, torch.zeros(max_turn_num - turn_nums[i], sent_repr.size(1)).to(self.device)], 0)
            sent_repr_mat.append(sent_repr)
            segment_label[i][turn_nums[i]:] = -1
        sent_repr_mat = torch.stack(sent_repr_mat, 0)  # [batch_size, max_turn_num, hd_dim]

        segment_label = segment_label[:, :max_turn_num]
        prediction_logits = self.classifier(sent_repr_mat)
        loss = F.cross_entropy(prediction_logits.reshape(-1, self.num_labels), segment_label.reshape(-1), ignore_index=-1,
                               reduction='mean')

        # For prediction_logits
        if evaluate:
            batch_size = prediction_logits.size(0)
            eval_prediction_logits = []
            for i in range(batch_size):
                eval_prediction_logits.append(prediction_logits[i][:turn_nums[i], :])
            # eval_prediction_logits = torch.cat(eval_prediction_logits, 0)
            return loss, eval_prediction_logits, segment_label
        else:
            return loss
        