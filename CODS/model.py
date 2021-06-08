
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
        