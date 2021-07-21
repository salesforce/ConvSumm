'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''


import re
import json
import string
import debugger

import torch
import torch.nn.functional as F

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

from transformers import BertTokenizer


class DialogueFunctionalPrediction:
    """
    for general usage of loading dialogues
    """

    def __init__(self, tokenizer, filepath='data/train.json', bert_path='bert-base-uncased', max_single_turn_length=30):
        
        with open(filepath) as f:
            data = json.load(f)
        
        self.data = []
        trig_count = 0
        for d in data:
            for i, text in enumerate(d["clean_dialog"]):
                local_d = {
                    "id": d["id"]+":turn{}".format(i),
                    "text": text,
                    "label": d["label"][i],
                }
                self.data.append(local_d)   
                
                if d["label"][i] == 1:
                    trig_count += 1
        
        self.max_single_turn_length = max_single_turn_length
        self.tokenizer = tokenizer #BertTokenizer.from_pretrained(bert_path) 
        
        print("[INFO] Positive Ratio: {:.4f}".format(trig_count/len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        
        _turn_plain = self.data[index]['text']

        PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')  # 0
        SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')  # 102
        CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')  # 101
        local_max_num_of_turns = None
        
        _turn = self.tokenizer.encode(_turn_plain, add_special_tokens=True)
        if len(_turn) > self.max_single_turn_length - 1:
            _turn = _turn[:self.max_single_turn_length - 1]
            _turn = _turn + [SEP]
        else:
            _turn += [PAD] * (self.max_single_turn_length - len(_turn))

        assert len(_turn) == self.max_single_turn_length

        return {
            'id': self.data[index]['id'],
            "document_plain": _turn_plain,
            'document': torch.LongTensor(_turn),
            'label': self.data[index]['label'] # torch.LongTensor(self.data[index]['label'])
       }
    
    
class DialogueSegmentation:
    """
    for general usage of loading dialogues
    """

    def __init__(self, tokenizer, filepath='data/train.json', bert_path='bert-base-uncased',
                 max_sequence_length=512, max_single_turn_length=30, max_num_of_turns=20):
        self.data = []
        with open(filepath) as f:
            data = json.load(f)
            #print(self.data[0])
        for d in data:
            if "segment_label" in d.keys():
                self.data.append(d)
            
        self.max_sequence_length = max_sequence_length
        self.max_single_turn_length = max_single_turn_length
        self.max_num_of_turns = max_num_of_turns
        self.tokenizer = tokenizer #BertTokenizer.from_pretrained(bert_path) # bert?

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        clean_dialog, segment_label = self.data[index]['clean_dialog'], self.data[index]['segment_label']

        PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')  # 0
        SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')  # 102
        CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')  # 101
        local_max_num_of_turns = None

        # Pad functional_dialogs with PAD
        input_ids = []
        clean_dialog_turns = []
        input_ids_len = 0
        for text in clean_dialog:
            _turn = self.tokenizer.encode(text, add_special_tokens=True)
            if len(_turn) > self.max_single_turn_length - 1:
                _turn = _turn[:self.max_single_turn_length - 1]
                _turn = _turn + [SEP]
            if len(clean_dialog_turns) < self.max_num_of_turns:
                input_ids.extend(_turn)
                clean_dialog_turns.append(_turn)
        
        if len(input_ids) < self.max_sequence_length:
            input_ids += [PAD] * (self.max_sequence_length - len(input_ids))
        else:
            # If longer than maximum length, then need to keep track of the left turns (smaller than max_num_of_turn)
            input_ids = input_ids[: self.max_sequence_length - 1]
            input_ids += [SEP]
            local_max_num_of_turns = input_ids.count(CLS)

        assert len(input_ids) == self.max_sequence_length

        if local_max_num_of_turns:
            segment_label = segment_label[:local_max_num_of_turns]
            clean_dialog_turns = clean_dialog_turns[:local_max_num_of_turns]

        # Pad module_index with -1
        if len(segment_label) < self.max_num_of_turns:
            segment_label += [-1] * (self.max_num_of_turns - len(segment_label))
        else:
            segment_label = segment_label[:self.max_num_of_turns]
        
        assert len([s for s in segment_label if s != -1]) == input_ids.count(CLS)
        
        return {'id':self.data[index]['id'],
                'document': torch.LongTensor(input_ids),
                'segment_label': torch.LongTensor(segment_label)}
    
    