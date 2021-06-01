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


class Dialogue:
    """
    for general usage of loading dialogues
    """

    def __init__(self, tokenizer, filepath='data/train.json', bert_path='bert-base-uncased',
                 max_sequence_length=128, max_single_turn_length=30, max_num_of_turns=20):
        self.data = []
        with open(filepath) as f:
            self.data = json.load(f)
            print(self.data[0])
            
        self.max_sequence_length = max_sequence_length
        self.max_single_turn_length = max_single_turn_length
        self.max_num_of_turns = max_num_of_turns
        self.tokenizer = tokenizer #BertTokenizer.from_pretrained(bert_path) # bert?

    def __len__(self):
        return len(self.data)

    @staticmethod
    def _convert_module_index(module_index):
        """

        :param module_index: WHY = 0; WHAT = 1; WHERE = 2; WHEN = 3; CONFIRM = 4; ABSTAIN = -1
        :return: (each index + 1): ABSTAIN = 0; WHY = 1; WHAT = 2; WHERE = 3; WHEN = 4; CONFIRM = 5
                 and postphone one pos
        """
        return [0] + [item + 1 for item in module_index][:-1]

    @staticmethod
    def _locate_sublist(sublist, parent):
        cursor = 0
        for i, ele in enumerate(parent):
            if ele == sublist[0]:
                if parent[i: i + len(sublist)] == sublist:
                    # start_index = cursor, end_index = cursor+len(sublist)-1
                    cursor = i
                    break
        return cursor, cursor + len(sublist) - 1

    def label_keyphrase_in_turns(self, function_dialog_turns, key_phrases):
        key_phrase_label = []
        key_phrase_length = []
        for turn, key_phrase in zip(function_dialog_turns, key_phrases):
            if not key_phrase:
                key_phrase_label.append([])
                key_phrase_length.append(0)
            else:
                sub_label = []
                kl = 0
                cursor_idx = 0
                for key_phrase_item in key_phrase:
                    key_phrase_tok = self.tokenizer.encode(key_phrase_item, add_special_tokens=False)
                    start_index, end_index = self._locate_sublist(key_phrase_tok, turn)
                    if start_index < cursor_idx:
                        continue
                    else:
                        sub_label.extend([start_index, end_index])
                        kl += (end_index - start_index + 1)
                        cursor_idx = end_index
                key_phrase_label.append(sub_label)
                key_phrase_length.append(kl)
        return key_phrase_label, key_phrase_length

    def __getitem__(self, index):
        function_dialogs, module_index = self.data[index]['function_dialogs'], self.data[index]['module_index']
        key_phrases = self.data[index]['key_phrases']
        module_index = self._convert_module_index(module_index)

        PAD = self.tokenizer.convert_tokens_to_ids('[PAD]')  # 0
        SEP = self.tokenizer.convert_tokens_to_ids('[SEP]')  # 102
        CLS = self.tokenizer.convert_tokens_to_ids('[CLS]')  # 101
        local_max_num_of_turns = None

        # Pad functional_dialogs with PAD
        input_ids = []
        function_dialog_turns = []
        input_ids_len = 0
        for text in function_dialogs:
            _turn = self.tokenizer.encode(text, add_special_tokens=True)
            if len(_turn) > self.max_single_turn_length - 1:
                _turn = _turn[:self.max_single_turn_length - 1]
                _turn = _turn + [SEP]
            input_ids.extend(_turn)
            function_dialog_turns.append(_turn)
        if len(input_ids) < self.max_sequence_length:
            input_ids += [PAD] * (self.max_sequence_length - len(input_ids))
        else:
            # If longer than maximum length, then need to keep track of the left turns (smaller than max_num_of_turn)
            input_ids = input_ids[: self.max_sequence_length - 1]
            input_ids += [SEP]
            local_max_num_of_turns = input_ids.count(CLS)

        assert len(input_ids) == self.max_sequence_length

        if not local_max_num_of_turns:
            module_index = module_index[:local_max_num_of_turns]
            function_dialog_turns = function_dialog_turns[:local_max_num_of_turns]
            key_phrases = key_phrases[:local_max_num_of_turns]

        # Pad module_index with -1
        if len(module_index) < self.max_num_of_turns:
            module_index += [-1] * (self.max_num_of_turns - len(module_index))
        else:
            module_index = module_index[:self.max_num_of_turns]

        # locate key phase for each turn, and process the last turn
        key_phrase_label, key_phrase_length = self.label_keyphrase_in_turns(function_dialog_turns, key_phrases)

        key_phrase_label_documents = []
        sent_start_index = [i for i, val in enumerate(input_ids) if val == CLS]
        key_phrase_label = key_phrase_label[:local_max_num_of_turns]

        for i, kp in enumerate(key_phrase_label):
            key_phrase_label_document = [0] * len(input_ids)
            if kp:
                start_index = sent_start_index[i]
                l = key_phrase_length[i]
                for idx in range(0, len(kp), 2):
                    if start_index + kp[idx + 1] < self.max_sequence_length:
                        key_phrase_label_document[start_index + kp[idx]: start_index + kp[idx + 1] + 1] = \
                            [float(1 / l)] * (kp[idx + 1] - kp[idx] + 1)
            assert len(key_phrase_label_document) == self.max_sequence_length
            key_phrase_label_documents.append(key_phrase_label_document)

        key_phrase_span = []
        for i, kp in enumerate(key_phrase_label):
            start_index = sent_start_index[i]
            if not kp:
                key_phrase_span.append([-1, -1])
            else:
                if start_index + kp[-1] < self.max_sequence_length:
                    key_phrase_span.append([start_index+kp[0], start_index+kp[-1]])
                else:
                    key_phrase_span.append([-1, -1])

        # find start_index & end_index for each turn in input_ids
        sent_end_index = [i for i, val in enumerate(input_ids) if val == SEP]
        turns_index = []
        for si, ei in zip(sent_start_index, sent_end_index):
            turns_index.append([si, ei])

        # Ensure the number of turns_index is padded/cut to self.max_num_of_turns
        if len(turns_index) < self.max_num_of_turns:
            turns_index += [[-1] * 2] * (self.max_num_of_turns - len(turns_index))
        else:
            turns_index = turns_index[:self.max_num_of_turns]

        # key phrase matching for each dialog turn
        if len(key_phrase_label_documents) < self.max_num_of_turns:
            key_phrase_label_documents += [[0.0] * self.max_sequence_length] * (
                        self.max_num_of_turns - len(key_phrase_label_documents))

        # key phrase span matching for each dialog turn
        if len(key_phrase_span) < self.max_num_of_turns:
            key_phrase_span += [[-1, -1]] * (self.max_num_of_turns - len(key_phrase_span))

        return {'id':self.data[index]['id'],
                'document': torch.LongTensor(input_ids),
                'module_index': torch.LongTensor(module_index),
                'turns': torch.LongTensor(turns_index),
                'key_phrase_label_documents': torch.FloatTensor(key_phrase_label_documents),
                'key_phrase_span': torch.LongTensor(key_phrase_span)}

    

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
    
    