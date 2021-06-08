'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import re
import os
import json
import string
from tqdm import tqdm
import emoji
import sys
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.tokenize import TweetTokenizer

nltk.download('stopwords')
nltk.download('punkt')
sys.path.append("..")

from twitter_handling import tokenize
from snorkel_label import get_snorkel_label

import debugger


"""
Preprocess the raw data and save it for later use:
1. generate 'clean_dialogs'
2. generate 'label': Label each dialog as 0 and 1, 1 means have meaningful overlaps with summary
3. generate 'function_dialogs': label the neighbor (left and right) dialog of '1' also as 1
3. generate 'module_index':  use Snorkel to label each function_dialog with module index
"""

def clean_data(raw_dialogs):
    """
    :param fname:
    :return: add one key 'clean_dialog':
    1. combine multiple consecutive turns from one speaker into single turn
    2. follow twitter preprocess for emojis e.t.c.
    """

    for line in raw_dialogs:
        # skip empty dialogue data
        if line['dialogue'] == '' or line['summary'] == '':
            raw_dialogs.remove(line)

    tknzr = TweetTokenizer()
    for i in tqdm(range(len(raw_dialogs))):
        line = raw_dialogs[i]
        if '\r\n' in line['dialogue']:
            dialog_sents = line['dialogue'].rstrip('\n').split('\r\n')
        else:
            dialog_sents = line['dialogue'].rstrip('\n').split('\n')
        dialogs = []
        # speaker = dialog_sents[0].split(": ")[0]
        speaker = None
        for sent in dialog_sents:
            try:
                cur_speaker = sent.split(": ")[0]
                cur_dialog = sent.split(": ")[1]
            except:
                cur_speaker = sent.split(":")[0]
                cur_dialog = sent.split(":")[1]
            # Preprpcess cur_dialog to remove emojis
            cur_dialog = tokenize(cur_dialog)
            # remove <file_photo> <file_gif> and all other annotations in <*>
            tok_dialog = tknzr.tokenize(cur_dialog)
            tok_dialog = [item for item in tok_dialog if item not in emoji.UNICODE_EMOJI]
            cur_dialog = " ".join([ele for ele in tok_dialog if '<' not in ele and '>' not in ele])
            if cur_dialog == "":
                continue
            if cur_speaker == speaker:
                dialogs[-1] += '. ' + cur_dialog
            else:
                dialogs.append(cur_speaker + ": " + cur_dialog)
            speaker = cur_speaker

        raw_dialogs[i]['clean_dialog'] = dialogs

    return raw_dialogs


def load_dialogue(dialogues_json_path):
    english_stopwords = list(stopwords.words('english'))

    with open(dialogues_json_path, 'r') as fin:
        dialogues = json.load(fin)

    # Clean data, remove empty lines and emojis
    dialogues = clean_data(dialogues)

    stemmer = PorterStemmer()
    for i, dialogue in enumerate(dialogues):
        # Tokenize summary and filter punctuation
        summary = word_tokenize(dialogue['summary'])
        summary = [word.lower() for word in summary if word not in string.punctuation]
        summary = [stemmer.stem(tok) for tok in summary]
        dlg = dialogue['clean_dialog']

        label = []
        for k in range(len(dlg)):
            # filter name and punctuation
            dlg_text = dlg[k].split()[1:]
            dlg_text = word_tokenize(" ".join(dlg_text))
            dlg_text = [word.lower() for word in dlg_text if word not in string.punctuation]
            dlg_text = [stemmer.stem(tok) for tok in dlg_text]
            res = set(summary).intersection(set(dlg_text))
            if res == set() or (len(res) == 1 and list(res)[0] in english_stopwords):
                label.append(0)
            else:
                label.append(1)

        # Sometimes the labels are all 0, we'll change them into 1 instead
        if set(label) == {0}:
            label = [1] * len(dlg)

        dialogues[i]['label'] = label

    # get_function_dialogs
    dialogues = get_function_dialogs(dialogues)
    return dialogues


def get_full_labels(label):
    """consider the previous and next turns of turn labeling 1"""
    new_label = [0] * len(label)
    for i in range(len(label)):
        if label[i] == 1:
            new_label[i] = 1
            if i-1 > -1:
                new_label[i - 1] = 1
            if i + 1 < len(label):
                new_label[i + 1] = 1
    return new_label


def get_function_dialogs(dialogs):
    for i, dialog in enumerate(dialogs):
        dialog_text = []
        label = dialog['label']
        new_label = get_full_labels(label)
        for k in range(len(new_label)):
            if new_label[k] == 1:
                dialog_text.append(dialog['clean_dialog'][k])
        dialogs[i]['function_dialogs'] = dialog_text
    return dialogs


def dump_file(json_obj, output_path):
    with open(output_path, 'w') as json_file:
        json.dump(json_obj, json_file)


def main():
    data_dir = '../SAMsum/'
    dump_data_dir = '../SAMsum/clean_data/'
    
    if not os.path.exists(dump_data_dir):
        os.makedirs(dump_data_dir)
    
    # get preprocessed data
    train_dataset = load_dialogue(os.path.join(data_dir, "train.json"))
    eval_dataset = load_dialogue(os.path.join(data_dir, "val.json"))
    test_dataset = load_dialogue(os.path.join(data_dir, "test.json"))

    # Snorkel labeling
    snorkel_train_data, snorkel_eval_data, snorkel_test_data = get_snorkel_label(train_dataset, eval_dataset,
                                                                                 test_dataset)

    # Dump datasets
    dump_file(snorkel_train_data, os.path.join(dump_data_dir, "train.json"))
    dump_file(snorkel_eval_data, os.path.join(dump_data_dir, "eval.json"))
    dump_file(snorkel_test_data, os.path.join(dump_data_dir, "test.json"))


if __name__ == '__main__':
    main()


