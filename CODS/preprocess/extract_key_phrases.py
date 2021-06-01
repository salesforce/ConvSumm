'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import json
import time
from benepar.spacy_plugin import BeneparComponent
import spacy
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from tqdm import tqdm
import benepar, nltk
benepar.download('benepar_en3')


"""
Given clean dialogues, for each functional turn, we extract the key phrases which overlap with summaries.
e.g.
'summary': "Megan needn't buy milk and cereals. They're in the drawer next to the fridge."
'dialog': 'Megan: hm , sure , i can do that. but did you check in the drawer next to the fridge ?'
'key_phrases': ['check in the drawer next to the fridge']
"""

nlp = spacy.load('en')
nlp.add_pipe(BeneparComponent('benepar_en3'))

stemmer = PorterStemmer()
english_stopwords = list(stopwords.words('english'))



def longest_common_subsequence(text1: list, text2: list):
    n, m = len(text1), len(text2)
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    path = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n):
        for j in range(m):
            stem_text1 = stemmer.stem(text1[i])
            stem_text2 = stemmer.stem(text2[j])
            if stem_text1 == stem_text2 and stem_text2 not in english_stopwords and stem_text2 not in string.punctuation:
                dp[i + 1][j + 1] = dp[i][j] + 1
                path[i + 1][j + 1] = 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
                if dp[i + 1][j + 1] == dp[i][j + 1]:
                    path[i + 1][j + 1] = 2

    word_indexes = []

    def backtrack(i, j):
        if i == 0 or j == 0:
            return
        if path[i][j] == 1:
            backtrack(i - 1, j - 1)
            word_indexes.append(j - 1)
        elif path[i][j] == 2:
            backtrack(i - 1, j)
        else:
            backtrack(i, j - 1)
    backtrack(n, m)
    return dp[-1][-1], word_indexes


def process_dialogue(dialogues_json_path):
    with open(dialogues_json_path, 'r') as fin:
        dialogues = json.load(fin)

    for dialogue in tqdm(dialogues):
        tokenized_summary = [str(token.text) for token in nlp(dialogue['summary'])]
        dialogue['key_phrases'] = []

        for sent in dialogue['function_dialogs']:
            doc = nlp(sent.split(": ")[-1])
            # print(doc) # hey , do you have betty's number ?
            tokenized_sent = [str(token.text) for token in doc]
            # print(tokenized_sent) # ['hey', ',', 'do', 'you', 'have', 'betty', "'s", 'number', '?']
            # dialogue['tokenized_function_dialogs'].append(tokenized_sent)
            parsed_sents = list(doc.sents)
            # print(parsed_sents) # [hey , do you have betty's number ?]
            phrases = []
            for p_sent in parsed_sents:
                phrases.extend([str(i) for i in list(p_sent._.children)])
            # print(phrases) # ['hey', ',', 'do', 'you', "have betty's number", '?']
            if phrases:
                word2phrase = dict()
                j = 0
                for i in range(len(tokenized_sent)):
                    if tokenized_sent[i] in phrases[j]:
                        word2phrase[i] = j
                    else:
                        while tokenized_sent[i] not in phrases[j]:
                            j += 1
                            if j == len(phrases):
                                break
                        if j == len(phrases):
                            for k in range(i, len(tokenized_sent)):
                                word2phrase[k] = j - 1
                            break
                        else:
                            word2phrase[i] = j

                # phrase2word = dict()
                # phrase2word[0] = [0, len(phrases[0].split()) - 1]
                # for i in range(1, len(phrases)):
                #     phrase2word[i] = [phrase2word[i - 1][1] + 1, phrase2word[i - 1][1] + len(phrases[i].split())]

                lcs_len, word_indexes = longest_common_subsequence(tokenized_summary, tokenized_sent)
                if lcs_len == 0:
                    dialogue['key_phrases'].append([])
                else:
                    # labels = [0] * len(tokenized_sent)
                    phrase_set = set()
                    for i in word_indexes:
                        # labels[i] = 1
                        phrase_set.add(word2phrase[i])
                    phrase_res = []
                    for phrase_id in phrase_set:
                        if phrases[phrase_id] in english_stopwords or phrases[phrase_id] in string.punctuation:
                            continue
                        phrase_res.append(phrases[phrase_id])
                    dialogue['key_phrases'].append(phrase_res)
            else:
                lcs_len, word_indexes = longest_common_subsequence(tokenized_summary, tokenized_sent)
                if lcs_len == 0:
                    dialogue['key_phrases'].append([])
                else:
                    phrase_res = []
                    for i in word_indexes:
                        # labels[i] = 1
                        phrase_res.append(tokenized_sent[i])
                    dialogue['key_phrases'].append(phrase_res)

        # print(dialogue['tokenized_function_dialogs'])
        # print(dialogue['key_phrases_labels'])
        # print(dialogue['function_dialogs'])
        # print(dialogue['key_phrases'])

    with open(dialogues_json_path, "w") as fo:
        json.dump(dialogues, fo, indent=4)


import time

st = time.time()
process_dialogue("../SAMsum/clean_data/test.json")
process_dialogue("../SAMsum/clean_data/eval.json")
process_dialogue("../SAMsum/clean_data/train.json")
print(time.time() - st)
