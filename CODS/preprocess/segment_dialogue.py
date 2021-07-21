'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import rouge
import time
import json
import nltk
import os
from tqdm import tqdm
import debugger
nltk.download('punkt')

sep_token = '</s>'
    
rouge_evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                        max_n=4,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=False,
                        apply_best=True,
                        alpha=0.5, # Default F1_score
                        weight_factor=1.2,
                        stemming=True)


def calculate_similarity(pred, summary):
    try:
        rouge_score = rouge_evaluator.get_scores([pred], [summary])
    except:
        print("pred", pred)
        print("summary", summary)
        return 0
    return rouge_score["rouge-1"]['f']


def segment_dial(dialog, summary):
    sum_list = [s.strip()+"." for s in summary.strip().split(".") if (s.strip() not in ["", "."] and len(s.strip()) > 5)]
    # print("[SUMMARY]", sum_list)
    if len(sum_list) == 0:
        print("[ERROR] summary without a '.':", summary)
        return 0, None, None, None
    elif len(sum_list) == 1:
        return 1, [dialog], [0] * len(dialog), sum_list
    elif len(dialog) < len(sum_list):
        return 1, [dialog], [0] * len(dialog), [" ".join(sum_list).replace(".", "")]
    else:
        seg_dial, seg_dial_idx = [], []
        dial_curser = 0
        #print(len(dialog))
        for si, sum_item in enumerate(sum_list):
            max_score = (dial_curser, 0.0) 
            #print(dial_curser, len(dialog)-len(sum_list)+si+1)
            #print(list(range(dial_curser, len(dialog)-(len(sum_list)-si)+1, 1)))
            for di in range(dial_curser, len(dialog)-(len(sum_list)-si)+1, 1):
                context = " ".join(dialog[dial_curser:di+1])
                score = calculate_similarity(context, sum_item)
                if score > max_score[1]:
                    max_score = (di, score)
            #print(max_score)
            if si == len(sum_list) - 1:
                seg_dial.append(list(dialog[dial_curser:]))
                #seg_dial_idx.append(max_score[0])
            else:
                seg_dial.append(list(dialog[dial_curser:max_score[0]+1])) 
                dial_curser = max_score[0] + 1
                seg_dial_idx.append(max_score[0])
        seg_label = [1 if i in seg_dial_idx else 0 for i in range(len(dialog))]
        return 1, seg_dial, seg_label, sum_list

def process_dialogue(file_path, d_type):
    sources, targets = [], []
    data = json.load(open(file_path, "r"))
    print("len(data)", len(data))
    data_counter = 0
    for di, d in enumerate(tqdm(data)):
        summary = d["summary"].replace("\015", "").replace("\n", "")
        d["clean_dialog"] = [item for item in d["clean_dialog"] if item.strip() != ""]
        flag, seg_dial, seg_label, sum_list = segment_dial(d["clean_dialog"], summary)
        
        if not flag: continue
        
        d["segment_dialog"] = list(seg_dial)
        d["segment_label"] = list(seg_label)
        d["sum_list"] = list(sum_list)
        data_counter += 1
        
    with open(file_path, "w") as fout:
        json.dump(data, fout, indent=4)

if __name__ == "__main__":
    st = time.time()
    process_dialogue("../SAMsum/clean_data/test.json", "test")
    process_dialogue("../SAMsum/clean_data/eval.json", "val")
    process_dialogue("../SAMsum/clean_data/train.json", "train")
    print(time.time() - st)
