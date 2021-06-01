'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

from transformers import BartForConditionalGeneration, BartTokenizer, AdamW, get_linear_schedule_with_warmup

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import torch.nn.functional as F
import torch.nn as nn
import json
from tqdm import tqdm, trange
import argparse
import random
import numpy as np
import os
import rouge
import wandb
from difflib import SequenceMatcher
import nltk
nltk.download('punkt')


import logging
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# from metrics import evaluate_nq

SUM_TOKEN = "TLDR"
BOS_TOKEN = "<s>"
HL_TOKEN = "<hl>"

evaluator = rouge.Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                        max_n=4,
                        limit_length=True,
                        length_limit=100,
                        length_limit_type='words',
                        apply_avg=False,
                        apply_best=True,
                        alpha=0.5, # Default F1_score
                        weight_factor=1.2,
                        stemming=True)

def evaluate_rouge(dev_examples, pred):
    true_sum_arr = [d.summary for d in dev_examples]
    pred_sum_arr = [pred[d.ID] for d in dev_examples]
    assert len(true_sum_arr) == len(pred_sum_arr)
    scores = evaluator.get_scores(pred_sum_arr, true_sum_arr)
    return scores

def removeElements(A, B): 
    n = len(A) 
    return any(A == B[i:i + n] for i in range(len(B)-n + 1))

def locate_sublist(sublist, parent):
    cursor = 0
    for i, ele in enumerate(parent):
        if ele == sublist[0]:
            if parent[i: i + len(sublist)] == sublist:
                cursor = i
                break
    return cursor, cursor + len(sublist)

def get_intent(index):
    # WHY = 0; WHAT = 1; WHERE = 2; WHEN = 3; CONFIRM = 4; ABSTAIN = -1
    intent_dict = {0:"why", 1:"what", 2:"where", 3:"when", 4:"confirm", -1:"abstain"}
    return intent_dict[index]
    
class InputExample:
    def __init__(self,
                 ID,
                 context,
                 summary,
                 func_turn_label,
                 module_index,
                 key_phrases):
        self.ID = ID
        self.context = context
        self.summary = summary
        self.func_turn_label = func_turn_label
        self.module_index = module_index
        self.key_phrases = key_phrases

class InputFeatures:
    def __init__(self,
                 ID,
                 example_index,
                 source_ids,
                 source_mask,
                 source_len,
                 target_ids,
                 target_labels,
                 target_len,
                 func_turn_label):
        self.ID = ID
        self.example_index = example_index
        self.source_ids = source_ids
        self.source_mask = source_mask
        self.source_len = source_len
        self.target_ids = target_ids
        self.target_labels = target_labels
        self.target_len = target_len
        self.func_turn_label = func_turn_label
        

class CDataset(torch.utils.data.Dataset):

    def __init__(self, features, is_train):
        self.is_train = is_train
        self.length = len(features)
        self.ID = [f.ID for f in features]
        self.all_example_indices = torch.tensor([f.example_index for f in features], dtype=torch.long)
        self.all_source_ids = torch.tensor([f.source_ids for f in features], dtype=torch.long)
        self.all_source_mask = torch.tensor([f.source_mask for f in features], dtype=torch.long)
        self.all_source_len = torch.tensor([f.source_len for f in features], dtype=torch.long)
        self.all_target_ids = torch.tensor([f.target_ids for f in features], dtype=torch.long)
        self.all_target_labels = torch.tensor([f.target_labels for f in features], dtype=torch.long)
        self.all_target_len = torch.tensor([f.target_len for f in features], dtype=torch.long)
        self.all_func_label = torch.tensor([f.func_turn_label for f in features], dtype=torch.long)
        
    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.is_train:
            data = {
                "ID": self.ID[idx],
                "source_ids": self.all_source_ids[idx],
                "source_mask": self.all_source_mask[idx],
                "source_len": self.all_source_len[idx],
                "target_ids": self.all_target_ids[idx],
                "target_labels": self.all_target_labels[idx],
                "target_len": self.all_target_len[idx],
                "func_label": self.all_func_label[idx],
            }
        else:
            data = {
                "ID": self.ID[idx],
                "example_indices": self.all_example_indices[idx],
                "source_ids": self.all_source_ids[idx],
                "source_mask": self.all_source_mask[idx],
                "source_len": self.all_source_len[idx],
            }
        return data
        
        
class Bart(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.args = self.parse_args()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cpu = torch.device("cpu")

        if self.args.load_path:
            self.generator = BartForConditionalGeneration.from_pretrained(self.args.model_name, state_dict = torch.load(self.args.load_path))
        else:
            self.generator = BartForConditionalGeneration.from_pretrained(self.args.model_name)
        self.generator.to(self.device)
        self.tokenizer = BartTokenizer.from_pretrained(self.args.model_name) # Need to add base to "tokenization_bart.py" when using transformers==2.11.0
        
        if self.args.add_module_loss:
            self.classifier = nn.Linear(self.generator.model.config.d_model, 7)
            self.classifier.to(self.device)
        elif self.args.add_functurn_loss:
            self.classifier = nn.Linear(self.generator.model.config.d_model, 2)
            self.classifier.to(self.device)
        
    def save(self):
        model_to_save = (
            self.generator.module if hasattr(self.generator, "module") else self.generator
        )
        torch.save(model_to_save.state_dict(), os.path.join(self.args.output_dir, "pytorch.bin"))
        # torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pt'))
        # torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
        
    def parse_args(self):
        parser = argparse.ArgumentParser()
        
        # Gerenal
        parser.add_argument("--do_train",
                            action="store_true",
                            help = 'do model training')
        parser.add_argument("--seed",
                            default=42, type=int,
                            help="Random seed")
        
        # Input/Output
        parser.add_argument('--train_file_path',
                            type = str, default = './SAMsum/clean_data/train.json',
                            help = 'Training data path')
        parser.add_argument('--dev_file_path',
                            type = str, default = './SAMsum/clean_data/eval.json',
                            help = 'Validation data path')
        parser.add_argument('--test_file_path',
                            type = str, default = './SAMsum/clean_data/test.json',
                            help = 'Test data path')
        parser.add_argument('--load_path',
                            type = str, default = None,
                            help = 'Load trained model file')
        parser.add_argument('--output_dir',
                            type = str, required = True,
                            help = 'output saving directory')
        parser.add_argument("--gen_keyphrase_summary",
                            action="store_true", 
                            help="for decoding, first generate keyphrase then generate summary")
        parser.add_argument("--oracle_functurn_context",
                            action="store_true", 
                            help="For oracle study, using functional turns as input")
        parser.add_argument("--do_segment",
                            action="store_true",
                            help="train and evaluate with segmented dialogues")
        parser.add_argument("--use_pred_segment",
                            action="store_true",
                            help="for inference, using the predicted dialogue segmentation")
        parser.add_argument("--add_coref_input",
                            action="store_true",
                            help="appending coreference text to the dialogue text.")
        parser.add_argument("--ctrl_nb_summary_sent", 
                            default=0, type=int, 
                            help="for inference, controlling number of summary sentences by dialogue segmentation")

        # Modeling
        parser.add_argument("--model_name",
                            type=str, default='facebook/bart-large-xsum', #'facebook/bart-base',
                            help="BART model")
        parser.add_argument("--source_max_len",
                            default=512, type=int,
                            help="Max len of source")
        parser.add_argument("--target_max_len",
                            default=50, type=int,
                            help="Max len of target")
        parser.add_argument("--test_target_max_len",
                            default=50, type=int,
                            help="Max len of target")
        parser.add_argument("--beam",
                            default=4, type=int,
                            help="Beam size")
        parser.add_argument("--train_batch_size",
                            default=4, type=int,
                            help="Total batch size for training.")
        parser.add_argument("--eval_batch_size",
                            default=8, type=int,
                            help="Total batch size for eval.")
        parser.add_argument("--learning_rate",
                            default=5e-5, type=float,
                            help="The initial learning rate for Adam.")
        parser.add_argument("--num_train_epochs",
                            default=300, type=int,
                            help="Total number of training epochs to perform.")
        parser.add_argument("--warmup_proportion",
                            default=0.1, type=float,
                            help="Proportion of training to perform linear learning rate warmup for. "
                                 "E.g., 0.1 = 10%% of training.")
        parser.add_argument("--adam_epsilon", 
                            default=1e-8, type=float, 
                            help="Epsilon for Adam optimizer.")
        parser.add_argument('--gradient_accumulation_steps',
                            type=int, default=1,
                            help="Number of updates steps to accumulate before performing a backward/update pass.")
        parser.add_argument('--max_grad_norm', 
                            required=False, default=1.0, type=float,
                            help='gradient clipping for Max gradient norm.')
        parser.add_argument("--patience", 
                            default=10, type=int, 
                            help="number of validation checking for earlystopping")
        parser.add_argument("--no_repeat_ngram_size", 
                            default=0, type=int, 
                            help="for decoding, give penalty to repeated ngram")
        
        # Objectives
        parser.add_argument("--add_functurn_loss",
                            action="store_true",
                            help="Add additional training loss (encoder side) to predict 0/1 for functional turns")
        parser.add_argument("--add_module_loss",
                            action="store_true",
                            help="Add additional training loss (encoder side) to predict 7-way for modules")
        parser.add_argument("--weight_addition_loss", 
                            default=1.0, type=float, 
                            help="weights to combine additional losses")
        
        # Saving/Logging
        parser.add_argument("--validation_timing",
                            default=2500, type=int,
                            help="Check dev score after every N updates")
        parser.add_argument("--wandb", 
                            action="store_true",
                            help="use weight and bias to monitor experiments")
        parser.add_argument("--dump_pred", 
                            default=1, type=int, 
                            help="for inference, dumping prediction files")
        parser.add_argument("--add_name",
                            default="",
                            type=str,
                            help="for inference, appending string to the prediction file name")

        args = parser.parse_args()
        print(args)
        return args

        
    def load_examples(self,
                      file_path):
        examples = []

        # Get predicted segmentation for inference
        if self.args.use_pred_segment:
            assert self.args.do_train == False
            with open("save/train_segment_predictor/pred_test_new.json", 'r') as f:
                pred_segment_dict = json.load(f)
        
        # Data reading
        with open(file_path, 'r') as f:
            jsn = json.load(f)
            for data in jsn:
                summary = data["summary"].replace("\n", " ").replace("\015", "")

                if len(summary.strip()) < 5: # there are several error summary in train set
                    print("[WARNING] Skip summary [{}]".format(summary))
                    continue

                _bos_token_ = " {} ".format(BOS_TOKEN)
                if self.args.oracle_functurn_context:
                    context = "{} ".format(BOS_TOKEN) + _bos_token_.join(data["function_dialogs"]).replace("\n", " ").replace("\015", "")
                else:
                    context = "{} ".format(BOS_TOKEN) + _bos_token_.join(data["clean_dialog"]).replace("\n", " ").replace("\015", "")

                func_turn_label = data["label"]
                module_index = data["module_index"]
                key_phrases = data["key_phrases"]

                if self.args.do_segment:
                    
                    # Whether use predicted segmentation or control nb of summary sentences
                    if self.args.use_pred_segment:
                        segment_label = pred_segment_dict[data["id"]]["segment_label"]
                        if self.args.ctrl_nb_summary_sent:
                            if self.args.ctrl_nb_summary_sent == 1:
                                segment_label = [0 for s in segment_label]
                            elif self.args.ctrl_nb_summary_sent >= len(segment_label):
                                pass
                            else:
                                segment_prob = np.array(pred_segment_dict[data["id"]]["segment_prob"])
                                topk_idx = segment_prob.argsort()[-self.args.ctrl_nb_summary_sent+1:][::-1]
                                segment_label = [1 if i in topk_idx else 0 for i in range(len(segment_label))]
                        sum_list = ["summary"] * (sum(segment_label) + 1)
                    else:
                        segment_label = data["segment_label"]
                        sum_list = data["sum_list"]
                    
                    # Process input and output for different segmentation results
                    seg_count, seg_idx = 0, 0
                    if sum(segment_label) == 0:
                        context = "{} {} {} {}".format(BOS_TOKEN, HL_TOKEN, context, HL_TOKEN)

                        e = InputExample(ID="{}#{}".format(data["id"], seg_count),
                                         context=context,
                                         summary=summary,
                                         func_turn_label=func_turn_label,
                                         module_index=module_index,
                                         key_phrases=key_phrases)
                        examples.append(e)
                    else:
                        for si, seg_l in enumerate(segment_label):
                            if seg_l == 1 or si == len(segment_label) - 1:
                                temp = list(data["clean_dialog"])
                                temp[seg_idx] = "{} {}".format(HL_TOKEN, temp[seg_idx])
                                temp[si] = "{} {}".format(temp[si], HL_TOKEN)
                                context = "{} ".format(BOS_TOKEN) + _bos_token_.join(temp)

                                e = InputExample(ID="{}#{}".format(data["id"], seg_count),
                                                 context=context,
                                                 summary=sum_list[seg_count],
                                                 func_turn_label=func_turn_label,
                                                 module_index=module_index,
                                                 key_phrases=key_phrases[seg_idx:si])
                                examples.append(e)
                                seg_idx = si + 1
                                seg_count += 1
                
                else: # not doing segmentation
                    e = InputExample(ID=data["id"],
                                     context=context,
                                     summary=summary,
                                     func_turn_label=func_turn_label,
                                     module_index=module_index,
                                     key_phrases=key_phrases)
                    examples.append(e)
        
        print()
        print(file_path, len(examples))
        print("examples[0].ID", examples[0].ID)
        print("examples[0].context", examples[0].context)
        print("examples[0].summary", examples[0].summary)
        print("examples[0].func_turn_label", examples[0].func_turn_label)
        print("examples[0].module_index", examples[0].module_index)
        print("examples[0].key_phrases", examples[0].key_phrases)
        print()
        
        return examples

    def convert_examples_to_features(self,
                                     examples):
        config = self.generator.model.config
        features = []
        index = 0
        max_target_len = 0

        for e in tqdm(examples, desc='Examples'):
            
            # Process source information
            source = e.context
            source_tokens = self.tokenizer.tokenize(source)[:self.args.source_max_len-2]
            source_ids = self.tokenizer.convert_tokens_to_ids(source_tokens) + [config.eos_token_id] # <s> ... </s>
            source_len = len(source_ids)
            source_mask = [1] * source_len
            padding_len = self.args.source_max_len - source_len
            source_ids += ([config.pad_token_id] * padding_len)
            source_mask += ([0] * padding_len)
            assert len(source_ids) == self.args.source_max_len
            assert len(source_mask) == self.args.source_max_len
            
            # Process target information
            if self.args.gen_keyphrase_summary:
                string_global = []
                max_nb_turns_kp = 20 
                for ki, key_phrases in enumerate(e.key_phrases[:max_nb_turns_kp]):
                    if len(key_phrases) > 0:
                        string = [str(ki), get_intent(e.module_index[ki])] + key_phrases
                        string = " ".join(string)
                        string_global.append(string)
                    else:
                        string_global.append("{} {}".format(ki, "none"))
                string_global = " ".join(string_global[:source_ids.count(config.bos_token_id)])
                string_output = string_global + " {} ".format(SUM_TOKEN) + e.summary
                answer_tokens = self.tokenizer.tokenize(string_output)
                if len(answer_tokens) > max_target_len: max_target_len = len(answer_tokens)
                answer_tokens = answer_tokens[-self.args.target_max_len+1:] # -1 for <s> or </s>
            
            else:
                answer_tokens = self.tokenizer.tokenize(e.summary)
                if len(answer_tokens) > max_target_len: max_target_len = len(answer_tokens)
                answer_tokens = answer_tokens[:self.args.target_max_len-1] # -1 for <s> or </s>
                
            answer_tokens_ = self.tokenizer.convert_tokens_to_ids(answer_tokens)
            target_ids = [config.bos_token_id] + answer_tokens_ # <s> ...
            target_labels = answer_tokens_ + [config.eos_token_id] # ... </s>
            target_len = len(target_ids)
            padding_len = self.args.target_max_len - target_len
            target_ids += ([config.pad_token_id] * padding_len)
            target_labels += ([-100] * padding_len) # -100 is the default index to be ignored
            assert len(target_ids) == self.args.target_max_len
            assert len(target_labels) == self.args.target_max_len
            
            # Get functional turns label (truncate to max_len), either only 0/1 or 0-6 modular index
            max_num_of_turns = 50
            local_max_num_of_turns = source_ids.count(config.bos_token_id)
            if self.args.add_module_loss:
                func_turn_label = []
                counter = 0
                for ftl_i, ftl in enumerate(e.func_turn_label):
                    if (ftl == 1) or \
                       (ftl_i > 0 and e.func_turn_label[ftl_i-1] == 1) or \
                       (ftl_i < len(e.func_turn_label)-1 and e.func_turn_label[ftl_i+1] == 1):
                        func_turn_label.append(e.module_index[counter]+2)
                        counter += 1
                    else:
                        func_turn_label.append(0)                
                assert len([i for i in func_turn_label if i!=0]) == len(e.module_index)
            elif self.args.add_functurn_loss:
                func_turn_label = e.func_turn_label
            else:
                func_turn_label = [-1] * max_num_of_turns
            func_turn_label = func_turn_label[:local_max_num_of_turns][:max_num_of_turns]
            padding_len = max_num_of_turns - local_max_num_of_turns
            func_turn_label += ([-1] * padding_len)

            f = InputFeatures(e.ID, index, source_ids, source_mask, source_len, target_ids, target_labels, target_len, func_turn_label)
            features.append(f)

            index += 1
        
        print("[INFO] max_target_len", max_target_len)
        return features

    def init_seed(self):
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        if torch.cuda.is_available:
            torch.cuda.manual_seed(self.args.seed)

    def get_optimizer(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.generator.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": 0.01},
            {"params": [p for n, p in self.generator.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        return AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)

    def get_train_dataloader(self,
                             train_features,
                             train_batch_size):
        train_data = CDataset(train_features, is_train=True)
        train_sampler = RandomSampler(train_data)
        return DataLoader(train_data, sampler=train_sampler, batch_size=train_batch_size)

    def get_eval_dataloader(self,
                            dev_features,
                            dev_batch_size):
        eval_data = CDataset(dev_features, is_train=False)
        eval_sampler = SequentialSampler(eval_data)
        return DataLoader(eval_data, sampler=eval_sampler, batch_size=dev_batch_size)
    
    def get_train_batch_data(self,
                             batch):
        batch_source_max_len = batch["source_len"].max().item()
        batch_target_max_len = batch["target_len"].max().item()
        # batch = tuple(t.to(self.device) for t in batch)
        source_ids = batch["source_ids"][:, :batch_source_max_len].to(self.device)
        source_mask = batch["source_mask"][:, :batch_source_max_len].to(self.device)
        target_ids = batch["target_ids"][:, :batch_target_max_len].to(self.device)
        target_labels = batch["target_labels"][:, :batch_target_max_len].contiguous().to(self.device)
        item = {
            "ID": batch["ID"],
            "source_ids": source_ids,
            "source_mask": source_mask,
            "target_ids": target_ids,
            "target_labels": target_labels,
            "func_label": batch["func_label"],
        }
        return item

    def get_eval_batch_data(self,
                            batch):
        example_indices = batch["example_indices"].tolist()
        batch_source_max_len = batch["source_len"].max().item()
        # batch = tuple(t.to(self.device) for t in batch)
        source_ids = batch["source_ids"][:, :batch_source_max_len].to(self.device)
        source_mask = batch["source_mask"][:, :batch_source_max_len].to(self.device)
        item = {
            "ID": batch["ID"],
            "example_indices":example_indices,
            "source_ids": source_ids,
            "source_mask": source_mask,
        }
        return item
    
    def encode(self,
               source_ids,
               source_mask):
        B = source_ids.size(0)
        N = source_ids.size(1)
        source_reps = self.generator.model.encoder(input_ids = source_ids,
                                                   attention_mask = source_mask)
        source_reps = source_reps[0]
        return source_reps, source_mask
    
    def train(self):
        self.init_seed()
        
        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        else:
            print("[WARNING] {} exists...".format(self.args.output_dir))
            exit(1)
        
        
        train_examples = self.load_examples(self.args.train_file_path)
        train_features = self.convert_examples_to_features(train_examples)
        dev_examples = self.load_examples(self.args.dev_file_path)
        dev_features = self.convert_examples_to_features(dev_examples)
        dev_data = (dev_examples, dev_features)
        train_batch_size = int(self.args.train_batch_size / self.args.gradient_accumulation_steps)
        num_train_steps = int(len(train_features) / train_batch_size / self.args.gradient_accumulation_steps * self.args.num_train_epochs)

        optimizer = self.get_optimizer()
        t_total = num_train_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(t_total * self.args.warmup_proportion), num_training_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(train_features))
        logger.info("  Batch size = %d", train_batch_size)
        logger.info("  Num steps = %d", num_train_steps)

        train_dataloader = self.get_train_dataloader(train_features, train_batch_size)
        
        self.generator.zero_grad()
        self.generator.train()
        
        num_updates = 0
        best_em = 0.0
        patience = 0

        f_log = open(os.path.join(self.args.output_dir, "{}.log".format(self.args.model_name.replace("/", "-"))), 'w')
            
        for _ in trange(int(self.args.num_train_epochs), desc="Epoch"):
            train_loss_tracker_gen, train_loss_tracker_func, train_loss_tracker_gp = [], [], []
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                
                loss = 0
                item_dict = self.get_train_batch_data(batch)
                source_ids, source_mask  = item_dict["source_ids"], item_dict["source_mask"] 
                target_ids, target_labels, func_labels = item_dict["target_ids"], item_dict["target_labels"], item_dict["func_label"]
                
                encoder_outputs, source_mask = self.encode(source_ids, source_mask)
                batch_size = encoder_outputs.size(0)
                
                if self.args.add_functurn_loss or self.args.add_module_loss:
                    sent_repr_mat = []
                    turn_nums = [(item == self.generator.model.config.bos_token_id).sum().cpu().item() for item in source_ids]
                    max_turn_num = max(turn_nums)
                    for i in range(batch_size):
                        sent_repr = encoder_outputs[i][source_ids[i] == self.generator.model.config.bos_token_id]  # [num_of_turns, hd_dim]
                        sent_repr = torch.cat(
                            [sent_repr, torch.zeros(max_turn_num - turn_nums[i], sent_repr.size(1)).to(self.device)], 0)
                        sent_repr_mat.append(sent_repr)
                        func_labels[i][turn_nums[i]:] = -1
                    sent_repr_mat = torch.stack(sent_repr_mat, 0)  # [batch_size, max_turn_num, hd_dim]

                    func_labels = func_labels[:, :max_turn_num]
                    prediction_logits = self.classifier(sent_repr_mat)
                    loss_functurn = F.cross_entropy(prediction_logits.reshape(-1, prediction_logits.size(-1)), \
                                                    func_labels.reshape(-1), ignore_index=-1, reduction='mean')
                    loss += loss_functurn * self.args.weight_addition_loss
                    
                    train_loss_tracker_func.append(loss_functurn.item())
                    if self.args.wandb and step % 50 == 0:
                        wandb.log({'avg_training_loss_functurn': np.mean(train_loss_tracker_func)})
 
                outputs = self.generator(input_ids = None,
                                         attention_mask = source_mask,
                                         encoder_outputs = (encoder_outputs,),
                                         decoder_input_ids = target_ids,
                                         labels = target_labels)
                
                # outputs = self.generator(input_ids = source_ids,
                #                          attention_mask = source_mask,
                #                          decoder_input_ids = target_ids,
                #                          labels = target_labels)
                
                loss_gen = outputs[0]
                #encoder_outputs = outputs[2]
                
                train_loss_tracker_gen.append(loss_gen.item())
                if self.args.wandb and step % 50 == 0:
                    wandb.log({'avg_training_loss_generation': np.mean(train_loss_tracker_gen)})
                
                loss += loss_gen
                
                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.generator.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    self.generator.zero_grad()
                    num_updates += 1

                    if num_updates % self.args.validation_timing == 0:
                        results = self.evaluate(dev_data)
                        em = results['rouge-1']['f']
                        
                        if self.args.wandb:
                            for r in results:
                                wandb.log({'eval_{}'.format(r): results[r]['f']})
                        
                        if f_log is not None:
                            f_log.write(json.dumps(results))
                            f_log.write('num_updates: {}\n'.format(num_updates))
                            f_log.flush()

                        if em > best_em:
                            best_em = em
                            patience = 0
                            self.save()
                        else:
                            patience += 1
                            print("[INFO] patience {}/{}".format(patience, self.args.patience))
                
                if patience > self.args.patience:
                    break
            
            if patience > self.args.patience:
                print("[INFO] Ran out of patience...")
                break

        if f_log is not None:
            f_log.close()

    def predict(self,
                dev_data):

        dev_examples, dev_features = dev_data
        eval_dataloader = self.get_eval_dataloader(dev_features, self.args.eval_batch_size)

        self.generator.eval()

        pred = {} #[None] * len(dev_examples)
        pred_kp = {}

        for bi, batch in enumerate(tqdm(eval_dataloader, desc="Generating")):
            item_dict = self.get_eval_batch_data(batch)
            IDs, example_indices, source_ids, source_mask = item_dict["ID"], item_dict["example_indices"], item_dict["source_ids"], item_dict["source_mask"]
            
            with torch.no_grad():
                # encoder_outputs, source_mask = self.encode(source_ids, source_mask)
                
                no_repeat_ngram_size = self.args.no_repeat_ngram_size
                    
                target_ids = self.generator.generate(input_ids = source_ids, attention_mask = source_mask,
                                                     num_beams = self.args.beam,
                                                     max_length = self.args.test_target_max_len,
                                                     no_repeat_ngram_size = no_repeat_ngram_size,
                                                     early_stopping = True)

            target_ids = target_ids.to(self.cpu)
            for i in range(len(example_indices)):

                if IDs[i] in pred.keys():
                    continue
                
                answer = self.tokenizer.decode(target_ids[i].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
                
                if self.args.gen_keyphrase_summary:
                    if SUM_TOKEN in answer:
                        answer_split = answer.split(SUM_TOKEN)
                        keyphrase, summary = answer_split[0], answer_split[1]
                    else:
                        summary = " ".join(answer.strip().split(" ")[-50:])
                        keyphrase = " ".join(answer.strip().split(" ")[:-50])
                        print("[WARNING] No special token [{}] found in the output...".format(SUM_TOKEN))
                        # print(appr_answer)
                    
                    pred_kp[IDs[i]] = keyphrase.strip()
                    pred[IDs[i]] = summary.strip()
                else:
                    pred[IDs[i]] = answer.strip()
            
        self.generator.train()
        return pred, pred_kp

    def evaluate(self,
                 dev_data = None,
                 source="dev",
                 dump_pred=False):

        if dev_data is None:
            file_path = self.args.dev_file_path if source == "dev" else self.args.test_file_path
            dev_examples = self.load_examples(file_path)
            dev_features = self.convert_examples_to_features(dev_examples)
            print("[INFO] Testing file_path:", file_path)
        else:
            dev_examples, dev_features = dev_data
        
        pred, pred_kp = self.predict((dev_examples, dev_features))
        scores = evaluate_rouge(dev_examples, pred)
        
        print("\n\n")
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            print('{}:\t{}: {:5.4f}\t{}: {:5.4f}\t{}: {:5.4f}'.format(metric, 'P', 100.0 * results['p'], 'R', 100.0 * results['r'], 'F1', 100.0 * results['f']))
        print("\n\n")

        if not os.path.exists(self.args.output_dir):
            os.makedirs(self.args.output_dir)
        
        if dump_pred:
            with open(os.path.join(self.args.output_dir, "{}.pred.summary{}.json".format(source, self.args.add_name)), "w") as fout:
                json.dump(pred, fout, indent=4)
        
            with open(os.path.join(self.args.output_dir, "{}.predkp.summary{}.json".format(source, self.args.add_name)), "w") as fout:
                json.dump(pred_kp, fout, indent=4)
        
        return scores
    
if __name__ == '__main__':
    model = Bart()
    
    # init wandb
    if model.args.wandb:
        name = model.args.output_dir.replace("/", "-")
        wandb.init(project="DialSum", name=name, group="hfbart", job_type="train")
        wandb.config.update(model.args)
        wandb.watch(model)
    
    if model.args.do_train:
        print("[INFO] Start training ...") 
        model.train()
        model.generator = BartForConditionalGeneration.from_pretrained(model.args.model_name, state_dict = torch.load(os.path.join(model.args.output_dir, "pytorch.bin"))) 
        model.generator.to(model.device)
    else:
        if model.args.load_path == "":
            print("[ERROR] No trained model specified...")
            exit(1)
    
    print("[INFO] Start generate test summary...") 
    scores = model.evaluate(source="test", dump_pred=model.args.dump_pred)
    with open(os.path.join(model.args.output_dir, "test.metrics{}".format(model.args.add_name)), "w") as fout:
        json.dump(scores, fout, indent=4)

    # Combine separate segments into one single summary for evaluation
    if model.args.do_segment:
        print("[INFO] Combine separate segments into one single summary...")
        
        summarys = {}
        f_gold = json.load(open("SAMsum/clean_data/test.json", "r"))
        for di, data in enumerate(f_gold):
            summarys[data["id"]] = data["summary"].strip().replace("\n", " ").replace("\015", "")

        f_pred_bart_single = json.load(open(os.path.join(model.args.output_dir, "test.pred.summary{}.json".format(model.args.add_name)), "r"))
        update_pred = {}
        for item_id, item in f_pred_bart_single.items():
            ID, turn = item_id.rsplit("#", 1)
            if ID not in update_pred.keys():
                update_pred[ID] = {}
            update_pred[ID][turn] = item
        
        final_full_pred = {}
        for ID, value in update_pred.items():
            final_full_pred[ID] = []
            for t in range(len(value)):
                if t == 0:
                    final_full_pred[ID].append(value[str(t)])
                else:
                    similar_score_max = max([SequenceMatcher(None, s, value[str(t)]).ratio() for s in final_full_pred[ID]])
                    if similar_score_max < 0.75:
                        final_full_pred[ID].append(value[str(t)])
        
        with open(os.path.join(model.args.output_dir, "test.pred.summary_full{}".format(model.args.add_name)), "w") as fout:
            json.dump(final_full_pred, fout, indent=4)

        test_summary = []
        test_prediction = []
        for ID, value in final_full_pred.items():
            test_summary.append(summarys[ID])
            test_prediction.append(" ".join(value))
        
        print("Testing {} samples...".format(len(test_prediction)))
        
        scores = evaluator.get_scores(test_prediction, test_summary)
        with open(os.path.join(model.args.output_dir, "test.metrics_full{}".format(model.args.add_name)), "w") as fout:
            json.dump(scores, fout, indent=4)
        
        
        print("\n\n")
        for metric, results in sorted(scores.items(), key=lambda x: x[0]):
            print('{}:\t{}: {:5.4f}\t{}: {:5.4f}\t{}: {:5.4f}'.format(metric, 'P', 100.0 * results['p'], 'R', 100.0 * results['r'], 'F1', 100.0 * results['f']))
        print("\n\n")
        