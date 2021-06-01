'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''

import argparse
import torch
from torch.utils.data import DataLoader
import os

from model import SegmentPredictor
import json

from transformers.optimization import AdamW
from transformers import get_linear_schedule_with_warmup
from transformers import BertTokenizer, AutoTokenizer

from dataset import DialogueSegmentation
import solver
import wandb


def get_optimzier(model, args):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
    ]
    return AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon, betas=(0., 0.999))


# CUDA_VISIBLE_DEVICES=2 python train_segmemt_predictor.py --data_dir=segment_data/clean_dial_single_strict/bart/ --output_dir=save/train_segment_predictor/ --do_train --wandb


def run():
    parser = argparse.ArgumentParser(description='Training Settings')
    parser.add_argument("--model_file", default="bert-base-uncased")
    # parser.add_argument("--data_dir", default='/export/home/dialog_summarization/labeling/data/')
    parser.add_argument("--data_dir", default='SAMsum/clean_data/')
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--max_single_turn_length", type=int, default=30)
    parser.add_argument("--max_num_of_turns", type=int, default=30)
    parser.add_argument("--num_train_epochs", type=int, default=200)
    parser.add_argument("--learning_rate", type=float, default=1E-5)
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--hidden_dimension", default=768, help='BERT hd')
    parser.add_argument("--load_model_from_ckpt", type=bool, default=True)
    parser.add_argument("--ckpt_path", default='ckpt/model_9.bin')
    
    # my add
    parser.add_argument("--patience", default=8, help='BERT hd')
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--output_dir", default=None, required=True)
    parser.add_argument("--load_path", default=None)
    parser.add_argument("--do_train", action="store_true")

    args = parser.parse_args()
    
    if os.path.exists(args.output_dir):
        if args.do_train:
            print("[WARNING] output_dir {} exists...".format(args.output_dir))
            exit(1)
    else:
        os.makedirs(args.output_dir)
    
    # init wandb
    if args.wandb:
        wandb.init(project="DialSum", group="segment_predictor", job_type="train")
        wandb.config.update(args)

    
    tokenizer = AutoTokenizer.from_pretrained(args.model_file)
    
    train_dataset = DialogueSegmentation(tokenizer, filepath=os.path.join(args.data_dir, 'train.json'),
                             bert_path=args.model_file, max_sequence_length=args.max_seq_length,
                             max_single_turn_length=args.max_single_turn_length,
                             max_num_of_turns=args.max_num_of_turns)
    dev_dataset = DialogueSegmentation(tokenizer, filepath=os.path.join(args.data_dir, 'eval.json'),
                           bert_path=args.model_file, max_sequence_length=args.max_seq_length,
                           max_single_turn_length=args.max_single_turn_length,
                           max_num_of_turns=args.max_num_of_turns)
    test_dataset = DialogueSegmentation(tokenizer, filepath=os.path.join(args.data_dir, 'test.json'),
                            bert_path=args.model_file, max_sequence_length=args.max_seq_length,
                            max_single_turn_length=args.max_single_turn_length,
                            max_num_of_turns=args.max_num_of_turns)

    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=args.eval_batch_size, shuffle=False, drop_last=False)

    t_total = len(train_dataloader) * args.num_train_epochs

    predictor = SegmentPredictor(bert_path=args.model_file, num_labels=2)

    optimizer = get_optimzier(predictor, args)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    n_gpu = torch.cuda.device_count()
    
    if args.load_path:
        print("[INFO] Loading model from", args.load_path)
        predictor.load_state_dict(torch.load(args.load_path))
    else:
        print("[INFO] Reinitializing model...")
    
    if args.wandb:
        wandb.watch(predictor)
    
    # Predict intent for each turn
    if args.do_train:
        solver.train_intent_predictor(predictor, args, wandb, optimizer, scheduler, train_dataloader, dev_dataloader,
                                  epochs=args.num_train_epochs, gpus=range(n_gpu), max_grad_norm=args.max_grad_norm)
        predictor.load_state_dict(torch.load(os.path.join(args.output_dir, "pytorch.bin")))
    
    solver.dump_segment(args, "pred_test_new", predictor, test_dataloader)

if __name__ == '__main__':
    run()
