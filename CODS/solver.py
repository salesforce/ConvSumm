'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''


import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os
import json
from sklearn.metrics import f1_score

from model import TypedModel
from util import compute_accuracy


def train_typed_classification(base_model: TypedModel, args, wandb, optimizer, scheduler, train_dataloader, dev_dataloader,
                               epochs=10, gpus=[], max_grad_norm=1.0):
    """
    Label each turn as useful (marked as 'function_dialogs' in data) or chitchat garbage
    """
    if len(gpus) > 1:
        # multi gpu
        parallel_model = nn.DataParallel(base_model, device_ids=gpus).cuda()
    elif len(gpus) == 1:
        # single gpu
        parallel_model = base_model.cuda()
    else:
        # no gpu available
        parallel_model = base_model
    
    # create logger
    f_write = open(os.path.join(args.output_dir, "train_log.txt"), "w")
    
    step = 0
    best_acc, patience = 0, 0
    for epoch_num in range(epochs):
        base_model.train()
        parallel_model.train()
        train_loss_tracker = []
        for di, data in tqdm(enumerate(train_dataloader)):
            
            if di == 0 and epoch_num == 0: 
                for k, v in data.items():
                    if torch.is_tensor(v):
                        print(k, v.size())
                    else:
                        print(k, v[:5])
                        
            loss = parallel_model(data)
            if len(gpus) > 1:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            train_loss_tracker.append(loss.item())
            
            if args.wandb and step % 10 == 0:
                wandb.log({'avg_training_loss': np.mean(train_loss_tracker)})
        
        eval_loss, eval_prediction, eval_truth = evaluate(base_model, dev_dataloader)
        acc = compute_accuracy(eval_prediction, eval_truth)
        train_loss = np.mean(train_loss_tracker)
        print_text = 'epoch {}, train loss: {}, dev loss: {}, dev acc: {}'.format(epoch_num, train_loss, eval_loss, acc)
        print(print_text)
        f_write.write(print_text+"\n")
        
        if args.wandb:
            wandb.log({'eval_loss': eval_loss})
            wandb.log({'dev_acc': acc})
        
        if acc > best_acc:
            # save model
            model_to_save = (base_model.module if hasattr(base_model, "module") else base_model)  # Take care of distributed/parallel training

            print("Saving model checkpoint to %s", args.output_dir)
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "pytorch.bin"))
            # tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
            
            best_acc = acc
            patience = 0
        else:
            patience += 1
            print("[INFO] Patience {}/{}".format(patience, args.patience))
        
        f_write.write("Best Dev Acc: {:.5f}\n".format(best_acc))
        
        if patience > args.patience:
            print("[INFO] Run out of patience...")
            break
    
    f_write.close()


def dump_functional_turns(args, data_type, base_model, evaluate_dataloader):
    print("[INFO] Dump functional turns...")
    
    base_model.eval()
    eval_predictions = {}
    eval_p = []
    eval_g = []
    for eval_data in tqdm(evaluate_dataloader):
        eval_loss, eval_prediction_logits = base_model(eval_data, evaluate=True)
        eval_prediction = eval_prediction_logits.topk(1)[1].squeeze()
        eval_p.extend(eval_prediction.cpu().tolist())
        eval_g.extend(eval_data['label'].cpu().tolist())
        
        for i, _id in enumerate(eval_data["id"]):
            #if eval_prediction[i].item() == 1:
            dial_id, turn_id = _id.split(":turn")
            turn_id = int(turn_id)
            if dial_id not in eval_predictions.keys():
                eval_predictions[dial_id] = {}
            eval_predictions[dial_id][turn_id] = {"text": eval_data["document_plain"][i], "pred":eval_prediction[i].cpu().item()}
    
    acc = compute_accuracy(eval_p, eval_g)
    print("ACC: {}".format(acc))
    
    # clean eval_predictions
    new_eval_predictions = []
    for _id, value in eval_predictions.items():
        triggered_ids = set()
        for key in sorted(value.keys()):
            if value[key]["pred"] == 1:
                triggered_ids.add(key)
                if key > 0:
                    triggered_ids.add(key-1)
                if key < len(value.keys()) - 1:
                    triggered_ids.add(key+1)
                    
        new_func_turns = [value[item]["text"] for item in sorted(triggered_ids)]
        new_eval_predictions.append({"id":_id, "function_dialogs": new_func_turns})
    
    # add fake labels
    for i, item in enumerate(new_eval_predictions):
        item["module_index"] = [-1] * len(item["function_dialogs"])
        item["key_phrases"] = [["-1"]] * len(item["function_dialogs"])
    
    with open(os.path.join(args.output_dir, "{}.json".format(data_type)), "w") as fout:
        json.dump(new_eval_predictions, fout, indent=4)
    
    with open(os.path.join(args.output_dir, "{}_log.txt".format(data_type)), "w") as fout:
        fout.write("ACC: {}".format(acc))
    
    return new_eval_predictions
    
    
def evaluate(base_model, evaluate_dataloader):
    base_model.eval()
    eval_losses = []
    eval_predictions = []
    eval_gtruths = []
    for eval_data in evaluate_dataloader:
        eval_loss, eval_prediction_logits = base_model(eval_data, evaluate=True)
        eval_prediction = eval_prediction_logits.topk(1)[1].squeeze()
        eval_losses.append(eval_loss.item())
        eval_predictions.extend(eval_prediction.cpu().tolist())
        eval_gtruths.extend(eval_data['label'].cpu().tolist())
    return np.mean(eval_losses), eval_predictions, eval_gtruths


def train_intent_predictor(base_model: TypedModel, args, wandb, optimizer, scheduler, train_dataloader, dev_dataloader,
                           epochs=10, gpus=[], max_grad_norm=1.0):
    """
    For each useful turn in 'function_dialogs', label its intent as
    WHY, WHAT, WHERE, WHEN, CONFIRM, or ABSTAIN
    """
    if len(gpus) > 1:
        # multi gpu
        parallel_model = nn.DataParallel(base_model, device_ids=gpus).cuda()
    elif len(gpus) == 1:
        # single gpu
        parallel_model = base_model.cuda()
    else:
        # no gpu available
        parallel_model = base_model

    # create logger
    f_write = open(os.path.join(args.output_dir, "train_log.txt"), "w")
    
    step = 0
    best_acc, patience = 0, 0
    for epoch_num in range(epochs):
        base_model.train()
        parallel_model.train()
        train_loss_tracker = []
        for di, data in enumerate(tqdm(train_dataloader)):
            
            if di == 0 and epoch_num == 0: 
                print()
                for k, v in data.items():
                    if torch.is_tensor(v):
                        print("[INFO] ", k, v.size())
                    else:
                        print("[INFO] ", k, v[:5])
            
            loss = parallel_model(data)
            if len(gpus) > 1:
                loss = loss.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parallel_model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            train_loss_tracker.append(loss.item())
            
            if args.wandb and step % 10 == 0:
                wandb.log({'avg_training_loss': np.mean(train_loss_tracker)})

        eval_loss, eval_prediction, eval_truth = evaluate_module_index(base_model, dev_dataloader)
        acc = compute_accuracy(eval_prediction, eval_truth)
        f1 = f1_score(y_pred=eval_prediction, y_true=eval_truth, average="macro")
        train_loss = np.mean(train_loss_tracker)
        print_text = 'epoch {}, train loss: {}, dev loss: {:.4f}, dev acc: {:.4f}, dev f1-macro: {:.4f}'.format(epoch_num, train_loss, eval_loss, acc, f1)
        print(print_text)
        f_write.write(print_text+"\n")
        
        if args.wandb:
            wandb.log({'eval_loss': eval_loss})
            wandb.log({'dev_acc': acc})
        
        if f1 > best_acc:
            # save model
            model_to_save = (base_model.module if hasattr(base_model, "module") else base_model)  # Take care of distributed/parallel training

            print("Saving model checkpoint to %s", args.output_dir)
            torch.save(model_to_save.state_dict(), os.path.join(args.output_dir, "pytorch.bin"))
            # tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
            torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(args.output_dir, "scheduler.pt"))
            
            best_acc = f1
            patience = 0
        else:
            patience += 1
            print("[INFO] Patience {}/{}".format(patience, args.patience))
        
        f_write.write("Best Dev Acc: {:.5f}\n".format(best_acc))
        
        if patience > args.patience:
            print("[INFO] Run out of patience...")
            break
    
    f_write.close()
    

def evaluate_module_index(base_model, evaluate_dataloader):
    base_model.eval()
    eval_losses = []
    eval_predictions = []
    eval_gtruths = []
    for eval_data in evaluate_dataloader:
        eval_loss, eval_prediction_logits, module_index = base_model(eval_data, evaluate=True)
        eval_prediction_logits = torch.cat(eval_prediction_logits, 0)
        #print(eval_loss, eval_prediction_logits, module_index)
        eval_prediction = eval_prediction_logits.topk(1)[1].squeeze()
        # eval_prediction = [ts.topk(1)[1] for ts in eval_prediction_logits]

        eval_losses.append(eval_loss.item())
        eval_predictions.extend(eval_prediction.cpu().tolist())
        # eval_predictions.extend(eval_prediction)
        module_index = module_index.cpu().tolist()
        for item in module_index:
            eval_gtruths.extend(list(filter((-1).__ne__, item)))
    return np.mean(eval_losses), eval_predictions, eval_gtruths


def dump_intent(args, data_type, base_model, evaluate_dataloader):
    print("[INFO] Dump functional turns...")
    
    base_model.eval()
    eval_losses = []
    eval_predictions = []
    eval_gtruths = []
    out_data = {}
    for eval_data in tqdm(evaluate_dataloader):
        eval_loss, eval_prediction_logits, module_index = base_model(eval_data, evaluate=True)
        eval_prediction = eval_prediction_logits.topk(1)[1].squeeze()
        eval_losses.append(eval_loss.item())
        eval_predictions.extend(eval_prediction.cpu().tolist())
        # Filter -1 in module_index
        module_index = module_index.cpu().tolist()
        for item in module_index:
            eval_gtruths.extend(list(filter((-1).__ne__, item)))
        
        for i, _id in enumerate(eval_data["id"]):
            out_data[_id] = {
                "module_index": [idx-1 for idx in module_index[i] if idx != -1],
            }
    
    acc = compute_accuracy(eval_predictions, eval_gtruths)
    f1 = f1_score(y_pred=eval_predictions, y_true=eval_gtruths, average="weighted")
    print("ACC: {}, F1-MA: {}".format(acc, f1))
    
    with open(os.path.join(args.output_dir, "{}.json".format(data_type)), "w") as fout:
        json.dump(out_data, fout, indent=4)
    
    with open(os.path.join(args.output_dir, "{}_log.txt".format(data_type)), "w") as fout:
        fout.write("ACC: {}, F1-MA: {}".format(acc, f1))
    
    return out_data

def dump_segment(args, data_type, base_model, evaluate_dataloader):
    print("[INFO] Dump Segment Label...")
    
    base_model.eval()
    eval_losses = []
    eval_predictions = []
    eval_gtruths = []
    out_data = {}
    softmax_layer = nn.Softmax(-1)
    for eval_data in tqdm(evaluate_dataloader):
        eval_loss, eval_prediction_logits, segment_index = base_model(eval_data, evaluate=True)
        eval_prediction_logits_cat = torch.cat(eval_prediction_logits, 0)
        eval_probs = [softmax_layer(logit) for logit in eval_prediction_logits]
        eval_prediction = eval_prediction_logits_cat.topk(1)[1].squeeze()
        eval_losses.append(eval_loss.item())
        eval_predictions.extend(eval_prediction.cpu().tolist())
        # Filter -1 in segment_index
        segment_index = segment_index.cpu().tolist()
        for item in segment_index:
            eval_gtruths.extend(list(filter((-1).__ne__, item)))
        
        for i, _id in enumerate(eval_data["id"]):
            
            segment_prob, segment_label = [], []
            for segi, idx in enumerate(segment_index[i]):
                if idx != -1:
                    segment_label.append(idx)
                    segment_prob.append(eval_probs[i][segi, 1].item())
            
            out_data[_id] = {
                "segment_label": segment_label,
                "segment_prob": segment_prob
            }
    
    acc = compute_accuracy(eval_predictions, eval_gtruths)
    f1 = f1_score(y_pred=eval_predictions, y_true=eval_gtruths, average="weighted")
    print("ACC: {}, F1-MA: {}".format(acc, f1))
    
    with open(os.path.join(args.output_dir, "{}.json".format(data_type)), "w") as fout:
        json.dump(out_data, fout, indent=4)
    
    with open(os.path.join(args.output_dir, "{}_log.txt".format(data_type)), "w") as fout:
        fout.write("ACC: {}, F1-MA: {}".format(acc, f1))
    
    return out_data
