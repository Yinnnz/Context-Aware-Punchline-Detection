#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import argparse
import csv
import logging
import os
import random
import pickle
import sys
from global_config import *
import numpy as np
import wandb

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from torch.nn import BCEWithLogitsLoss
from transformers import (
    AlbertTokenizer,
    AlbertModel,
    get_linear_schedule_with_warmup,
)
from models import *
from transformers.optimization import AdamW

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, choices=["HKT"], default="HKT")
parser.add_argument("--dataset", type=str, choices=["humor", "sarcasm"], default="humor")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=85)
parser.add_argument("--n_layers", type=int, default=1)
parser.add_argument("--n_heads", type=int, default=1)
parser.add_argument("--cross_n_layers", type=int, default=1)
parser.add_argument("--cross_n_heads", type=int, default=4)
parser.add_argument("--fusion_dim", type=int, default=172)
parser.add_argument("--dropout", type=float, default=0.2366)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--seed", type=int, default=100)
parser.add_argument("--learning_rate", type=float, default=0.000005)
parser.add_argument("--learning_rate_a", type=float, default=0.003)
parser.add_argument("--learning_rate_h", type=float, default=0.0003)
parser.add_argument("--warmup_ratio", type=float, default=0.07178)
parser.add_argument("--save_weight", type=str, choices=["True", "False"], default="False")
args = parser.parse_args()

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, acoustic, hcf, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.acoustic = acoustic
        self.hcf = hcf
        self.label_id = label_id

def get_inversion(tokens, SPIECE_MARKER="▁"):
    inversion_index = -1
    inversions = []
    for token in tokens:
        if SPIECE_MARKER in token:
            inversion_index += 1
        inversions.append(inversion_index)
    return inversions

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    pop_count = 0
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) == 0:
            tokens_b.pop()
        else:
            pop_count += 1
            tokens_a.pop(0)
    return pop_count

def convert_humor_to_features(examples, tokenizer):
    features = []
    for (ex_index, example) in enumerate(examples):
        (p_words, _, p_acoustic, p_hcf), (c_words, _, c_acoustic, c_hcf), hid, label = example
        text_a = ". ".join(c_words)
        text_b = p_words + "."
        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)

        inversions_a = get_inversion(tokens_a)
        inversions_b = get_inversion(tokens_b)

        pop_count = _truncate_seq_pair(tokens_a, tokens_b, args.max_seq_length - 3)
        inversions_a = inversions_a[pop_count:]
        inversions_b = inversions_b[: len(tokens_b)]

        acoustic_a = [c_acoustic[inv_id, :].reshape(1, -1) for inv_id in inversions_a]
        acoustic_b = [p_acoustic[inv_id, :].reshape(1, -1) for inv_id in inversions_b]
        hcf_a = [c_hcf[inv_id, :].reshape(1, -1) for inv_id in inversions_a]
        hcf_b = [p_hcf[inv_id, :].reshape(1, -1) for inv_id in inversions_b]

        acoustic = np.concatenate([
            np.zeros((1, ACOUSTIC_DIM_ALL))] + acoustic_a + [
            np.zeros((1, ACOUSTIC_DIM_ALL))] + acoustic_b + [
            np.zeros((1, ACOUSTIC_DIM_ALL))])
        acoustic = np.take(acoustic, acoustic_features_list, axis=1)

        hcf = np.concatenate([
            np.zeros((1, 4))] + hcf_a + [
            np.zeros((1, 4))] + hcf_b + [
            np.zeros((1, 4))])

        tokens = ["[CLS]"] + tokens_a + ["[SEP]"] + tokens_b + ["[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
        input_mask = [1] * len(input_ids)

        pad_len = args.max_seq_length - len(input_ids)
        input_ids += [0] * pad_len
        input_mask += [0] * pad_len
        segment_ids += [0] * pad_len

        acoustic = np.concatenate((acoustic, np.zeros((pad_len, acoustic.shape[1]))))
        hcf = np.concatenate((hcf, np.zeros((pad_len, hcf.shape[1]))))

        features.append(InputFeatures(input_ids, input_mask, segment_ids, acoustic, hcf, float(label)))
    return features

def get_appropriate_dataset(data, tokenizer):
    features = convert_humor_to_features(data, tokenizer)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    hcf = torch.tensor([f.hcf for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
    dataset = TensorDataset(all_input_ids, all_acoustic, all_input_mask, all_segment_ids, hcf, all_label_ids)
    return dataset

def set_up_data_loader():
    data_file = "ur_funny.pkl" if args.dataset == "humor" else "mustard.pkl"
    with open(os.path.join(DATASET_LOCATION, args.dataset, data_file), "rb") as handle:
        all_data = pickle.load(handle)
    train_data, dev_data, test_data = all_data["train"], all_data["dev"], all_data["test"]

    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    train_dataset = get_appropriate_dataset(train_data, tokenizer)
    dev_dataset = get_appropriate_dataset(dev_data, tokenizer)
    test_dataset = get_appropriate_dataset(test_data, tokenizer)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    dev_dataloader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_dataloader, dev_dataloader, test_dataloader


def train_epoch(model, train_dataloader, optimizer, scheduler, loss_fct):
    model.train()
    tr_loss = 0
    nb_tr_steps = 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, acoustic, input_mask, segment_ids, hcf, label_ids = batch

        outputs = model(input_ids, acoustic, hcf, token_type_ids=segment_ids, attention_mask=input_mask)
        logits = outputs[0]
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        loss.backward()
        for o_i in range(len(optimizer)):
            optimizer[o_i].step()
            scheduler[o_i].step()
        model.zero_grad()

        tr_loss += loss.item()
        nb_tr_steps += 1
    return tr_loss / nb_tr_steps

def eval_epoch(model, dev_dataloader, loss_fct):
    model.eval()
    dev_loss = 0
    nb_dev_steps = 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Eval")):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, acoustic, input_mask, segment_ids, hcf, label_ids = batch
            outputs = model(input_ids, acoustic, hcf, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = outputs[0]
            loss = loss_fct(logits.view(-1), label_ids.view(-1))
            dev_loss += loss.item()
            nb_dev_steps += 1
    return dev_loss / nb_dev_steps

def test_score_model(model, test_dataloader, loss_fct):
    model.eval()
    eval_loss = 0
    preds = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Test"):
            batch = tuple(t.to(DEVICE) for t in batch)
            input_ids, acoustic, input_mask, segment_ids, hcf, label_ids = batch
            outputs = model(input_ids, acoustic, hcf, token_type_ids=segment_ids, attention_mask=input_mask)
            logits = torch.sigmoid(outputs[0])
            preds.append(logits.detach().cpu().numpy())
            all_labels.append(label_ids.detach().cpu().numpy())
    preds = np.round(np.concatenate(preds)).squeeze()
    all_labels = np.concatenate(all_labels).squeeze()
    accuracy = accuracy_score(all_labels, preds)
    f_score = f1_score(all_labels, preds, average="weighted")
    return accuracy, f_score


def prep_for_training(num_training_steps):
    if args.dataset == "humor":
        acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=8, nhead=3, dim_feedforward=256)
        acoustic_model.load_state_dict(torch.load("./model_weights/init/humor/humorAcousticTransformer.pt"))
        hcf_model = Transformer(HCF_DIM, num_layers=3, nhead=2, dim_feedforward=128)
        hcf_model.load_state_dict(torch.load("./model_weights/init/humor/humorHCFTransformer.pt"))
    elif args.dataset == "sarcasm":
        acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=1, nhead=3, dim_feedforward=512)
        acoustic_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmAcousticTransformer.pt"))
        hcf_model = Transformer(HCF_DIM, num_layers=8, nhead=4, dim_feedforward=128)
        hcf_model.load_state_dict(torch.load("./model_weights/init/sarcasm/sarcasmHCFTransformer.pt"))

    text_model = AlbertModel.from_pretrained('albert-base-v2')
    model = HKT_no_visual(text_model, acoustic_model, hcf_model, args)
    model.to(DEVICE)
    loss_fct = BCEWithLogitsLoss()

    acoustic_params, hcf_params, other_params = model.get_params()
    optimizer_o, scheduler_o = get_optimizer_scheduler(other_params, num_training_steps, learning_rate=args.learning_rate)
    optimizer_h, scheduler_h = get_optimizer_scheduler(hcf_params, num_training_steps, learning_rate=args.learning_rate_h)
    optimizer_a, scheduler_a = get_optimizer_scheduler(acoustic_params, num_training_steps, learning_rate=args.learning_rate_a)
    optimizers = [optimizer_o, optimizer_h, optimizer_a]
    schedulers = [scheduler_o, scheduler_h, scheduler_a]

    return model, optimizers, schedulers, loss_fct

def get_optimizer_scheduler(params, num_training_steps, learning_rate=1e-5):
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {"params": [p for n, p in params if not any(nd in n for nd in no_decay)], "weight_decay": 0.01},
        {"params": [p for n, p in params if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(num_training_steps * args.warmup_ratio), num_training_steps=num_training_steps)
    return optimizer, scheduler

def set_random_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

def train(model, train_dataloader, dev_dataloader, test_dataloader, optimizers, schedulers, loss_fct):
    best_valid_loss = float('inf')
    run_name = str(wandb.run.id)
    for epoch_i in range(args.epochs):
        train_loss = train_epoch(model, train_dataloader, optimizers, schedulers, loss_fct)
        valid_loss = eval_epoch(model, dev_dataloader, loss_fct)
        test_accuracy, test_f_score = test_score_model(model, test_dataloader, loss_fct)
        print(f"Epoch {epoch_i}, Train Loss: {train_loss}, Valid Loss: {valid_loss}, Test Acc: {test_accuracy}, F1: {test_f_score}")

        if valid_loss <= best_valid_loss:
            best_valid_loss = valid_loss
            best_valid_test_accuracy = test_accuracy
            best_valid_test_fscore = test_f_score
            if args.save_weight == "True":
                torch.save(model.state_dict(), f"./best_weights/{run_name}.pt")

        wandb.log({
            "train_loss": train_loss,
            "valid_loss": valid_loss,
            "test_acc": test_accuracy,
            "test_f1": test_f_score,
            "best_valid_loss": best_valid_loss,
            "best_valid_test_acc": best_valid_test_accuracy,
            "best_valid_test_fscore": best_valid_test_fscore
        })

def main():
    wandb.init(project="HKT_no_visual")
    wandb.config.update(args)

    seed = args.seed if args.seed != -1 else random.randint(0, 9999)
    wandb.config.update({"seed": seed}, allow_val_change=True)
    set_random_seed(seed)

    train_loader, dev_loader, test_loader = set_up_data_loader()
    num_training_steps = len(train_loader) * args.epochs
    model, optimizers, schedulers, loss_fct = prep_for_training(num_training_steps)
    train(model, train_loader, dev_loader, test_loader, optimizers, schedulers, loss_fct)

if __name__ == '__main__':
    main()

