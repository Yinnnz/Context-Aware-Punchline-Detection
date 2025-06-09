import os
import random
import pickle
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import AlbertTokenizer, AlbertModel
from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from models import HKT_no_visual, Transformer
from global_config import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="HKT")
parser.add_argument("--dataset", type=str, choices=["humor", "sarcasm"], default="humor")
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--max_seq_length", type=int, default=85)
parser.add_argument("--cross_n_layers", type=int, default=1)
parser.add_argument("--cross_n_heads", type=int, default=4)
parser.add_argument("--fusion_dim", type=int, default=172)
parser.add_argument("--dropout", type=float, default=0.2366)
args = parser.parse_args()

class InputFeatures(object):
    def __init__(self, input_ids, input_mask, segment_ids, acoustic, hcf, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.acoustic = acoustic
        self.hcf = hcf
        self.label_id = label_id

def get_inversion(tokens, marker="▁"):
    idx, inv = -1, []
    for t in tokens:
        if marker in t:
            idx += 1
        inv.append(idx)
    return inv

def _truncate(tokens_a, tokens_b, max_length):
    while len(tokens_a) + len(tokens_b) > max_length:
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop(0)
        else:
            tokens_b.pop()
    return tokens_a, tokens_b

def convert_to_features(data, tokenizer):
    features = []
    for (p_words, _, p_acoustic, p_hcf), (c_words, _, c_acoustic, c_hcf), _, label in data:
        text_a = ". ".join(c_words)
        text_b = p_words + "."

        tokens_a = tokenizer.tokenize(text_a)
        tokens_b = tokenizer.tokenize(text_b)
        tokens_a, tokens_b = _truncate(tokens_a, tokens_b, args.max_seq_length - 3)

        inv_a = get_inversion(tokens_a)
        inv_b = get_inversion(tokens_b)

        acoustic_a = [c_acoustic[i].reshape(1, -1) for i in inv_a]
        acoustic_b = [p_acoustic[i].reshape(1, -1) for i in inv_b]
        hcf_a = [c_hcf[i].reshape(1, -1) for i in inv_a]
        hcf_b = [p_hcf[i].reshape(1, -1) for i in inv_b]

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

def load_dataset():
    with open(os.path.join(DATASET_LOCATION, args.dataset, "ur_funny.pkl"), "rb") as f:
        all_data = pickle.load(f)
    test_data = all_data["test"]
    tokenizer = AlbertTokenizer.from_pretrained("albert-base-v2")
    features = convert_to_features(test_data, tokenizer)

    return DataLoader(
        TensorDataset(
            torch.tensor([f.input_ids for f in features], dtype=torch.long),
            torch.tensor([f.input_mask for f in features], dtype=torch.long),
            torch.tensor([f.segment_ids for f in features], dtype=torch.long),
            torch.tensor([f.acoustic for f in features], dtype=torch.float),
            torch.tensor([f.hcf for f in features], dtype=torch.float),
            torch.tensor([f.label_id for f in features], dtype=torch.float),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )

def build_model():
    acoustic_model = Transformer(ACOUSTIC_DIM, num_layers=8, nhead=3, dim_feedforward=256)
    acoustic_model.load_state_dict(torch.load("./model_weights/init/humor/humorAcousticTransformer.pt"))

    hcf_model = Transformer(HCF_DIM, num_layers=3, nhead=2, dim_feedforward=128)
    hcf_model.load_state_dict(torch.load("./model_weights/init/humor/humorHCFTransformer.pt"))

    text_model = AlbertModel.from_pretrained("albert-base-v2")
    model = HKT_no_visual(text_model, acoustic_model, hcf_model, args)
    model.load_state_dict(torch.load("./best_weights/bestweights.pt"))
    return model.to(DEVICE)

def evaluate(model, dataloader):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, input_mask, segment_ids, acoustic, hcf, label_ids = (t.to(DEVICE) for t in batch)
            logits = model(input_ids, acoustic, hcf, token_type_ids=segment_ids, attention_mask=input_mask)[0]
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.extend(probs)
            labels.extend(label_ids.cpu().numpy())

    preds = np.round(np.array(preds)).squeeze()
    labels = np.array(labels).squeeze()
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="weighted")
    print(f"Accuracy: {acc:.4f}, F1-score: {f1:.4f}")

if __name__ == "__main__":
    dataloader = load_dataset()
    model = build_model()
    evaluate(model, dataloader)



