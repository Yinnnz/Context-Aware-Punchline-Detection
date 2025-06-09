from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
from global_config import *
import math
from torch.nn.utils.rnn import pad_sequence
import copy

import numpy as np
from sklearn.metrics.pairwise import cosine_distances
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from transformers import (
    AlbertModel,
    AlbertPreTrainedModel,
    AlbertConfig,
    load_tf_weights_in_albert,
)
from transformers.modeling_albert import AlbertEmbeddings, AlbertLayerGroup


class Transformer(nn.Module):
    def __init__(self, d_model, num_layers=1, nhead=1, dropout=0.1, dim_feedforward=128, max_seq_length=5000):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(max_seq_length, d_model)
        self.encoder = TransformerEncoder(TransformerLayer(d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout), num_layers=num_layers)
        self.decoder = nn.Linear(d_model, 1)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input, attention_mask=None):
        seq_length = input.size()[1]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input.device)
        positions_embedding = self.pos_encoder(position_ids).unsqueeze(0).expand(input.size())
        input = input + positions_embedding
        input = self.norm(input)
        hidden = self.encoder(input, attention_mask=attention_mask)
        out = self.decoder(hidden)
        out = (out[:,0,:], out, hidden)
        return out


class TransformerLayer(nn.Module):
    def __init__(self, hidden_size, nhead=1, dim_feedforward=128, dropout=0.1):
        super(TransformerLayer, self).__init__()
        self.self_attention = Attention(hidden_size, nhead, dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size, dim_feedforward), nn.ReLU(), nn.Linear(dim_feedforward, hidden_size))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, attention_mask=None):
        src_1 = self.self_attention(src, src, attention_mask=attention_mask)
        src = src + self.dropout1(src_1)
        src = self.norm1(src)
        src_2 = self.fc(src)
        src = src + self.dropout2(src_2)
        src = self.norm2(src)
        return src


class TransformerEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)

    def forward(self, src, attention_mask=None):
        for layer in self.layers:
            new_src = layer(src, attention_mask=attention_mask)
            src = src + new_src
        return src


class Attention(nn.Module):
    def __init__(self, hidden_size, num_attention_heads, attention_probs_dropout_prob, ctx_dim=None):
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("The hidden size (%d) is not a multiple of the number of attention heads (%d)" % (hidden_size, num_attention_heads))
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = int(hidden_size / num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        if ctx_dim is None:
            ctx_dim = hidden_size
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(ctx_dim, self.all_head_size)
        self.value = nn.Linear(ctx_dim, self.all_head_size)

        self.dropout = nn.Dropout(attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, context, attention_mask=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(context)
        mixed_value_layer = self.value(context)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, context_size, nhead=1, dropout=0.1):
        super(CrossAttentionLayer, self).__init__()
        self.src_cross_attention = Attention(hidden_size, nhead, dropout, ctx_dim=context_size)
        self.context_cross_attention = Attention(context_size, nhead, dropout, ctx_dim=hidden_size)
        self.self_attention = Attention(hidden_size + context_size, nhead, dropout)
        self.fc = nn.Sequential(nn.Linear(hidden_size + context_size, hidden_size + context_size), nn.ReLU())
        self.norm1 = nn.LayerNorm(hidden_size + context_size)
        self.norm2 = nn.LayerNorm(hidden_size + context_size)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, context, attention_mask=None):
        new_src = self.src_cross_attention(src, context, attention_mask=attention_mask)
        new_context = self.context_cross_attention(context, src, attention_mask=attention_mask)
        cross_src = torch.cat((new_src, new_context), dim=2)
        cross_src_1 = self.self_attention(cross_src, cross_src, attention_mask)
        cross_src = cross_src + self.dropout1(cross_src_1)
        cross_src = self.norm1(cross_src)
        cross_src_2 = self.fc(cross_src)
        cross_src = cross_src + self.dropout2(cross_src_2)
        cross_src = self.norm2(cross_src)
        return cross_src


class CrossAttentionEncoder(nn.Module):
    def __init__(self, layer, num_layers):
        super(CrossAttentionEncoder, self).__init__()
        self.layers = _get_clones(layer, num_layers)

    def forward(self, src, context, attention_mask=None):
        src_dim = src.size()[2]
        context_dim = context.size()[2]
        for layer in self.layers:
            output = layer(src, context, attention_mask=attention_mask)
            new_src = output[:,:,0:src_dim]
            new_context = output[:,:,src_dim:src_dim+context_dim]
            src = src + new_src
            context = context + new_context
        return output


class HKT_no_visual(nn.Module):
    def __init__(self, text_model, acoustic_model, hcf_model, args, dropout=0.1):
        super(HKT_no_visual, self).__init__()
        self.args = args
        self.text_model = text_model
        self.acoustic_model = acoustic_model
        self.hcf_model = hcf_model
        self.L_AH = CrossAttentionLayer(LANGUAGE_DIM + HCF_DIM, ACOUSTIC_DIM, nhead=args.cross_n_heads, dropout=args.dropout)
        fusion_input_dim = 2 * (LANGUAGE_DIM + HCF_DIM + ACOUSTIC_DIM)
        self.fc = nn.Sequential(
            nn.Linear(fusion_input_dim, args.fusion_dim),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(args.fusion_dim, 1)
        )

    def get_params(self):
        acoustic_params = list(self.acoustic_model.named_parameters())
        hcf_params = list(self.hcf_model.named_parameters())
        other_params = list(self.text_model.named_parameters()) + \
                       list(self.L_AH.named_parameters()) + \
                       list(self.fc.named_parameters())
        return acoustic_params, hcf_params, other_params

    def forward(self, input_ids, acoustic, hcf, attention_mask=None, token_type_ids=None):
        text_output, _ = self.text_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        _, _, acoustic_output = self.acoustic_model(acoustic)
        _, _, hcf_output = self.hcf_model(hcf)
        text_hcf = torch.cat((text_output, hcf_output), dim=2)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=text_hcf.dtype)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        fusion_context = self.L_AH(text_hcf, acoustic_output, attention_mask=extended_attention_mask)
        text_embedding = text_hcf[:, 0, :]
        acoustic_embedding = F.max_pool1d(acoustic_output.permute(0, 2, 1).contiguous(), acoustic_output.shape[1]).squeeze(-1)
        fusion_embedding = F.max_pool1d(fusion_context.permute(0, 2, 1).contiguous(), fusion_context.shape[1]).squeeze(-1)
        fused = torch.cat((text_embedding, acoustic_embedding, fusion_embedding), dim=1)
        out = self.fc(fused)
        return (out, fused)


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

