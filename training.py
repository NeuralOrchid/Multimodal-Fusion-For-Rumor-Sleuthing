import gdown
from tqdm.notebook import tqdm_notebook as tqdm
import os.path as osp
from typing import Literal
import random
from random import shuffle

import pandas as pd
import numpy as np
import copy
import math

from torch_geometric.data import Dataset, download_url, Data, Batch
from torch_geometric.loader import DataLoader
from torch_geometric.sampler import SamplerOutput

import tiktoken

import torch
from torch import nn
from torch.nn import functional as F
from torcheval.metrics.functional import multiclass_f1_score

from torch_scatter import scatter_mean
from torch_geometric.nn import GCNConv

from sklearn.model_selection import train_test_split, StratifiedKFold

from dataset.dataset import TwitterDataset
from model.bcfn import RumorSleuthNet
from config import *


@torch.no_grad()
def validate_fn(model, loop):
    model.eval()
    temp_val_losses = []
    temp_val_accs = []
    for Batch_data in loop:
        Batch_data.to(DEVICE)

        val_out = model(Batch_data)
        val_loss  = F.cross_entropy(val_out, Batch_data.y)
        temp_val_losses.append(val_loss.item())

        _, val_pred = val_out.max(dim=1)
        val_acc = multiclass_f1_score(val_pred, Batch_data.y, num_classes=4)
        temp_val_accs.append(val_acc.cpu())

        loop.set_postfix(Val_Loss=f"{np.mean(temp_val_losses):.4f}", Val_Accuracy=f"{np.mean(temp_val_accs):.4f}")
    return np.mean(temp_val_losses), np.mean(temp_val_accs)


def train_fn(
        model,
        training_loader,
        validation_loader,
        opt,
        epoch,
        ):
    ## Training Loop ##
    model.train()
    training_loop = tqdm(training_loader, leave=False, desc=f"Training[{epoch}/{EPOCH}]")
    for Batch_data in training_loop:
        Batch_data.to(DEVICE)
        out_labels = model(Batch_data)
        finalloss = F.cross_entropy(out_labels, Batch_data.y)
        loss = finalloss

        opt.zero_grad()
        loss.backward()
        opt.step()

        _, pred = out_labels.max(dim=-1)
        train_acc = multiclass_f1_score(pred, Batch_data.y, num_classes=4)

        training_loop.set_postfix(
            Train_Loss=f"{loss.item():.4f}",
            Train_Accuracy=f"{train_acc:.4f}",
            )


    ## Validating Loop ##
    if (epoch+1)%5 == 0:
        validating_loop = tqdm(validation_loader, leave=False, desc="Validating")
        val_losse, val_acc = validate_fn(model, validating_loop)

        print(f"""
        Train Loss:             {loss.item():.4f}
        Train Accuracy:         {train_acc:.4f}
        Validation Loss:        {val_losse:.4f}
        Validation Accuracy:    {val_acc:.4f}
        """)


dataloader = DataLoader(
    TwitterDataset(),
    batch_size=BATCH,
    num_workers=2
)

model = RumorSleuthNet().to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

if LOAD_MODEL:
    load_checkpoint(model)

for epoch in range(EPOCH):
    train_fn(
        model=model,
        training_loader=dataloader,
        validation_loader=dataloader,
        opt=optimizer,
        epoch=epoch,
    )

if SAVE_MODEL:
    save_checkpoint(model)