import argparse
import numpy as np
import torch
import torch.optim as optim
import model_utils
from tqdm import tqdm
import net
import data_loader as data_loader
import torch.nn as nn


parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', default='data/64x64_SIGNS',
#                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='params.json',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before training")

def evaluate(model, loss_fn, dataloader, params):

    model.eval()
    loss_avg = model_utils.RunningAverage()
    
    for i, (train_batch, labels_batch) in enumerate(dataloader):
        if params.cuda:
            train_batch, labels_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
        output_batch = model(train_batch)
        loss = loss_fn(output_batch, labels_batch)
        loss_avg.update(loss.item())

        return loss_avg()
