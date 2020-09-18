import argparse
import numpy as np
import torch
import torch.optim as optim
import model_utils
from tqdm import tqdm
import net
import data_loader as data_loader
import torch.nn as nn
import torchvision


parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', default='data/64x64_SIGNS',
#                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='params.json',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default='best',
                    help="Optional, name of the file in --model_dir containing weights to reload before training")

def evaluate(epoch, model, writer, loss_fn, dataloader, params):

    model.eval()
    loss_avg = model_utils.RunningAverage()
    with torch.no_grad():
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)
            loss_avg.update(loss.item())

    writer.add_scalar('Stats/val_loss', loss_avg(), epoch)

    ref_grid = torchvision.utils.make_grid(train_batch[:, :1])
    writer.add_image('Visual/ref', ref_grid, epoch)
    out_grid = torchvision.utils.make_grid(output_batch)
    writer.add_image('Visual/out', out_grid, epoch)
    gt_grid = torchvision.utils.make_grid(labels_batch)
    writer.add_image('Visual/gt', gt_grid, epoch)
#     display_grid = torchvision.utils.make_grid(torch.cat([train_batch[:, :1], output_batch, labels_batch]))
#     writer.add_image('Visual/ref-out-gt', display_grid, epoch)
    
    return loss_avg()
