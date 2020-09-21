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
        for i, (val_batch, labels_batch) in enumerate(dataloader):
            n_seq = val_batch.size(1)
            output_batch = val_batch[:, 0:1].clone()
            if params.cuda:
                output_batch = output_batch.cuda(non_blocking=True)
            for j in range(1, n_seq):
                input_batch = torch.cat([val_batch[:, 0:1], val_batch[:, j:j+1]], dim = 1)
                if params.cuda:
                    input_batch, labels_batch = input_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
                res = model(input_batch)
#                 output_batch = res
                output_batch += res.clone()
#             #########
#             output_batch = torch.sigmoid(output_batch)
            output_batch = output_batch/float(n_seq)
            loss = loss_fn(output_batch, labels_batch)
            loss_avg.update(loss.item())
            model.init_hidden()

    writer.add_scalar('Stats/val_loss', loss_avg(), epoch)

    ref_grid = torchvision.utils.make_grid(val_batch[:, :1])
    writer.add_image('Val/ref', ref_grid, epoch)
    out_grid = torchvision.utils.make_grid(output_batch)
    writer.add_image('Val/out', out_grid, epoch)
    gt_grid = torchvision.utils.make_grid(labels_batch)
    writer.add_image('Val/gt', gt_grid, epoch)
#     display_grid = torchvision.utils.make_grid(torch.cat([val_batch[:, :1], output_batch, labels_batch]))
#     writer.add_image('Visual/ref-out-gt', display_grid, epoch)
    
    return loss_avg()
