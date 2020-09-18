import argparse
import numpy as np
import torch
import torch.optim as optim
import model_utils
from tqdm import tqdm
import net
import data_loader as data_loader
import torch.nn as nn
import os
from evaluate import evaluate
from tensorboardX import SummaryWriter
import time

parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', default='data/64x64_SIGNS',
#                    help="Directory containing the dataset")
parser.add_argument('--logs_dir', type=str,
                    default='runs/'+time.strftime("%m%d_%H_%M"),
                    help='Directory in which Tensorboard logs wil be stored')
parser.add_argument('--model_dir', default='.',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")

def train(epoch, model, writer, optimizer, loss_fn, dataloader, params):

    model.train()
    loss_avg = model_utils.RunningAverage()
    
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, labels_batch = train_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

        writer.add_scalar('Stats/loss', loss_avg(), epoch)
        for n, p in model.named_parameters():
            if(p.requires_grad) and ("bias" not in n) and ("bn" not in n):
                writer.add_histogram('hist/'+n, p, epoch)
    
def train_and_evaluate(model, writer, train_dataloader, val_dataloader, optimizer, loss_fn, params, model_dir, restore_file=None):

    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        model_utils.load_checkpoint(restore_path, model, optimizer)

    best_val_loss = 1e10

    for epoch in range(params.num_epochs):

        train(epoch, model, writer, optimizer, loss_fn, train_dataloader, params)

        val_loss = evaluate(epoch, model, writer, loss_fn, val_dataloader, params)

        is_best = val_loss <= best_val_loss

        model_utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict()},
                                    is_best = is_best,
                                    checkpoint = model_dir
        )
        if is_best:
            best_val_loss = val_loss            

if __name__=='__main__':

    args = parser.parse_args()
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = model_utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    torch.manual_seed(38)
    if params.cuda:
        torch.cuda.manual_seed(38)

    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    writer = SummaryWriter(args.logs_dir)
        
    dataloaders = data_loader.fetch_dataloader(params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    model = net.Net().cuda() if params.cuda else net.Net()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = params.learning_rate)

    loss_fn = nn.L1Loss()
    
    train_and_evaluate(model, writer, train_dl, val_dl, optimizer, loss_fn, params, args.model_dir, args.restore_file)
