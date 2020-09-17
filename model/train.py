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

parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', default='data/64x64_SIGNS',
#                    help="Directory containing the dataset")
parser.add_argument('--model_dir', default='.',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', default=None,
                    help="Optional, name of the file in --model_dir containing weights to reload before training")

def train(model, optimizer, loss_fn, dataloader, params):

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

    
def train_and_evaluate(model, train_dataloader, val_dataloader, optimizer, loss_fn, params, model_dir, restore_file=None):

    if restore_file is not None:
        restore_path = os.path.join(
            args.model_dir, args.restore_file + '.pth.tar')
        model_utils.load_checkpoint(restore_path, model, optimizer)

    best_val_loss = 1e10

    for epoch in range(params.num_epochs):

        train(model, optimizer, loss_fn, train_dataloader, params)

        val_loss = evaluate(model, loss_fn, val_dataloader, params)

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

    dataloaders = data_loader.fetch_dataloader(params)
    dataloader = dataloaders['train']

    model = net.Net().cuda() if params.cuda else net.Net()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = params.learning_rate)

    loss_fn = nn.MSELoss()
    
    for epoch in range(2):
        train(model, optimizer, loss_fn, dataloader, params)

