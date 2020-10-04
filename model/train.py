import argparse
import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import model_utils
from tqdm import tqdm
import net
import data_loader as data_loader
import torch.nn as nn
import os
from evaluate import evaluate
from tensorboardX import SummaryWriter
import time
import torchvision
import json

parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', default='data/64x64_SIGNS',
#                    help="Directory containing the dataset")
parser.add_argument('--note', type=str, default=None, help='note wrote to logs')
parser.add_argument('--merge_ver', type=str, 
                    default='m',
                    help='Load assigned MergeNet version. Available types: [p]: MergeNet; [m]: MergeNetM; [mp]: MergeNetMP; [s]: MergeNetS')
parser.add_argument('--ckpt_dir', type=str, 
                    default='ckpt/'+time.strftime("%m%d_%H_%M"),
                    help='Directory in which to save model checkpoints')
parser.add_argument('--logs_dir', type=str,
                    default='runs/'+time.strftime("%m%d_%H_%M"),
                    help='Directory in which Tensorboard logs wil be stored')
parser.add_argument('--model_dir', type=str, default='.',
                    help="Directory containing params.json")
parser.add_argument('--restore_file', type=str, default=None,
                    help="Optional, name of the file in --ckpt_dir containing weights to reload before training")

## RAFT args
parser.add_argument('--small', action='store_true', help='use small model')
parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')

def train(epoch, model, writer, optimizer, loss_fn, dataloader, params):

    model.train()
    loss_avg = model_utils.RunningAverage()
    
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            n_seq = train_batch.size(1)
            output_batch = train_batch[:, 0:1].clone()
            if params.cuda:
                output_batch = output_batch.cuda(non_blocking=True)
            for j in range(1, n_seq):
                input_batch = torch.cat([train_batch[:, 0:1], train_batch[:, j:j+1]], dim = 1)
                if params.cuda:
                    input_batch, labels_batch = input_batch.cuda(non_blocking=True), labels_batch.cuda(non_blocking=True)
                res = model(input_batch)
#                 output_batch = res
                output_batch += res.clone()
#            ######
#             output_batch = torch.sigmoid(output_batch)
            output_batch = output_batch/float(n_seq)
            loss = loss_fn(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:.5f}'.format(loss_avg()))
            t.update()
            
            model.init_hidden()

        writer.add_scalar('Stats/loss', loss_avg(), epoch +1)
        for n, p in model.named_parameters():
            if(p.requires_grad) and ("bias" not in n) and ("bn" not in n):
                writer.add_histogram('hist/'+n, p, epoch +1)
        
        ref_grid = torchvision.utils.make_grid(train_batch[:4, :1])
        writer.add_image('Train/ref', ref_grid, epoch + 1)
        out_grid = torchvision.utils.make_grid(output_batch[:4])
        writer.add_image('Train/out', out_grid, epoch + 1)
        gt_grid = torchvision.utils.make_grid(labels_batch[:4])
        writer.add_image('Train/gt', gt_grid, epoch + 1)

    return loss_avg()
    
def train_and_evaluate(model, writer, train_dataloader, val_dataloader, optimizer, loss_fn, params, ckpt_dir, restore_file=None):

    start_epoch = 0
    best_val_loss = 1e10
    
    if restore_file is not None:
        restore_path = os.path.join(args.ckpt_dir, args.restore_file + '.pth.tar')
        model, optimizer, start_epoch, best_val_loss = model_utils.load_checkpoint(restore_path, model, optimizer)

#     scheduler = MultiStepLR(optimizer, milestones=[2, 5], gamma=.99)

    for epoch in range(start_epoch, params.num_epochs):

        print('\nEpoch: {}/{}'.format(epoch+1, params.num_epochs))

        train_loss = train(epoch, model, writer, optimizer, loss_fn, train_dataloader, params)

        val_loss = evaluate(epoch, model, writer, loss_fn, val_dataloader, params)
        
#         scheduler.step()

        is_best = val_loss <= best_val_loss

        msg1 = "train loss: {:.5f}"
        msg2 = " - val loss: {:.5f}"
        if is_best:
            msg2 += " [*]"
        msg = msg1 + msg2
        print(msg.format(train_loss, val_loss))

        model_utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optim_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss},
                                    is_best = is_best,
                                    checkpoint = ckpt_dir
        )
        if is_best:
            best_val_loss = val_loss            

if __name__=='__main__':

    args = parser.parse_args()
    with open('args.json', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = model_utils.Params(json_path)
    params.cuda = torch.cuda.is_available()
    torch.manual_seed(38)
    if params.cuda:
        torch.cuda.manual_seed(38)

    print('[*] Saving tensorboard logs to {}'.format(args.logs_dir))
    if not os.path.exists(args.logs_dir):
        os.makedirs(args.logs_dir)
    writer = SummaryWriter(args.logs_dir)
    logs_path = os.path.join(args.logs_dir, 'logs.json')
    with open(logs_path, 'w') as f:
        json.dump(args.__dict__, f, indent=4)
        json.dump(params.__dict__, f, indent=4)
    
        
    dataloaders = data_loader.fetch_dataloader(params)
    train_dl = dataloaders['train']
    val_dl = dataloaders['val']

    model = net.Net(args).cuda() if params.cuda else net.Net(args)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr = params.learning_rate)

    loss_fn = nn.L1Loss()
    
    train_and_evaluate(model, writer, train_dl, val_dl, optimizer, loss_fn, params, args.ckpt_dir, args.restore_file)
