import numpy as np
import torch
import torch.optim as optim
import utils
from tqdm import tqdm
import net

def train(model, optimizer, loss_fn, dataloader, params):

    model.train()
    loss_avg = utils.RunningAverage()
    
    with tqdm(total=len(dataloader)) as t:
        for i, (train_batch, labels_batch) in enumerate(dataloader):
            if params.cuda:
                train_batch, label_batch = train_batch.cuda(), labels_batch.cuda()
            output_batch = model(train_batch)
            loss = loss_fn(output_batch, labels_batch)

            optimizer.zero_grad()
            loss.backward()

            optimizer.step()

            loss_avg.update(loss.item())

            t.set_postfix(loss='{:05.3f}'.format(loss_avg()))
            t.update()

if __name__=='__main__':
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(38)
    if params.cuda:
        torch.cuda.manual_seed(38)

    dataloader = None

    model = net.Net().cuda() if params.cuda else net.Net()

    optimizer = optim.Adam(model.parameters(), lr = params.learning_rate)

    loss_fn = None

    train(model, optimizer, loss_fn, dataloader, params)
    
    
