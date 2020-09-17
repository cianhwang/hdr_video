from pwc import Network as PWC
from merge import MergeNet
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from model_utils import *
import torch.optim as optim

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.pwc_net = PWC()
        for name, param in self.pwc_net.named_parameters():
            param.requires_grad = False
        self.merge_net = MergeNet()

    
    def raw_to_stack_tensor(self, raw_tensor):
        batch_size, C, H, W = raw_tensor.size()
        assert C == 1
        return F.unfold(raw_tensor, 2, stride=2).view(-1, 4, H//2, W//2)

    def stack_to_raw_tensor(self, stack_tensor):
        batch_size, C, H, W = stack_tensor.size()
        assert C == 4
        return F.pixel_shuffle(stack_tensor, 2)

        
    def forward(self, x):
        ## x: RGGB raw input, (batch_size, 2, H, W)
        batch_size, _, H, W = x.size()
        gray_img1, gray_img2 = (F.avg_pool2d(x[:, :1], 2), F.avg_pool2d(x[:, 1:], 2))
        #gray_img1 = F.interpolate(gray_img1, (448, 1024), mode='bilinear')/20.0
        #gray_img2 = F.interpolate(gray_img2, (448, 1024), mode='bilinear')/20.0
        flow = self.pwc_net(gray_img1.repeat(1, 3, 1, 1), gray_img2.repeat(1, 3, 1, 1))
        flow = 20.0 * F.interpolate(flow, (H//2, W//2), mode='nearest')
        #flow[:, 0] *= float(H//2) / float(448)
        #flow[:, 1] *= float(W//2) / float(1024)

        x1 = (self.raw_to_stack_tensor(x[:, :1].clone()))
        x2 = (self.raw_to_stack_tensor(x[:, 1:].clone()))

        x = torch.cat([x1, x2, flow], dim = 1)
        x = self.merge_net(x)
        x = self.stack_to_raw_tensor(x)
        return x

if __name__ == '__main__':
    x = torch.rand(1, 2, 1024, 1024).cuda()
    y = torch.rand(1, 1, 1024, 1024).cuda()
    model = Net().cuda()
    print(model)
    print_model_params(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    for epoch in range(2):        
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()
    print("Finish Training.")
    

    
    
        
