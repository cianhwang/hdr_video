from pwc import Network as PWC
from merge import MergeNet, MergeNetM, MergeNetS, MergeNetMP, MergeNetBP, MergeNetMBP
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from model_utils import *
import torch.optim as optim
sys.path.append('raft_core')
from raft import RAFT

class Net(nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()

#         self.pwc_net = PWC()
        self.raft = RAFT(args)
        for name, param in self.raft.named_parameters():
            param.requires_grad = False
            
        self.merge_ver = args.merge_ver
        if self.merge_ver == 'p':
            self.merge_net = MergeNet()
        elif self.merge_ver == 'm':
            self.merge_net = MergeNetM()
        elif self.merge_ver == 'mp':
            self.merge_net = MergeNetMP()
        elif self.merge_ver == 's':
            self.merge_net = MergeNetS()
        elif self.merge_ver == 'bp':
            self.merge_net = MergeNetBP()
        else:
            raise KeyError("MergeNet verion [{}] not found.".format(self.merge_ver))

    
    def raw_to_stack_tensor(self, raw_tensor):
        batch_size, C, H, W = raw_tensor.size()
        assert C == 1
        return F.unfold(raw_tensor, 2, stride=2).view(-1, 4, H//2, W//2)

    def stack_to_raw_tensor(self, stack_tensor):
        batch_size, C, H, W = stack_tensor.size()
        assert C == 4
        return F.pixel_shuffle(stack_tensor, 2)
    
    def init_hidden(self):
        self.merge_net.init_hidden()

        
    def forward(self, x):
        ## x: RGGB raw input, (batch_size, 2, H, W)
        batch_size, _, H, W = x.size()
        gray_img1, gray_img2 = (F.avg_pool2d(x[:, :1], 2), F.avg_pool2d(x[:, 1:], 2))
        #gray_img1 = F.interpolate(gray_img1, (448, 1024), mode='bilinear')/20.0
        #gray_img2 = F.interpolate(gray_img2, (448, 1024), mode='bilinear')/20.0
        _, flow = self.raft(gray_img1.repeat(1, 3, 1, 1), gray_img2.repeat(1, 3, 1, 1))
        #flow = 20.0 * F.interpolate(flow, (H//2, W//2), mode='nearest')
        #flow[:, 0] *= float(H//2) / float(448)
        #flow[:, 1] *= float(W//2) / float(1024)

        x1 = (self.raw_to_stack_tensor(x[:, :1].clone()))
        x2 = (self.raw_to_stack_tensor(x[:, 1:].clone()))

        x = torch.cat([x1, x2, flow], dim = 1)
        x = self.merge_net(x)
        x = self.stack_to_raw_tensor(x)
        return x

if __name__ == '__main__':
    
    model = Net().cuda()
    print(model)
    print_model_params(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    for epoch in range(2):        
        print("test epoch", epoch)
        gt = torch.rand(2, 1, 512, 512).cuda()
        pred = torch.rand(2, 1, 512, 512).cuda()
        optimizer.zero_grad()
        for t in range(7):
            print("test seq", t)
            inputs = torch.rand(2, 2, 512, 512).cuda()
            pred += model(inputs)
        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()
        model.init_hidden()
    print("Finish Training.")
    

    
    
        
