import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def conv3x3(in_planes, out_planes, stride=1, dilation=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride = stride, padding=dilation, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):

    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def warp(x, flo):
    """
    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = x.size()
    # mesh grid 
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = Variable(grid) + flo
        
    # scale grid to [-1,1] 
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:].clone() / max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:].clone() / max(H-1,1)-1.0
    
    vgrid = vgrid.permute(0,2,3,1)        
    output = nn.functional.grid_sample(x, vgrid)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid)
    
    mask[mask<0.9999] = 0
    mask[mask>0] = 1
    
    return output*mask


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes:
            self.shortcut = nn.Sequential(
                conv1x1(inplanes, planes, stride=stride),
                norm_layer(planes)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        identity = self.shortcut(x)

        out += identity
        out = self.relu(out)

        return out
    

class MergeNet(nn.Module):

    def __init__(self):
        super(MergeNet, self).__init__()

        self.warp = warp
        self.layer1 = BasicBlock(8, 16)
        self.layer2 = BasicBlock(16, 32)
        self.layer3 = BasicBlock(32, 32)
        self.layer4 = conv1x1(32, 4)

    def forward(self, x):

        ref = x[:, :4]
        out = torch.cat([x[:, :4], self.warp(x[:, 4:8], x[:, 8:])], dim = 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out += ref
        return out


if __name__ == '__main__':
    import model_utils
    model = MergeNet()
    print(model)
    model_utils.print_model_params(model)
