import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from convolutional_rnn import Conv2dLSTM
import model_utils
import torch.optim as optim


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
    output = nn.functional.grid_sample(x, vgrid, align_corners=True)
    mask = torch.autograd.Variable(torch.ones(x.size())).cuda()
    mask = nn.functional.grid_sample(mask, vgrid, align_corners=True)
    
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
    
    
class LstmBlock(nn.Module):
    
    def __init__(self, inplanes, planes, stride=1, norm_layer=None):
        super(LstmBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        
        assert inplanes == planes and stride == 1
        self.clstm1 = Conv2dLSTM(inplanes, planes,
                                kernel_size=3, num_layers=1, bias=False)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.clstm2 = Conv2dLSTM(planes, planes,
                                kernel_size=3, num_layers=1, bias=False)
        self.bn2 = norm_layer(planes)
        
    def forward(self, x, h1 = None, h2 = None):
        identity = x

        out, h1 = self.clstm1(x.view(1, *x.size()), h1)
        out = self.bn1(out[0])
        out = self.relu(out)

        out, h2 = self.clstm2(out.view(1, *out.size()), h2)
        out = self.bn2(out[0])

        out += identity
        out = self.relu(out)
        
        return out, h1, h2
    
class BiLstmBlock(nn.Module):
    
    def __init__(self, inplanes, planes, stride=1, norm_layer=None, bidirectional=True):
        super(BiLstmBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
            
        assert stride == 1
        num_directions = 2 if bidirectional else 1
        
        self.clstm1 = Conv2dLSTM(inplanes, planes,
                                kernel_size=3, num_layers=1, bias=False, bidirectional=bidirectional)
        self.bn1 = norm_layer(planes * num_directions)
        self.relu = nn.ReLU(inplace=True)
        self.clstm2 = Conv2dLSTM(planes * num_directions, planes,
                                kernel_size=3, num_layers=1, bias=False, bidirectional=bidirectional)
        self.bn2 = norm_layer(planes * num_directions)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or inplanes != planes * num_directions:
            self.shortcut = nn.Sequential(
                conv1x1(inplanes, planes * num_directions, stride=stride),
                norm_layer(planes * num_directions)
            )
        
    def forward(self, x, h1 = None, h2 = None):
        identity = x

        out, h1 = self.clstm1(x.view(1, *x.size()), h1)
        out = self.bn1(out[0])
        out = self.relu(out)

        out, h2 = self.clstm2(out.view(1, *out.size()), h2)
        out = self.bn2(out[0])

        identity = self.shortcut(x)
        
        out += identity
        out = self.relu(out)
        
        return out, h1, h2
    
## ---------------------- begin Net[P] ---------------------
class MergeNet(nn.Module):

    def __init__(self):
        super(MergeNet, self).__init__()

        self.warp = warp
        
        self.layer0 = LstmBlock(8, 8)

        self.layer1 = BasicBlock(8, 16)
        self.layer2 = BasicBlock(16, 32)
        self.layer3 = BasicBlock(32, 32)
        
        self.layer4 = nn.Conv2d(32, 4, kernel_size=1, stride=1, bias=True)
        
        self.hidden1 = None
        self.hidden2 = None
        
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(mm, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(mm.weight, 1)
                        nn.init.constant_(mm.bias, 0)
            elif isinstance(m, Conv2dLSTM):
                if isinstance(m, Conv2dLSTM):
                    nn.init.kaiming_normal_(m.weight_ih_l0, mode='fan_out', nonlinearity='relu')
                    nn.init.kaiming_normal_(m.weight_hh_l0, mode='fan_out', nonlinearity='relu')
                
    def init_hidden(self):
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, x):

        ref = x[:, :4]
        alt_warp = self.warp(x[:, 4:8], x[:, 8:])
        out = torch.cat([ref, alt_warp], dim = 1)
#         out, self.hidden = self.clstm(out.view(1, *out.size()), self.hidden)
#         out = out[0]
        out, self.hidden1, self.hidden2 = self.layer0(out, self.hidden1, self.hidden2)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out + alt_warp
#         out = torch.sigmoid(out)
        return out

## ---------------------- end Net[P] ---------------------


## ---------------------- begin Net[BP] ---------------------
class MergeNetBP(nn.Module):

    def __init__(self):
        super(MergeNetBP, self).__init__()

        self.warp = warp
        
        self.layer0 = BiLstmBlock(8, 8)

        self.layer1 = BasicBlock(16, 16)
        self.layer2 = BasicBlock(16, 32)
        self.layer3 = BasicBlock(32, 32)
        
        self.layer4 = nn.Conv2d(32, 4, kernel_size=1, stride=1, bias=True)
        
        self.hidden1 = None
        self.hidden2 = None
        
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(mm, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(mm.weight, 1)
                        nn.init.constant_(mm.bias, 0)
            elif isinstance(m, Conv2dLSTM):
                if isinstance(m, Conv2dLSTM):
                    nn.init.kaiming_normal_(m.weight_ih_l0, mode='fan_out', nonlinearity='relu')
                    nn.init.kaiming_normal_(m.weight_hh_l0, mode='fan_out', nonlinearity='relu')
                
    def init_hidden(self):
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, x):

        ref = x[:, :4]
        alt_warp = self.warp(x[:, 4:8], x[:, 8:])
        out = torch.cat([ref, alt_warp], dim = 1)
#         out, self.hidden = self.clstm(out.view(1, *out.size()), self.hidden)
#         out = out[0]
        out, self.hidden1, self.hidden2 = self.layer0(out, self.hidden1, self.hidden2)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = out + alt_warp
#         out = torch.sigmoid(out)
        return out

## ---------------------- end Net[BP] ---------------------


## ---------------------- begin Net[M] ---------------------
class MergeNetM(nn.Module):

    def __init__(self):
        super(MergeNetM, self).__init__()

        self.warp = warp

        self.layer1 = BasicBlock(8, 16)
        self.layer2 = BasicBlock(16, 32)
        self.layer3 = BasicBlock(32, 32)
        self.layer4 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1,bias=True)
        self.hidden = None
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(mm, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(mm.weight, 1)
                        nn.init.constant_(mm.bias, 0)
                
    def init_hidden(self):
        self.hidden = None

    def forward(self, x):

        ref = x[:, :4]
        alt_warp = self.warp(x[:, 4:8], x[:, 8:])
        out = torch.cat([ref, alt_warp], dim = 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.softmax(out, dim = 1)
        out = ref * out[:, :1].repeat(1, 4, 1, 1) + alt_warp * out[:, 1:].repeat(1, 4, 1, 1)
        return out
## ---------------------- end Net[M] ---------------------

## ---------------------- begin Net[MP] ---------------------
class MergeNetMP(nn.Module):

    def __init__(self):
        super(MergeNetMP, self).__init__()

        self.warp = warp

        self.layer1 = BasicBlock(8, 16)
        self.layer2 = BasicBlock(16, 32)
        self.layer3 = BasicBlock(32, 32)
#         self.layer4 = nn.Conv2d(32, 2, kernel_size=1, stride=1, bias=True)
        self.layer4 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1,bias=True)
        self.layer5 = BasicBlock(4, 4)
        self.layer6 = nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1,bias=True)
        self.hidden = None
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(mm, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(mm.weight, 1)
                        nn.init.constant_(mm.bias, 0)
                
    def init_hidden(self):
        self.hidden = None

    def forward(self, x):

        ref = x[:, :4]
        alt_warp = self.warp(x[:, 4:8], x[:, 8:])
        out = torch.cat([ref, alt_warp], dim = 1)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.softmax(out, dim = 1)
        out = ref * out[:, :1].repeat(1, 4, 1, 1) + alt_warp * out[:, 1:].repeat(1, 4, 1, 1)
        #         out = self.layer5(out)
#         out = self.layer6(out)
        return out
## ---------------------- end Net[MP] ---------------------

## ---------------------- begin Net[MBP] ---------------------
class MergeNetMBP(nn.Module):

    def __init__(self):
        super(MergeNetMBP, self).__init__()

        self.warp = warp

        self.layer1 = BiLstmBlock(8, 8)
        self.layer2 = BasicBlock(16, 32)
        self.layer3 = BasicBlock(32, 32)
#         self.layer4 = nn.Conv2d(32, 2, kernel_size=1, stride=1, bias=True)
        self.layer4 = nn.Conv2d(32, 2, kernel_size=3, stride=1, padding=1,bias=True)
        self.hidden1 = None
        self.hidden2 = None
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Sequential):
                for mm in m.modules():
                    if isinstance(mm, nn.Conv2d):
                        nn.init.kaiming_normal_(mm.weight, mode='fan_out', nonlinearity='relu')
                    elif isinstance(mm, (nn.BatchNorm2d, nn.GroupNorm)):
                        nn.init.constant_(mm.weight, 1)
                        nn.init.constant_(mm.bias, 0)
                
    def init_hidden(self):
        self.hidden1 = None
        self.hidden2 = None

    def forward(self, x):

        ref = x[:, :4]
        alt_warp = self.warp(x[:, 4:8], x[:, 8:])
        out = torch.cat([ref, alt_warp], dim = 1)
        bi_out, self.hidden1, self.hidden2 = self.layer1(out, self.hidden1, self.hidden2)
        ref = ref + bi_out[:, :4] + bi_out[:, 8:12]
        alt_warp = alt_warp + bi_out[:, 4:8] + bi_out[:, 12:]
        out = self.layer2(bi_out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.softmax(out, dim = 1)
        out = ref * out[:, :1].repeat(1, 4, 1, 1) + alt_warp * out[:, 1:].repeat(1, 4, 1, 1)
        return out
## ---------------------- end Net[MBP] ---------------------



## ---------------------- begin Net[S] ---------------------

class MergeNetS(nn.Module):

    def __init__(self):
        super(MergeNetS, self).__init__()

        self.warp = warp
        #self.clstm = Conv2dLSTM(in_channels=8, out_channels=8,
                              #kernel_size=5, num_layers=1, bias=False)

        self.conv1 = nn.Conv2d(8, 4, 15, padding = 7, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(8, 4, 15, padding = 7, bias=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(8, 1, 1, bias=True)
        
        self.load_state_dict(torch.load('SimpleMergeNet.pth'))
        
        self.hidden = None
                
    def init_hidden(self):
        self.hidden = None

    def forward(self, x):

        ref = x[:, :4]
        alt = self.warp(x[:, 4:8], x[:, 8:])
        out = torch.cat([ref, alt], dim = 1)

        out1 = self.relu1(self.conv1(out))
        out2 = self.relu2(self.conv2(out))
        out = torch.cat([out1, out2], dim = 1)
        out = self.conv3(out)
        out = torch.sigmoid(out)
        out = out.repeat(1, 4, 1, 1)
        out = out * ref + (1 - out) * alt
        
        return out

## ---------------------- end Net[S] ---------------------


if __name__ == '__main__':
    model = MergeNetMBP().cuda()
    print(model)
    model_utils.print_model_params(model)
    
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 0.001, momentum=0.9)
    
    for epoch in range(3):
        print("test epoch", epoch)
        gt = torch.rand(2, 4, 256, 256).cuda()
        pred = torch.rand(2, 4, 256, 256).cuda()
        optimizer.zero_grad()
        for t in range(7):
            print("test seq", t)
            inputs = torch.rand(2, 10, 256, 256).cuda()
            pred += model(inputs)
        loss = criterion(pred, gt)
        loss.backward()
        optimizer.step()
        model.init_hidden()
    print("Finish Training.") 
            

            
        
