import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class MergeLayerC(nn.Module):

    def __init__(self):
        super(MergeLayerC, self).__init__()
        self.conv0 = nn.Conv2d(8, 4, 15, padding = 7, bias=False)
        
        self.conv1 = nn.Conv2d(4, 4, 3, stride = 2, bias=False) #padding left to F.pad
        self.conv2 = nn.Conv2d(4, 4, 3, stride = 2, bias=False)
        self.conv3 = nn.Conv2d(4, 4, 3, stride = 2, bias=False)
        self.upconv1 = nn.ConvTranspose2d(4, 4, 2, stride = 2, bias=False) 
        self.upconv2 = nn.ConvTranspose2d(4, 4, 2, stride = 2, bias=False) 
        self.upconv3 = nn.ConvTranspose2d(4, 4, 2, stride = 2, bias=False) 
        
    def gauss_pyramid(self, x):
        G = x.clone()
        gp = [G]
        x = self.conv1(F.pad(x, (0, 1, 0, 1), mode = 'reflect'))
        gp.append(x)
        x = self.conv2(F.pad(x, (0, 1, 0, 1), mode = 'reflect'))
        gp.append(x)
        x = self.conv3(F.pad(x, (0, 1, 0, 1), mode = 'reflect'))
        gp.append(x)
        
        return gp
    
    def lap_pyramid(self, x):
        gp = self.gauss_pyramid(x)
        lp = [gp[-1]]
        GE = self.upconv1(gp[-1])
        L = gp[-2] - GE
        lp.append(L)
        GE = self.upconv2(gp[-2])
        L = gp[-3] - GE
        lp.append(L)
        GE = self.upconv3(gp[-3])
        L = gp[-4] - GE
        lp.append(L)
        return lp

    def warp(self, x, flo):
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

        # if W==128:
            # np.save('mask.npy', mask.cpu().data.numpy())
            # np.save('warp.npy', output.cpu().data.numpy())
        
        mask[mask<0.9999] = 0
        mask[mask>0] = 1
        
        return output*mask
        
    
    def forward(self, x):
        ## x size: (batch_size, 4(rggb)*2 + 2(optical flow), H//2, W//2)
        x = torch.cat([x[:, :4], self.warp(x[:, 4:8], x[:, 8:])], dim = 1)
        mm = F.relu(torch.cat([self.conv0(x), -self.conv0(x)], dim = 1))
        mm = torch.sum(mm, dim = 1, keepdim=True)
        mask = torch.sigmoid(mm.repeat(1, 4, 1, 1) - 700.0)
        ref = x[:, :4]
        alt = x[:, 4:]
        lp_ref = self.lap_pyramid(ref)
        lp_alt = self.lap_pyramid(alt)
        gp_mask = [mask, F.max_pool2d(mask, 2), F.max_pool2d(F.max_pool2d(mask, 2), 2), 
                   F.max_pool2d(F.max_pool2d(F.max_pool2d(mask, 2), 2), 2)]#self.gauss_pyramid(mask)
        
        LS = []
        for la,lb,mask in zip(lp_ref,lp_alt, gp_mask[::-1]):
            ls = la * mask + lb * (1.0 - mask)
            LS.append(ls)
        
        ls_ = LS[0]
        ls_ = self.upconv1(ls_)
        ls_ = ls_ + LS[1]
        ls_ = self.upconv2(ls_)
        ls_ = ls_ + LS[2]
        ls_ = self.upconv3(ls_)
        ls_ = ls_ + LS[3]
        
        return ls_

if __name__=='__main__':
    model = MergeLayerC()
    model.load_state_dict(torch.load('model.pth'))
    print(model)
