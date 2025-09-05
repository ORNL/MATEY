#SSIM functions modified from https://github.com/jinh0park/pytorch-ssim-3D
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import sys
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window_3D(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t())
    _3D_window = _1D_window.mm(_2D_window.reshape(1, -1)).reshape(window_size, window_size, window_size).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_3D_window.expand(channel, 1, window_size, window_size, window_size).contiguous())
    return window

def _ssim_3D(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv3d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv3d(img2, window, padding = window_size//2, groups = channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)

    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv3d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv3d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv3d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2


    C1 = (0.1)**2
    C2 = (0.3)**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

class SSIM3D(torch.nn.Module):
    def __init__(self, window_size = 9, size_average = True):
        super(SSIM3D, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window_3D(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window_3D(self.window_size, channel)
            
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim_3D(img1, img2, window, self.window_size, channel, self.size_average)


def ssim3D(img1, img2, window_size = 9, size_average = True):
    (_, channel, _, _, _) = img1.size()
    window = create_window_3D(window_size, channel)
    
    window = window.type_as(img1)
    
    return _ssim_3D(img1, img2, window, window_size, channel, size_average)

def remove_edges(arr):
    return arr[:,:,1:-1,1:-1,1:-1]

def torch_dx(phi,h):
    assert len(phi.shape) == 5
    batch_size = phi.shape[0]
    h = h.reshape(-1)
    my_list = []
    for i in range(batch_size):
        my_list.append(torch.gradient(phi[i:i+1], spacing=h[i],dim=2,edge_order=1)[0])
        assert len(my_list[-1].shape) == 5
    t = torch.cat(my_list,0)

    return t

def torch_dy(phi,h):
    assert len(phi.shape) == 5
    batch_size = phi.shape[0]
    h = h.reshape(-1)
    my_list = []
    for i in range(batch_size):
        my_list.append(torch.gradient(phi[i:i+1], spacing=h[i],dim=3,edge_order=1)[0])
        assert len(my_list[-1].shape) == 5
    t = torch.cat(my_list,0)

    return t

def torch_dz(phi,h):
    assert len(phi.shape) == 5
    batch_size = phi.shape[0]
    h = h.reshape(-1)
    my_list = []
    for i in range(batch_size):
        my_list.append(torch.gradient(phi[i:i+1], spacing=h[i],dim=4,edge_order=1)[0])
        assert len(my_list[-1].shape) == 5
    t = torch.cat(my_list,0)

    return t

    
def torch_diff(phi, dx,dy,dz): # x is a 3D tensor (batch, channel, x, y, z)
    diff_x = torch_dx(phi, dx)
    diff_y = torch_dy(phi, dy)
    diff_z = torch_dz(phi, dz)
    return diff_x, diff_y, diff_z


class GradLoss(nn.Module):
    def __init__(self):
        super(GradLoss, self).__init__()

    def forward(self, input,target):
        nbatch = input.shape[0]
        #just use dx=1 for everything
        dx = torch.ones(nbatch,device=input.device)
        dy = torch.ones(nbatch,device=input.device)
        dz = torch.ones(nbatch,device=input.device)

        ri = input[:,0:1,:,:,:]
        ui = input[:,1:2,:,:,:]
        vi = input[:,2:3,:,:,:]
        wi = input[:,3:4,:,:,:]

        rt = target[:,0:1,:,:,:]
        ut = target[:,1:2,:,:,:]
        vt = target[:,2:3,:,:,:]
        wt = target[:,3:4,:,:,:]

        dxri = torch_dx(ri,dx)
        dyri = torch_dy(ri,dy)
        dzri = torch_dz(ri,dz)
        dxui = torch_dx(ui,dx)
        dyui = torch_dy(ui,dy)
        dzui = torch_dz(ui,dz)
        dxvi = torch_dx(vi,dx)
        dyvi = torch_dy(vi,dy)
        dzvi = torch_dz(vi,dz)
        dxwi = torch_dx(wi,dx)
        dywi = torch_dy(wi,dy)
        dzwi = torch_dz(wi,dz)

        dxrt = torch_dx(rt,dx)
        dyrt = torch_dy(rt,dy)
        dzrt = torch_dz(rt,dz)
        dxut = torch_dx(ut,dx)
        dyut = torch_dy(ut,dy)
        dzut = torch_dz(ut,dz)
        dxvt = torch_dx(vt,dx)
        dyvt = torch_dy(vt,dy)
        dzvt = torch_dz(vt,dz)
        dxwt = torch_dx(wt,dx)
        dywt = torch_dy(wt,dy)
        dzwt = torch_dz(wt,dz)

        total_loss = F.mse_loss(dxri,dxrt) + F.mse_loss(dyri,dyrt) + F.mse_loss(dzri,dzrt) + \
                        F.mse_loss(dxui,dxut) + F.mse_loss(dyui,dyut) + F.mse_loss(dzui,dzut) + \
                        F.mse_loss(dxvi,dxvt) + F.mse_loss(dyvi,dyvt) + F.mse_loss(dzvi,dzvt) + \
                        F.mse_loss(dxwi,dxwt) + F.mse_loss(dywi,dywt) + F.mse_loss(dzwi,dzwt)
        
        return total_loss