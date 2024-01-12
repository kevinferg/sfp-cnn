import torch
from torch import nn
from cnn_model import tensor_interp2d

def get_conv_layer(n_in, n_mid, n_out):
    seq = nn.Sequential(nn.Conv2d(n_in, n_mid, 3, padding = 1),
                        nn.ReLU(), nn.BatchNorm2d(n_mid),
                        nn.Conv2d(n_mid, n_out, 3, padding = 1),
                        nn.ReLU(), nn.BatchNorm2d(n_out))
    return seq

class UNet(nn.Module):
    def __init__(self, n_in, filter_sizes, n_out):
        super(UNet, self).__init__()
        self.n_in = n_in
        self.filter_sizes= filter_sizes
        self.n_out = n_out

        self.dn_convs = nn.ModuleList()
        self.up_convs = nn.ModuleList()

        for i in range(len(filter_sizes)-1):
            self.dn_convs.append(get_conv_layer(n_in if i==0 else filter_sizes[i-1], filter_sizes[i], filter_sizes[i]))
            self.up_convs.append(get_conv_layer(2*filter_sizes[i], filter_sizes[i], filter_sizes[i] if i==0 else filter_sizes[i-1]))

        self.mid_conv = get_conv_layer(filter_sizes[-2], filter_sizes[-1], filter_sizes[-2])
        self.out_conv = nn.Conv2d(filter_sizes[0], n_out, 3, padding = 1)
        
        self.dn_pool = nn.MaxPool2d(2, stride = 2, return_indices = True)
        self.up_pool = nn.MaxUnpool2d(2, stride = 2)
    
    def forward(self, x):
        indices = []
        xs = []
        
        # Left-hand (encoder) side of U-Net.
        for i, conv in enumerate(self.dn_convs):
            x = conv(x)
            xs.append(x)
            x, idx = self.dn_pool(x)
            indices.append(idx)
        
        # Middle/bottom of U-Net
        x = self.mid_conv(x)
        
        # Right-hand (decoder) side of U-Net.
        # Note up-conv list is in reverse order
        for i, conv in enumerate(self.up_convs[::-1]): 
            x = self.up_pool(x, indices[-i-1])
            x = torch.cat((xs[-i-1], x),1)
            x = conv(x)

        # Final convolution to output size
        x = self.out_conv(x)
        
        return x
    
class InterpolatedUNet(nn.Module):
    def __init__(self, n_in, filter_sizes, n_out, xlim=[0,1],ylim=[0,1],method="linstep"):
        super(InterpolatedUNet, self).__init__()
        self.unet = UNet(n_in, filter_sizes, n_out)
        self.method = method
        self.xlim = xlim
        self.ylim = ylim
    def forward(self, data):
        xs = (data.x[:,0] - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        ys = (data.x[:,1] - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        pts = torch.vstack([xs,ys]).T
        sdf = data.sdf
        y_img = self.unet(sdf)
        y = tensor_interp2d(torch.squeeze(y_img,0), pts, 0.0001, method=self.method)
        return y