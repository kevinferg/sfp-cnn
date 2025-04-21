import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Net(torch.nn.Module):
    def __init__(self, dims):
        super(Net, self).__init__()
        self.layers = torch.nn.ModuleList()
        self.dims = dims
        for i in range(len(self.dims)-1):
            self.layers.append(torch.nn.Linear(self.dims[i], self.dims[i+1]))

    def forward(self, x):
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            if i+1 < len(self.layers):
                x = F.relu(x)
        return x

class MultiNet(nn.Module):
    def __init__(self, mlp_dims=[96, 128, 96], num_filters=16, num_layers=4, kernel_size=5, xlim=[0,1], ylim=[0,1]):
        super().__init__()
        self.mlp_dims = mlp_dims
        self.num_filters = num_filters
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.act=torch.relu
        self.xlim = xlim
        self.ylim = ylim

        self.enc = nn.Linear(1, num_filters)

        pool_size = 2
        self.pool = nn.AvgPool2d(pool_size, stride=pool_size)

        self.convs = nn.ModuleList()
        for i in range(num_layers):
            self.convs.append(nn.Conv2d(num_filters, num_filters, kernel_size, padding = int((kernel_size-1)/2)))
        
        self.mlp = Net([num_filters*num_layers, *mlp_dims, 1])

    def forward(self, data):
        xs = (data.x[:,0] - self.xlim[0]) / (self.xlim[1] - self.xlim[0])
        ys = (data.x[:,1] - self.ylim[0]) / (self.ylim[1] - self.ylim[0])
        x = torch.vstack([xs,ys]).T
        
        sdf = torch.permute(data.sdf,[0,3,2,1])
        sdf = self.enc(sdf)
        sdf = torch.permute(sdf,[0,3,2,1])

        locals = []
        for i in range(self.num_layers):
            sdf = self.act(self.convs[i](sdf))
            local = tensor_interp2d(torch.squeeze(sdf), x[:,:2], 0.0001, method="linstep")
            locals.append(local)
            sdf = self.pool(sdf)

        combined = torch.cat([*locals], 1)
        y = self.mlp(combined)

        return y

def smoothstep(a0, a1, w):
    ''' 
    smoothstep: Interpolates between two values/arrays/tensors
    using cubic Hermite interpolation (smoothstep)
    
    a0 - Value(s)
    a1 - Value(s)
    w - Fraction/weight of a0 in the interpolation
    
    Examples:
    w = 0   --> Returns a0
    w = 0.5 --> Returns average of a0 and a1
    w = 1   --> Returns a1
    
    Returns - The interpolated value/array/tensor  
    '''
    return (a1 - a0) * (3.0 - w * 2.0) * w * w + a0

def linstep(a0, a1, w):
    ''' 
    linstep: Linearly interpolates between two values/arrays/tensors
    
    a0 - Value(s)
    a1 - Value(s)
    w - Fraction/weight of a0 in the interpolation
    
    Examples:
    w = 0   --> Returns a0
    w = 0.5 --> Returns average of a0 and a1
    w = 1   --> Returns a1
    
    Returns - The interpolated value/array/tensor  
    '''
    return (a1-a0) * w + a0


def tensor_interp2d(grid, pts, epsilon = 1e-9, method = "smoothstep"):
    ''' 
    tensor_interp: interpolates a PyTorch tensor at the x-y coordinates requested
    
    grid - the array of values to interpolate, first 2 dimensions are for varying x and y,
           the last dimension has the values
    pts - A tensor of the x-y coordinates to get interpolated values at.
          0th dimension is points, 1st dimension is [xval, yval]
          The coordinates should be scaled between [0,0] and [1,1], corresponding to corners of 'grid'
    epsilon - A tolerance for making sure values do not exceed the allowable range
    method - Interpolation method. "linstep" or "smoothstep"
    
    Returns - tensor with number of rows equal to number of points, and columns containing the interpolated values
    '''
    if method == "smoothstep":
        step = smoothstep
    else:
        step = linstep

    pts = pts.view(-1,2)
    pts = pts.clip(min = torch.tensor(epsilon), max = torch.tensor(1 - epsilon))
    size = grid.shape
    grid = torch.transpose(grid,1,2)
    if 2 == len(size):
        grid = grid[None,:,:]
    rows, columns = size[1], size[2]
    
    x, y = ((pts[:,0])*(columns-1)), ((pts[:,1])*(rows-1))
    
    x_f, x_i = torch.frac(x).view(1,-1), torch.floor(x).long()
    y_f, y_i = torch.frac(y).view(1,-1), torch.floor(y).long()

    # Sample the four surrounding corners 
    nw = grid[:, x_i,     y_i    ]
    ne = grid[:, x_i + 1, y_i    ]
    sw = grid[:, x_i,     y_i + 1]
    se = grid[:, x_i + 1, y_i + 1]

    # Interpolate east-west at north and south rows
    north = step(nw, ne, x_f)
    south = step(sw, se, x_f)

    # Interpolate north-south between the two rows
    vals = step(north, south, y_f)

    return torch.transpose(vals, 0, 1)
