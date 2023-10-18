import numpy as np
import matplotlib.pyplot as plt
import torch
from evaluate import *
import matplotlib.image as mpimg

def plot_graph(xy, node_color='black', node_size = 100, 
               color_bounds = None, label = None, linewidth = 1, colorbar=True, cmap="jet"):
    ''' 
    plot_graph: Plots a 2D graph's nodes and edges on the current axes
    
    Args:
    xy - Two-column array with coordinates: [x, y]
    node_color - Color string, array, or rgb triple, defaults to 'black'
    node_size - Size of each node, defaults to 100
    color_bounds - [value of lowest color, value of highest color], when node_color is an array
    label - Name of the plot for use in a legend (optional)
    
    Returns:
    - handle of node scatter plot
    '''

    x = xy[:,0]
    y = xy[:,1]
    
    if label is not None:
        title_height = 0.96
        fontsize = 12
        plt.title(label, fontsize=fontsize, y=title_height)

    if type(node_color) == str:
        handle = plt.scatter(x,y, s=node_size, c=node_color, zorder=1, label=label, cmap=cmap)
    else:
        if color_bounds is None:
            tick_min = np.round(np.min(node_color),3)
            tick_max = np.round(np.max(node_color),3)
        elif len(color_bounds) == 1:
            tick_min = np.round(color_bounds[0],3)
            tick_max = np.round(np.max(node_color),3)
        else:
            tick_min = np.round(color_bounds[0],3)
            tick_max = np.round(color_bounds[1],3)
        cb = dict(vmin=tick_min, vmax=tick_max)
        tick_min = cb["vmin"]
        tick_max = cb["vmax"]
        tick_med = np.round((tick_min + tick_max)/2,3)
        handle = plt.scatter(x,y, s=node_size, c=node_color, zorder=1, label=label, cmap=cmap, **cb)

        if colorbar:
            cbar_shrink = 0.93
            cbar_pad = -0.02
            bar = plt.colorbar(shrink=cbar_shrink, location='bottom', pad=cbar_pad, ticks=[tick_min, tick_med, tick_max])
            bar.ax.set_xticklabels([tick_min, tick_med, tick_max])

    plt.axis("equal")
    plt.axis("off")
    return handle

def plot_data(data, colors="black", color_bounds=None, label = None, size=30, width=1, colorbar=True, cmap="jet"):
    xy = data.x[:,:2].detach().numpy()
    node_color = colors.detach().numpy() if type(colors) == torch.Tensor else colors
    handle = plot_graph(xy, node_color=node_color, node_size=size, 
                        color_bounds=color_bounds, label=label, linewidth=width, colorbar=colorbar, cmap=cmap)
    return handle

def plot_comparison(model, shapes, filename=None, dpi=300, size=15):
    if type(shapes) != list and type(shapes) != tuple:
        shapes = [shapes,]
    N = len(shapes)
    plt.figure(figsize=(16, 4.6*N), dpi=dpi)
    rhs_axes = []

    for i, data in enumerate(shapes):
        gt = data.y
        pred = model(data)
        s = size / (1 + 7*(3000<data.x.shape[0]))
        maxval = max(torch.max(gt).detach().numpy(), torch.max(pred).detach().numpy())
        maxerr = np.max(torch.abs(gt-pred).detach().numpy())

        plt.subplot(N,4,1+i*4)
        plot_data(data, data.y, size=s, label="Ground Truth", color_bounds=[0,maxval])

        plt.subplot(N,4,2+i*4)
        plot_data(data, model(data), size=s, label="Prediction", color_bounds=[0,maxval])

        plt.subplot(N,4,3+i*4)
        plot_data(data, pred - gt, size=s, label="Prediction $-$ Ground Truth", color_bounds=[-maxerr, maxerr], cmap="bwr")

        ax = plt.subplot(N,4,4+i*4)
        plot_model_r2(model, data)
        plt.axis("scaled")
        rhs_axes.append(ax)

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.14, hspace=0)
    for ax in rhs_axes:
        pos1 = ax.get_position() # get the original position 
        pos2 = [pos1.x0 + 0.01, pos1.y0 + 0.1/N,  pos1.width * 0.9, pos1.height * 0.9] 
        ax.set_position(pos2) # set a new position

    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()

def stack_images(image_paths, output_path):
    images = []
    height = 0
    width = 0
    depth = 3
    for path in image_paths:
        image = mpimg.imread(path)
        if image.dtype == np.float32:
            image = (image*255).astype(np.uint8)
        images.append(image)
        height = height + image.shape[0]
        width = max(width, image.shape[1])
        depth = max(depth, image.shape[2])

    new_image = np.zeros((height, width, depth), dtype=np.uint8)

    row = 0
    for image in images:
        new_image[row:(row+image.shape[0]), :image.shape[1], :image.shape[2]] = image
        row += image.shape[0]

    plt.imsave(output_path, new_image)
