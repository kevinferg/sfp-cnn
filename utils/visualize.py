import numpy as np
import matplotlib.pyplot as plt
import torch
from evaluate import *

def plot_graph(xy, edges, node_color='black', edge_color=None, node_size = 100, 
               color_bounds = None, label = None, linewidth = 1, colorbar=True):
    ''' 
    plot_graph: Plots a 2D graph's nodes and edges on the current axes
    
    Args:
    xy - Two-column array with coordinates: [x, y]
    edges - Array of node index pairs for each 1-directional edge
    node_color - Color string, array, or rgb triple, defaults to 'black'
    edge_color - Color string or rgb triple for each edge (None: no edges)
    node_size - Size of each node, defaults to 100
    color_bounds - [value of lowest color, value of highest color], when node_color is an array
    label - Name of the plot for use in a legend (optional)
    
    Returns:
    - handle of node scatter plot
    '''

    x = xy[:,0]
    y = xy[:,1]


    if edge_color is not None:
        edges = edges[:,edges[0,:] < edges[1,:]]
        for edge in edges.T:
            plt.plot([x[edge[0]],x[edge[1]]], [y[edge[0]],y[edge[1]]], c=edge_color, alpha=.5, zorder=0, linewidth=linewidth)
    
    if label is not None:
        title_height = 0.88
        fontsize = 12
        plt.title(label, fontsize=fontsize, y=title_height)

    if type(node_color) == str:
        handle = plt.scatter(x,y, s=node_size, c=node_color, zorder=1, label=label, cmap='jet')
    else:
        if color_bounds is None:
            tick_min = np.round(np.min(node_color),3)
            tick_max = np.round(np.max(node_color),3)
        elif len(color_bounds) == 1:
            tick_min = color_bounds[0]
            tick_max = np.round(np.max(node_color),3)
        else:
            tick_min = color_bounds[0]
            tick_max = color_bounds[1]
        cb = dict(vmin=tick_min, vmax=tick_max)
        tick_min = cb["vmin"]
        tick_max = cb["vmax"]
        tick_med = np.round((tick_min + tick_max)/2,3)
        handle = plt.scatter(x,y, s=node_size, c=node_color, zorder=1, label=label, cmap='jet', **cb)

        if colorbar:
            cbar_shrink = 0.9
            cbar_pad = -0.1
            bar = plt.colorbar(shrink=cbar_shrink, location='bottom', pad=cbar_pad, ticks=[tick_min, tick_med, tick_max])
            bar.ax.set_xticklabels([tick_min, tick_med, tick_max])

    plt.axis("equal")
    plt.axis("off")
    return handle

def plot_data(data, colors="black", show_edges=False, color_bounds=None, label = None, size=30, width=1, colorbar=True):
    xy = data.x[:,:2].detach().numpy()
    edges = data.edge_index.detach().numpy()
    node_color = colors.detach().numpy() if type(colors) == torch.Tensor else colors
    edge_color = "black" if show_edges else None
    handle = plot_graph(xy, edges, node_color=node_color, edge_color=edge_color, node_size=size, 
                        color_bounds=color_bounds, label=label, linewidth=width, colorbar=colorbar)
    return handle

def plot_comparison(model, data, filename=None, dpi=300):
    plt.figure(figsize=(16,5), dpi=dpi)
    size=15

    plt.subplot(1,4,1)
    plot_data(data, data.y, show_edges=False, size=size, label="Ground Truth", color_bounds=[0,])

    plt.subplot(1,4,2)
    plot_data(data, model(data), show_edges=False, size=size, label="Prediction", color_bounds=[0,])

    plt.subplot(1,4,3)
    plot_data(data, torch.abs(model(data) - data.y), show_edges=False, size=size, label="Absolute Error", color_bounds=[0,])

    ax = plt.subplot(1,4,4)
    plot_model_r2(model,data)
    plt.axis("scaled")

    plt.tight_layout()
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.15, hspace=None)

    pos1 = ax.get_position() # get the original position 
    pos2 = [pos1.x0 + 0.01, pos1.y0 + 0.06,  pos1.width * 0.9, pos1.height * 0.9] 
    ax.set_position(pos2) # set a new position

    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()