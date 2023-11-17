import torch
import numpy as np
import matplotlib.pyplot as plt

def get_r2(a,b):
    ''' 
    get_r2: Computes an R-squared value to evaluate the goodness
    of fit between two nd-arrays
    
    a - The ground-truth data
    b - The predicted data
    
    Returns
    - The R2 value
    '''
    N = len(a)
    SS_tot = np.sum((b-np.mean(b))**2)
    SS_res = np.sum((a-b)**2)
    R2 = 1-SS_res/SS_tot
    return R2

def plot_model_r2(model, data, lims=None, size=1):
    pred = model(data).detach().numpy()
    gt = data.y.detach().numpy()
    plot_r2(gt, pred, lims=lims, size=size)

def plot_r2(a, b, lims=None, size=1):
    if lims is None:
        vmin = np.min(np.concatenate((a,b)))
        vmax = np.max(np.concatenate((a,b)))
        lims = [vmin, vmax]

    y = x = np.array([lims[0], lims[-1]])

    plt.plot(x,y,"r-")
    plt.scatter(a, b, s=size, c='k')
    
    r2 = get_r2(a, b)
    plt.title(r"$R^2$ = " + f"{np.round(r2, 3)}")
    plt.xlabel("Ground Truth")
    plt.ylabel("Prediction")
    plt.xlim(x)
    plt.ylim(y)

def eval_model(model, data):
    ''' 
    evaluate_model: Runs a model and computes the R-squared value on a single data point
    
    model - The model to evaluate
    data - the data point
    
    Returns
    - The R2 value from using 'model' to predict the field in 'data'
    '''

    pred = model(data).detach().numpy().flatten()
    gt = data.y.detach().numpy().flatten()

    return get_r2(gt, pred)

def eval_model_all(model, dataset):
    vals = []
    for data in dataset:
        vals.append(eval_model(model, data))
    vals = np.array(vals)
    return vals

def eval_model_multiple(model, datasets):
    vals = dict()
    for key in datasets:
        vals[key] = eval_model_all(model, datasets[key])
    return vals

def plot_boxes(vals, titles=dict(tr="Training", te="Testing", od="Out-of-Dist."), lims = [-0.25, 1], filename = None, dpi=175):
    n = len(vals)
    plt.figure(figsize=(2*n, 3.4), dpi=dpi)

    vals_list = []
    ticklocs = []
    ticklabels = []

    for i, key in enumerate(vals):
        vals_list.append(vals[key])
        ticklocs.append(i)
        ticklabels.append(f"{titles[key]}, N={len(vals[key])}")

    plt.boxplot(vals_list, positions=ticklocs)
    plt.xticks(ticklocs, ticklabels)
    plt.ylabel('R-Squared')
    plt.ylim(lims)
    plt.xlim(ticklocs[0]-0.5, ticklocs[-1]+0.5)
    plt.plot([ticklocs[0]-0.5, ticklocs[-1]+0.5], [0, 0],'k-', linewidth=0.5, zorder=-1)

    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()

def plot_violins(vals, titles=dict(tr="Training", te="Testing", od="Out-of-Dist."), lims = [-0.25, 1], filename = None, dpi=175):
    n = len(vals)
    plt.figure(figsize=(2*n, 3.4), dpi=dpi)

    vals_list = []
    ticklocs = []
    ticklabels = []

    for i, key in enumerate(vals):
        vals_list.append(vals[key])
        ticklocs.append(i)
        ticklabels.append(f"{titles[key]}, N={len(vals[key])}")

    plt.violinplot(vals_list, positions=ticklocs)
    plt.xticks(ticklocs, ticklabels)
    plt.ylabel('R-Squared')
    plt.ylim(lims)
    plt.xlim(ticklocs[0]-0.5, ticklocs[-1]+0.5)
    plt.plot([ticklocs[0]-0.5, ticklocs[-1]+0.5], [0, 0],'k-', linewidth=0.5, zorder=-1)

    if filename is not None:
        plt.savefig(filename, bbox_inches = "tight")
        plt.close()
    else:
        plt.show()