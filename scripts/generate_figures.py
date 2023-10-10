import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../utils')

import numpy as np
import matplotlib.pyplot as plt
import torch

from load_data import *
from training import *
from evaluate import *
from visualize import *
from cnn_model import *

def get_datasets(dset):
    scale = 1 if dset=="temp" else 10000
    datasets_vor = load_tr_te_od_data(f"../data/{dset}_vor_w.mat", f"../data/{dset}_vor_o.mat", scale=scale)
    datasets_lat = load_tr_te_od_data(f"../data/{dset}_lat_w.mat", f"../data/{dset}_lat_o.mat", scale=scale)
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]
    return datasets_vor, datasets_lat, datasets

def plot_stress_visualizations(output_file="../figures/visualize-stress.png"):
    model = torch.load("../models/multi_model_6.pth")
    _, _, datasets = get_datasets("stress")
    vals = eval_model_all(model, datasets["te"])
    order = np.argsort(vals)
    N = len(vals)
    ##Near-median ranks hard-coded to give 1 voronoi & 1 lattice:
    ranks = [-1, N//2 + 2, N//2, 0] 
    files = []
    for rank in ranks:
       filename = f"tmp-{order[rank]}.png"
       plot_comparison(model, datasets["te"][order[rank]], filename=filename)
       files.append(filename)

    stack_images(files, output_file)
    
    for file in files:
        os.remove(file)

if __name__ == "__main__":
    plot_stress_visualizations()