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

DIR = "../figures/"
EXT = ".png"
DPI = 300

def to_path(name):
    return DIR + name + EXT

def get_datasets(dset):
    scale = 1 if dset=="temp" else 10000
    datasets_vor = load_tr_te_od_data(f"../data/{dset}_vor_w.mat", f"../data/{dset}_vor_o.mat", scale=scale)
    datasets_lat = load_tr_te_od_data(f"../data/{dset}_lat_w.mat", f"../data/{dset}_lat_o.mat", scale=scale)
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]
    return datasets_vor, datasets_lat, datasets

def plot_stress_visualizations(output_file="visualize-stress"):
    output_file = to_path(output_file)
    print(f"Creating stress visualization figure: {output_file}")
    if os.path.exists(output_file):
        print("Figure already exists. Skipping.")
        return

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
       plot_comparison(model, datasets["te"][order[rank]], filename=filename, dpi=DPI)
       files.append(filename)

    stack_images(files, output_file)
    for file in files:
        os.remove(file)

def plot_temperature_visualizations(output_file="visualize-temperature"):
    output_file = to_path(output_file)
    print(f"Creating temperature visualization figure: {output_file}")
    if os.path.exists(output_file):
        print("Figure already exists. Skipping.")
        return

    model = torch.load("../models/temp_multi_model_6.pth")
    _, _, datasets = get_datasets("temp")
    vals = eval_model_all(model, datasets["te"])
    order = np.argsort(vals)
    N = len(vals)
    # Near-median ranks hard-coded to give 1 voronoi & 1 lattice:
    ranks = [-1, N//2 + 2, N//2, 0] 
    files = []
    for rank in ranks:
       filename = f"tmp-{order[rank]}.png"
       plot_comparison(model, datasets["te"][order[rank]], filename=filename, dpi=DPI)
       files.append(filename)

    stack_images(files, output_file)
    for file in files:
        os.remove(file)

def plot_r2_figures():
    _, _, datasets = get_datasets("stress")
    layer_counts = np.arange(1, 6 + 1)
    r2s = dict(tr=[],te=[],od=[])
    for i in layer_counts:
        model = torch.load(f"../models/multi_model_{i}.pth")
        vals = eval_model_multiple(model, datasets)
        for key in r2s:
            r2s[key].append(np.median(vals[key]))

    # vals now contains r2s for 6-layer network
    plot_boxes(vals, filename=to_path("box"))
    plot_violins(vals, filename=to_path("violin"))

    plt.figure(figsize=[7,5], dpi=DPI)
    plt.title("Model Performance by Layer Count")
    plt.plot(layer_counts, r2s["tr"], "o-", label="Training")
    plt.plot(layer_counts, r2s["te"], "o-", label="Testing")
    plt.plot(layer_counts, r2s["od"], "o-", label="Out-of-Distribution")

    plt.xlabel("Layers")
    plt.ylabel("Median $R^2$")
    plt.legend(loc="lower right")
    plt.savefig(to_path("parametric-study"), bbox_inches="tight")

if __name__ == "__main__":
    plot_stress_visualizations()
    plot_temperature_visualizations()
    plot_r2_figures()
