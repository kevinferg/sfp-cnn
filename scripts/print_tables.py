import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../utils')

import numpy as np
import torch

from load_data import *
from training import *
from evaluate import *
from cnn_model import *

def get_datasets(dset):
    scale = 1 if dset=="temp" else 10000
    datasets_vor = load_tr_te_od_data(f"../data/{dset}_vor_w.mat", f"../data/{dset}_vor_o.mat", scale=scale)
    datasets_lat = load_tr_te_od_data(f"../data/{dset}_lat_w.mat", f"../data/{dset}_lat_o.mat", scale=scale)
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]
    return datasets_vor, datasets_lat, datasets

def print_all_median_r2s(vals_vor, vals_lat, vals_both, filename):
    table_string = r"""
    \begin{tabular}{V{2} c V{2} c | c | c V{2}}\Xhline{2\arrayrulewidth}
    \multirow{2}{*}{Dataset} & \multicolumn{3}{c V{2} }{Median $R^2$}\\\cline{2-4}
        & Training & Testing & Out-of-Distribution\\\Xhline{2\arrayrulewidth}
        Voronoi Set & %0.3f & %0.3f & %0.3f\\
        Lattice Set & %0.3f & %0.3f & %0.3f\\
        Combined Set & %0.3f & %0.3f & %0.3f\\\Xhline{2\arrayrulewidth}
    \end{tabular}
    """
    r2s = []
    for vals in [vals_vor, vals_lat, vals_both]:
        for key in ["tr","te","od"]:
            med_r2 = np.median(vals[key])
            r2s.append(med_r2)

    with open(filename, "w") as file:
        print(table_string %tuple(r2s), file=file)

def generate_r2_table(dset):
    scale = 1 if dset=="temp" else 10000
    datasets_vor = load_tr_te_od_data(f"../data/{dset}_vor_w.mat", f"../data/{dset}_vor_o.mat", scale=scale)
    datasets_lat = load_tr_te_od_data(f"../data/{dset}_lat_w.mat", f"../data/{dset}_lat_o.mat", scale=scale)
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]

    if dset == "temp":
        dset_prefix = "temp_"
    else:
        dset_prefix = ""
    model = torch.load(f"../models/{dset_prefix}multi_model_6.pth")
    model_vor = torch.load(f"../models/{dset_prefix}vor_model.pth")
    model_lat = torch.load(f"../models/{dset_prefix}lat_model.pth")

    vals = eval_model_multiple(model, datasets)
    vals_vor = eval_model_multiple(model_vor, datasets_vor)
    vals_lat = eval_model_multiple(model_lat, datasets_lat)
    print_all_median_r2s(vals_vor, vals_lat, vals, f"../figures/r2_table_{dset}.txt")

def generate_parametric_table(filename):
    table_string = r"""
    \begin{tabular}{V{3} c V{2} c | c V{3} c | c | c V{3}}\Xhline{3\arrayrulewidth}
    \multicolumn{3}{V{3} c V{3} }{ Model Information } & \multicolumn{3}{c V{3} }{Median $R^2$}\\\Xhline{2\arrayrulewidth}
    Pooling Layers & Parameters & Training Time & Training & Testing & Out-of-Distribution\\\Xhline{2\arrayrulewidth}
        %d & %d & %0.1f min. & %0.3f & %0.3f & %0.3f\\
        %d & %d & %0.1f min. & %0.3f & %0.3f & %0.3f\\
        %d & %d & %0.1f min. & %0.3f & %0.3f & %0.3f\\
        %d & %d & %0.1f min. & %0.3f & %0.3f & %0.3f\\
        %d & %d & %0.1f min. & %0.3f & %0.3f & %0.3f\\
        %d & %d & %0.1f min. & %0.3f & %0.3f & %0.3f\\
        \Xhline{3\arrayrulewidth}
    \end{tabular}
    """

    _, _, datasets = get_datasets("stress")
    layer_counts = np.arange(1, 6 + 1)
    r2s = dict(tr=[],te=[],od=[])
    params = []
    for i in layer_counts:
        model = torch.load(f"../models/multi_model_{i}.pth")
        vals = eval_model_multiple(model, datasets)
        for key in r2s:
            r2s[key].append(np.median(vals[key]))
        params.append(sum(p.numel() for p in model.parameters() if p.requires_grad))
    
    # For now, hard-code these because it is easier than parsing the text files I stored them in
    times = [21.7, 33.2, 41.5, 50.8, 61.2, 68.1]

    entries = []
    for i in range(6):
        entries.append(i+1)
        entries.append(params[i])
        entries.append(times[i])
        entries.append(r2s["tr"][i])
        entries.append(r2s["te"][i])
        entries.append(r2s["od"][i])

    with open(filename, "w") as file:
        print(table_string %tuple(entries), file=file)


def generate_less_data_table(filename):
    table_string = r"""
    \begin{tabular}{V{2} c V{2} c | c | c V{2}}\Xhline{2\arrayrulewidth}
        \multirow{2}{*}{Size of} & \multicolumn{3}{c V{2} }{Median $R^2$}\\\cline{2-4}
    Training Set & Training & Testing & Out-of-Dist.\\\Xhline{2\arrayrulewidth}
        %d & %0.3f & %0.3f & %0.3f\\
        %d & %0.3f & %0.3f & %0.3f\\
        %d & %0.3f & %0.3f & %0.3f\\
        %d & %0.3f & %0.3f & %0.3f\\
        %d & %0.3f & %0.3f & %0.3f\\
        %d & %0.3f & %0.3f & %0.3f\\
        \Xhline{3\arrayrulewidth}
    \end{tabular}
    """

    Ns = [50,100,200,400,800]
    r2s = dict(tr=[],te=[],od=[])
    params = []
    for i in Ns:
        model = torch.load(f"../models/small_model_{i}.pth")
        vals = model.r2s
        for key in r2s:
            r2s[key].append(np.median(vals[key]))
    
    # For now, hard-code these because it is easier than parsing the text files I stored them in
    # times = [1.7, 3.5, 7.4, 15.6, 34.1]

    entries = []
    for i in range(len(Ns)):
        entries.append(Ns[i])
        entries.append(r2s["tr"][i])
        entries.append(r2s["te"][i])
        entries.append(r2s["od"][i])

    # Hard-code results for model with 1600 training points
    entries = entries + [1600,0.925,0.911,0.881]

    with open(filename, "w") as file:
        print(table_string %tuple(entries), file=file)

def generate_unet_table(filename):
    table_string = r"""
    \begin{tabular}{V{3} c V{3} c V{3} c | c | c V{3}}\Xhline{3\arrayrulewidth}
    \multirow{2}{*}{Field} & \multirow{2}{*}{Model} & \multicolumn{3}{c V{3} }{Median $R^2$}\\\cline{3-5}
        & & Training & Testing & Out-of-Distribution\\\Xhline{3\arrayrulewidth}
        \multirow{2}{*}{Stress} & Interpolated U-Net & %0.3f & %0.3f & %0.3f\\
        & \textbf{Interpolated Multi-Resolution CNN} & %0.3f & %0.3f & %0.3f\\\hline
        \multirow{2}{*}{Temperature} & Interpolated U-Net & %0.3f & %0.3f & %0.3f\\
        & \textbf{Interpolated Multi-Resolution CNN} & %0.3f & %0.3f & %0.3f\\\Xhline{3\arrayrulewidth}
    \end{tabular}
    """
    entries = []

    _, _, datasets = get_datasets("stress")
    
    model = torch.load(f"../models/stress_unet.pth")
    vals = eval_model_multiple(model, datasets)
    for key in ["tr", "te", "od"]:
        entries.append(np.median(vals[key]))

    model = torch.load(f"../models/multi_model_6.pth")
    vals = eval_model_multiple(model, datasets)
    for key in ["tr", "te", "od"]:
        entries.append(np.median(vals[key]))


    _, _, datasets = get_datasets("temp")
    
    model = torch.load(f"../models/temp_unet.pth")
    vals = eval_model_multiple(model, datasets)
    for key in ["tr", "te", "od"]:
        entries.append(np.median(vals[key]))

    model = torch.load(f"../models/temp_multi_model_6.pth")
    vals = eval_model_multiple(model, datasets)
    for key in ["tr", "te", "od"]:
        entries.append(np.median(vals[key]))


    with open(filename, "w") as file:
        print(table_string %tuple(entries), file=file)

def generate_r2_table_unet(dset):
    scale = 1 if dset=="temp" else 10000
    datasets_vor = load_tr_te_od_data(f"../data/{dset}_vor_w.mat", f"../data/{dset}_vor_o.mat", scale=scale)
    datasets_lat = load_tr_te_od_data(f"../data/{dset}_lat_w.mat", f"../data/{dset}_lat_o.mat", scale=scale)
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]

    model = torch.load(f"../models/{dset}_combined_unet.pth")
    model_vor = torch.load(f"../models/{dset}_vor_unet.pth")
    model_lat = torch.load(f"../models/{dset}_lat_unet.pth")

    vals = eval_model_multiple(model, datasets)
    vals_vor = eval_model_multiple(model_vor, datasets_vor)
    vals_lat = eval_model_multiple(model_lat, datasets_lat)
    print_all_median_r2s(vals_vor, vals_lat, vals, f"../figures/r2_table_{dset}_unet.txt")


if __name__ == "__main__":
    generate_r2_table("stress")
    generate_r2_table("temp")
    generate_parametric_table("../figures/param_table.txt")
    generate_less_data_table("../figures/less_data.txt")
    generate_unet_table("../figures/unet_table.txt")
    generate_r2_table_unet("stress")