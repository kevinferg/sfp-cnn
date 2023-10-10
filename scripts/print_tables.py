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

if __name__ == "__main__":
    generate_r2_table("stress")
    generate_r2_table("temp")
