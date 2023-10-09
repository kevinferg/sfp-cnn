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

if __name__ == "__main__":
    datasets = load_tr_te_od_data("../data/stress_vor_w.mat", "../data/stress_vor_o.mat")
    model = torch.load("../models/multi_model_6.pth")
    vals = eval_model_multiple(model, datasets)
    print_all_median_r2s(vals, vals, vals, "../figures/r2_table.txt")
