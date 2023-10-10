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


def train_model_to_file(args, filename):
    print(f"\nTraining model and saving to file: {filename}")
    if os.path.exists(filename):
        print("File already exists. Skipping.")
        return
    result = train_model(**args)
    torch.save(args["model"], filename)
    root, _ = os.path.splitext(filename)
    with open(root+"_log.txt", "w") as file:
        print(f"\n______   Model {root}   _____\n", file=file)
        print(f"Time: {result['time']}\n", file=file)
        print(f"Loss (training): {result['loss_hist']}\n", file=file)
        print(f"Loss (validation): {result['val_hist']}\n", file=file)
    print(f"Done. Results printed to {root}_log.txt")
    

def train_stress_models():
    datasets_vor = load_tr_te_od_data("../data/stress_vor_w.mat", "../data/stress_vor_o.mat")
    datasets_lat = load_tr_te_od_data("../data/stress_lat_w.mat", "../data/stress_lat_o.mat")
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]
    
    vor_model = MultiNet(kernel_size=5, num_layers = 6, num_filters=20)
    args = dict(model=vor_model, dataset=datasets_vor["tr"], valset=datasets_vor["te"])
    train_model_to_file(args, "../models/vor_model.pth")

    lat_model = MultiNet(kernel_size=5, num_layers = 6, num_filters=20)
    args = dict(model=lat_model, dataset=datasets_lat["tr"], valset=datasets_lat["te"])
    train_model_to_file(args, "../models/lat_model.pth")

    model = MultiNet(kernel_size=5, num_layers = 6, num_filters=20)
    args = dict(model=model, dataset=datasets["tr"], valset=datasets["te"])
    train_model_to_file(args, "../models/multi_model_6.pth")

def train_temp_models():
    datasets_vor = load_tr_te_od_data("../data/temp_vor_w.mat", "../data/temp_vor_o.mat", scale=1)
    datasets_lat = load_tr_te_od_data("../data/temp_lat_w.mat", "../data/temp_lat_o.mat", scale=1)
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]
    
    vor_model = MultiNet(kernel_size=5, num_layers = 6, num_filters=20)
    args = dict(model=vor_model, dataset=datasets_vor["tr"], valset=datasets_vor["te"])
    train_model_to_file(args, "../models/temp_vor_model.pth")

    lat_model = MultiNet(kernel_size=5, num_layers = 6, num_filters=20)
    args = dict(model=lat_model, dataset=datasets_lat["tr"], valset=datasets_lat["te"])
    train_model_to_file(args, "../models/temp_lat_model.pth")

    model = MultiNet(kernel_size=5, num_layers = 6, num_filters=20)
    args = dict(model=model, dataset=datasets["tr"], valset=datasets["te"])
    train_model_to_file(args, "../models/temp_multi_model_6.pth")

def train_diff_layer_models():
    datasets_vor = load_tr_te_od_data("../data/stress_vor_w.mat", "../data/stress_vor_o.mat")
    datasets_lat = load_tr_te_od_data("../data/stress_lat_w.mat", "../data/stress_lat_o.mat")
    datasets = dict()
    for key in datasets_vor:
        datasets[key] = datasets_vor[key] + datasets_lat[key]
    
    for i in range(6):
        model = MultiNet(kernel_size=5, num_layers = i+1, num_filters=20)
        args = dict(model=model, dataset=datasets["tr"], valset=datasets["te"])
        train_model_to_file(args, f"../models/multi_model_{i+1}.pth")

if __name__ == "__main__":
    train_stress_models()
    train_temp_models()
    train_diff_layer_models()