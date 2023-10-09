import numpy as np
import random
import time
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch import nn, optim


def train_model(model, dataset, valset = None, epochs = 50, lr = 0.001, print_progress = True):
    loss_hist = []
    val_hist = []
    opt = optim.Adam(params = model.parameters(), lr=lr)

    indices = range(len(dataset))
    if valset is not None:
        val_indices = range(len(valset))

    start_time = time.time()

    for epoch in range(epochs):
        indices = random.sample(indices, len(indices))
        if valset is not None:
            val_indices = random.sample(val_indices, len(val_indices))
            loss_val = []
        this_loss = []
        
        for i, k in enumerate(indices):
            data = dataset[k]

            model.train()
            out = model(data)
            loss = F.mse_loss(out, data.y)
            this_loss.append(loss.item())

            opt.zero_grad()
            loss.backward()
            opt.step()

            if valset is not None:
                model.eval()
                idx = val_indices[i % len(val_indices)]
                loss_val.append(F.mse_loss(model(valset[idx]), valset[idx].y).item())

            if print_progress:
                print("\r[%-25s]       \r" %("========================="[24-int(25*i/800):]),end="",flush=True)

        loss_hist.append(np.mean(np.array(this_loss)))

        if valset is not None:
            val_hist.append(np.mean(np.array(loss_val)))
            string = f"Epoch {epoch} of {epochs}... Train loss: {loss_hist[-1]}      Test loss: {val_hist[-1]}"
        else:
            string = f"Epoch {epoch} of {epochs}... Train loss: {loss_hist[-1]}"

        if print_progress:
            print(string)


    model.eval()
    end_time = time.time()
    total_time = end_time - start_time
    if print_progress:
        print(total_time/60, "minutes")
    
    return dict(model=model, loss_hist=loss_hist, val_hist=val_hist, time=total_time/60)


def plot_loss(results):
    train = results["loss_hist"]
    val = results["val_hist"]

    plt.figure(dpi=120)
    plt.plot(train, label="Training")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    if val:
        plt.plot(val, label="Validation")
        plt.legend()

    plt.show()
