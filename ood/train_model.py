import os
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)
import sys
sys.path.append('../utils')

from cnn_model import *
from evaluate import *
from load_data import *
from training import *
from visualize import *


if __name__ == "__main__":



    ####################
    # Load datasets

    train_data = load_matlab_dataset("ood_data_train.mat")
    test_data = load_matlab_dataset("ood_data_test.mat")

    #
    ####################


    ####################
    # Train + save model

    model = MultiNet()
    results = train_model(model, train_data, test_data)
    torch.save(model, "ood_model.pth")

    #
    ####################


    ####################
    # Evaluate model, save results

    vals_tr = eval_model_all(model, train_data)
    vals_te = eval_model_all(model, test_data)
    r2s = np.concatenate([vals_tr, vals_te])

    sdfs = []
    for data in train_data + test_data:
        sdfs.append(torch.squeeze(torch.squeeze(data.sdf,0),0).detach().numpy())
    sdfs = np.array(sdfs)
    
    tr_te = np.array([[0,]*len(vals_tr) + [1,]*len(vals_te)]).flatten()

    np.save("sdfs.npy",sdfs)
    np.save("r2s.npy", r2s)
    np.save("tr_te.npy", tr_te)

    #
    ####################