import numpy as np
import scipy
from scipy import io
import random
import torch


class Data:
    '''
    This class holds data for a single geometry
    x - The x and y coordinates at each node
    y - The scalar field values at each node
    sdf - An NxN array of SDF values sampled across the geometry
    '''
    def __init__(self, x = None, y = None, sdf = None):
        self.x = x
        self.y = y
        self.sdf = sdf
        
def get_graph(mat,index):
    '''
    get_graph: Reads a single data point from already-loaded matlab data
    
    mat - The dictionary of values read from a .mat file
    index - The index of the data point
    
    Returns - The Data() representation of 'mat'
    '''
    
    nodes = mat['nodes'][index,0].T
    stress = mat['stress'][index,0]
    dt = mat['dt'][index,0]
    sdf = torch.tensor(mat['sdf'][index][0].T, dtype=torch.float)

    # elems = mat['elem'][index,0].T-1
    # f2e = FaceToEdge(remove_faces=True)
    # data = f2e(data)

    x = torch.tensor(np.concatenate((nodes,dt), axis=1), dtype=torch.float)
    y = torch.tensor(stress, dtype=torch.float)

    data = Data(x=x, y=y, sdf=sdf)
    return data


def load_matlab_dataset(filename, scale = 10000):
    '''
    load_matlab_dataset: Loads a scalar field dataset from a .mat file  
    
    Inputs:
    - filename - The .mat dataset consisting of meshes, the scalar field and SDF at each node, and an SDF array
    - scale - The number to divide each scalar field value by, defaults to 10000   
    
    Returns:
    - The dataset as a list of Data() objects
    
    '''
    mat = io.loadmat(filename)
    dataset = []
    for i in range(len(mat['nodes'])):
        data = get_graph(mat, i)
        data.y /= scale
        data.sdf = data.sdf[None, None, :, :] * 10
        dataset.append(data)
        
    return dataset


def get_split_indices(dataset, train_fraction = 0.8, seed = 0):
    '''
    get_split_indices: Given a dataset, randomly generates indices for testing and training  
    
    Inputs: 
    dataset - The list of data points, or the number of data points
    train_fraction - The fraction of points to use for training, defaults to 0.8
    seed - The seed for random number generation, defaults to 0  
    
    Returns:
    - Indices of data points to use for training
    - Indices of data points to use for testing
    '''
    random.seed(seed)
    if type(dataset) == int:
        N = dataset
    else:
        N = len(dataset)
    idxs = random.sample(range(N),N)
    idxs_tr = idxs[:int(train_fraction*N)]
    idxs_te = idxs[int(train_fraction*N):]
    return idxs_tr, idxs_te

def split_data(dataset, idxs_tr, idxs_te):
    data_tr = [dataset[i] for i in idxs_tr]
    data_te = [dataset[i] for i in idxs_te]
    return data_tr, data_te

def load_tr_te_od_data(wss_file, oss_file, scale=10000, frac=0.8, seed=0):
    wss = load_matlab_dataset(wss_file, scale)
    oss = load_matlab_dataset(oss_file, scale)
    idxs_tr, idxs_val = get_split_indices(wss, frac, seed)
    data_tr, data_val = split_data(wss, idxs_tr, idxs_val)
    return dict(tr=data_tr, te=data_val, od=oss)
