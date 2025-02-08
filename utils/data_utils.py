import numpy as np
import random
import torch

from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


def get_graph_mat(n=10, size=1, i=0):
    """ Throws n nodes uniformly at random on a square, and build a (fully connected) graph.
        Returns the (N, 2) coordinates matrix, and the (N, N) matrix containing pairwise euclidean distances.
    """
    #filename = r"C:\Users\MSI\Documents\work\phd\reinforcement learning\vrp dqn gnn\vrptwValidate100_14Nodes\coords_"  + str(i) + ".txt"
    filename = r".\dataset\coords_10_real copy.txt"

    coords = np.loadtxt(filename)
    coords = coords[:n]
    # define min max scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # transform data
    scaled_coords = scaler.fit_transform(coords) 
    #dist_mat = distance_matrix(coords, coords)
    #df = pd.read_excel(r'C:\Users\MSI\Documents\work\phd\reinforcement learning\vrp dqn gnn\test steg\distance30N.xlsx')   #distance30N
    #dist_mat = df.values
    dist_mat =np.loadtxt(r".\dataset\dist_10.txt") 
    #print("dist_mat", dist_mat)
    # define min max scaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    # transform data
    scaled_dist = scaler.fit_transform(dist_mat) 
    #print("scaled_dist", scaled_dist)
    return scaled_coords, scaled_dist

def get_dynamics(n =10, ncs=3):
    """ Returns dynamic variables: 
        * demands of customers, it's a zero for depot and cs
        * load of ev
        * soc of ev
        * system time based on the traveling time of ev from vertex to another + time required to charge at cs
    """      

    n_cs = n-ncs
    #list_values = [0.1,0.05,0.2] 
    #demands_list = [0.0000, 0.2000, 0.2000, 0.2000, 0.1000, 0.0500, 0.0500, 0.0500, 0.2000,
        #0.1000, 0.1000, 0.0000, 0.0000, 0.0000]
    demands_list =    [0.0000, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500, 0.0500,
        0.0500, 0.0500, 0.0000, 0.0000, 0.0000]
    #demands_list = random.choices(list_values, k=n)
    demands = np.array(demands_list)
    demands[0] =0
    demands[n_cs:] =0 
    tensor_demands = torch.from_numpy(demands)
    # load initially at 1
    load = torch.full((n,), 1.) #1.    
    # soc initialized at 100%
    soc = torch.full((n,), 7)
    # system time initialized at 0
    system_time = 0
    # set time window for each node
    time_window = [[0,1]]
    for i in range(n-1): 
        center =np.random.uniform(0,1)
        length = np.random.normal(loc=0.2, scale=0.05)
        due = round(center+length,2)
        start = round(center-length,2)
        if start <0:
            start =0
        if due>1:
            due = 1
        time_window = np.append(time_window,[[start, due]], axis=0)
        
    return tensor_demands, load, soc, system_time, time_window

def plot_graph(coords, mat):
    """ Utility function to plot the fully connected graph
    """
    n = len(coords)
    
    plt.scatter(coords[:,0], coords[:,1], s=[50 for _ in range(n)])
    for i in range(n):
        for j in range(n):
            if j < i:
                plt.plot([coords[i,0], coords[j,0]], [coords[i,1], coords[j,1]], 'b', alpha=0.7)