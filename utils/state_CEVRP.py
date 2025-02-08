
import numpy as np
import random
import torch

from collections import namedtuple

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') # Note: the code is not optimized for GPU



State = namedtuple('State', ('W', 'coords', 'partial_solution','demands', 'load', 'soc', 'tw', 'st'))
  
def state2tens(state):
    """ Creates a Pytorch tensor representing the history of visited nodes, from a (single) state tuple.
        
        Returns a (Nx5) tensor, where for each node we store whether this node is in the sequence,
        whether it is first or last, and its (x,y) coordinates.
    """
    solution = set(state.partial_solution)
    sol_last_node = state.partial_solution[-1] if len(state.partial_solution) > 0 else -1
    sol_first_node = state.partial_solution[0] if len(state.partial_solution) > 0 else -1
    coords, demand  = state.coords, state.demands
    nr_nodes = coords.shape[0]

    xv = [[(1 if i in solution else 0),
           (1 if demand[i] >0 else 0),
           (1 if i == sol_last_node else 0),
           coords[i,0],
           coords[i,1]
          ] for i in range(nr_nodes)]
    
    return torch.tensor(xv, dtype=torch.float32, requires_grad=False, device=device)


def total_distance(solution, W):
    if len(solution) < 2:
        return 0  # there is no travel

    total_dist = 0
    for i in range(len(solution) - 1):
        total_dist += W[solution[i], solution[i+1]].item()
     
    # if this solution is "complete", go back to initial point
    arr_solution = np.array(solution)
    where_zeros = np.where(arr_solution == 0)[0]
    nbr_depots = where_zeros.shape[0]  
    if len(solution) == W.shape[0]+nbr_depots-1: # -1: remove initial depot starting point
        total_dist += W[solution[-1], solution[0]].item()

    return total_dist

def is_state_final(state, NR_EV):

    demands, solution = state.demands, state.partial_solution
    # nbr of depot visits
    arr_solution = np.array(solution)
    where_zeros = np.where(arr_solution == 0)[0]
    nbr_depots = where_zeros.shape[0]  
    
    if nbr_depots>NR_EV:
        return True
    if demands.eq(0).all():
        demands * 0.
        return True
    else: 
        return False

def update_length_state(state, next_node, NR_CS):    
    """ Update the system time, soc, demand / load values"""
    
    # static variables
    W, coords, demands, load, system_time, old_soc = state.W, state.coords, state.demands, state.load, state.st, state.soc
    depot_coords = state.coords[0]
    nr_nodes = coords.shape[0] 
    nr_cs = int(nr_nodes-NR_CS) 
    # dynamic variables
    clone_demands = demands.clone()
    clone_load = load.clone()
    remaining_capacity = old_soc[next_node].item()
    last_visited_node = state.partial_solution[-1]  
    service_time = 0.1
 
    # get distance from last visited node to next node
    distTo_next_node = W[last_visited_node, next_node].item()
    traveling_time= (0.4*distTo_next_node)/6

    # we update the state based on the next node id 
    if next_node == 0: #depot chosen
        # update load
        clone_load = torch.full((nr_nodes,), 1.)
        # update demand
        clone_demands[next_node] = 0
        tensor_soc = torch.full((nr_nodes,), 7)
        system_time =0
        return clone_load, clone_demands, tensor_soc, system_time

    if next_node in range(1, nr_cs): #customer chosen   
        # update load
        new_load = torch.clamp(clone_load - clone_demands[next_node], min=0)
        # update demands
        clone_demands[next_node] = 0  
        # update soc
        #new_soc = remaining_capacity -(distTo_next_node) 
        new_soc=7
        tensor_soc = torch.full((nr_nodes,), new_soc) 
        # update time
        #system_time = system_time+traveling_time+service_time # distTo_next_node /speed *** speed ~ 17-20, service time 0.5
        system_time +=0.01 
        return new_load, clone_demands, tensor_soc, system_time
    
    #station
    tensor_soc = torch.full((nr_nodes,), 7)
    # update time
    system_time = system_time+traveling_time+0.1
    return state.load, demands, tensor_soc, system_time