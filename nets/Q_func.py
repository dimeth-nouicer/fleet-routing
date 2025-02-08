import numpy as np
import torch

class QFunction():
    def __init__(self, model):
        self.model = model  # The actual QNet
    
    def predict(self, state_tsr, W):
        # batch of 1 - only called at inference time
        with torch.no_grad():
            estimated_rewards = self.model(state_tsr.unsqueeze(0), W.unsqueeze(0))
        return estimated_rewards[0]
                
    def get_best_action(self, state_tsr, state, NR_CS, min_demand):
        """ Computes the best (greedy) action to take from a given state
            Returns a tuple containing the ID of the next node and the corresponding estimated reward
        """
        coords, W, load, demands, solution = state.coords, state.W, state.load, state.demands, state.partial_solution
        soc, time_window, system_time = state.soc, state.tw, state.st
        depot=solution[0]
        nr_nodes = coords.shape[0]  
        n_cs= int(nr_nodes-NR_CS)
        last_visited_node = state.partial_solution[-1]  

        estimated_rewards = self.predict(state_tsr, W)  # size (nr_nodes,)
        sorted_reward_idx = estimated_rewards.argsort(descending=True) 
        
        # When the load is empty return to depot
        if load[0]< min_demand: 
            return depot, estimated_rewards[depot].item()

        # Mask when no time window is available and return to depot
        x = np.array(demands)
        d_candidates = np.asarray(x).nonzero() 
        tw_candidates =[] 
        for i in range(time_window.shape[0]):
            if system_time<time_window[i,1] and i in d_candidates[0]:
                tw_candidates +=[i]
        if len(tw_candidates)==0:
            return depot, estimated_rewards[depot].item()
        
        # Otherwise choose the best candidate
        for idx in sorted_reward_idx.tolist():
            if (((W[last_visited_node, idx]+W[idx, depot]).item()) > soc[0].item()) :
                if idx in range(n_cs, nr_nodes):
                    return idx, estimated_rewards[idx].item()
            if (len(solution) == 0 or W[solution[-1], idx] > 0) and idx in tw_candidates and demands[idx]>0:
                return idx, estimated_rewards[idx].item()
            