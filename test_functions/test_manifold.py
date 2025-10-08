#%%
%load_ext autoreload 
%autoreload 2

import sys
sys.path.append("..") 

import numpy as np
import torch
from pyriemann.datasets import generate_random_spd_matrix
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.tangentspace import exp_map_riemann, log_map_riemann
from pyriemann.utils.mean import mean_logeuclid, mean_riemann
from pyriemann.utils.geodesic import geodesic_riemann

from manifold import SPDManifold

np.random.seed(3)
torch.manual_seed(0)

C_list = [generate_random_spd_matrix(n_dim=3) for _ in range(10)]
C_pairs_arrays = list(zip(C_list[::2], C_list[1::2]))
C_pairs_tensors = list(zip(torch.tensor(C_list[::2]), torch.tensor(C_list[1::2])))

spd_man = SPDManifold()

#test affine_inv distance
#Pyriemann
dist_pyr = [distance_riemann(*cp, squared = True) for cp in C_pairs_arrays]
#SPDManifold opt (GOOD)
dist_my_op = [spd_man.squared_distance_opt(*cp) for cp in C_pairs_tensors]
#SPDManifold regular (GOOD)
dist_my = [spd_man.squared_distance(*cp) for cp in C_pairs_tensors]


#test exp map
#Pyriemann
exp_pyr = [exp_map_riemann(*cp[::-1], Cm12=True) for cp in C_pairs_arrays]

#SPDManifold  (GOOD)
exp_my = [spd_man.expmap(*cp) for cp in C_pairs_tensors]


#test log map
#Pyriemann
log_pyr = [log_map_riemann(*cp[::-1], C12=True) for cp in C_pairs_arrays]

#SPDManifold (GOOD)
log_my = [spd_man.log_map(*cp) for cp in C_pairs_tensors]



#test log-euclidean mean
#Pyriemann
log_euc_mean_pyr = mean_logeuclid(np.array(C_list)) 

#SPDManifold (GOOD)
log_euc_mean_my = spd_man.log_euclidean_mean(torch.Tensor(C_list))


#test affine-invariant mean with karcher flow 
#Pyriemann
mean_pyr = mean_riemann(np.array(C_list))

#SPDManifold (GOOD)
mean_my_looks_like_log_euc_pyr = spd_man.karcher_flow(torch.Tensor(C_list), steps=1, initialize="Id", eta=1)

#SPDManifold with default options (GOOD)
mean_my = spd_man.karcher_flow(torch.Tensor(C_list), steps=100, initialize="EucMean", eta=1)

mean_my_imp = spd_man.karcher_flow_improv(torch.Tensor(C_list), steps=100)

#test interpolation
#Pyriemann
point_pyr = geodesic_riemann(*C_pairs_arrays[0], alpha=0.2)

#SPDManifold (GOOD)
point_my = spd_man.interpolate(*C_pairs_tensors[0], gamma=0.2)

# Test 
#spd_man.parallel_transp_to_id
#spd_man.norm

# # %%

# %%
