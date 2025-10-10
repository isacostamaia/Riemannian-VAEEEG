import torch
import torch.nn as nn
import numpy as np
from geoopt.tensor import ManifoldParameter, ManifoldTensor
from geoopt.manifolds import Stiefel

from utils.manifold import SPDManifold

class SPDBatchNorm(nn.Module):
    def __init__(self, shape, num_epochs, karcher_steps=1, gamma_test=0.1,
                 mean = None, std = None, learn_mean = False, learn_std = True, 
                 learn_rot = True,
                 eps=1e-5, device = "cpu"):
        super().__init__()
        self.shape = shape
        self.karcher_steps = karcher_steps
        self.learn_mean = learn_mean
        self.learn_std = learn_std
        self.learn_rot = learn_rot
        self.eps = eps
        self.manifold = SPDManifold()
        self.num_epochs = num_epochs
        # self.gamma = gamma
        self.gamma_min = 0.2
        self.gamma_max = 1
        self.K = int(self.num_epochs*.8) #training step at which gamma reaches it maximum value 
        self.gamma = np.clip([1 - self.gamma_min**(np.max([self.K-k, 0])/(self.K-1)) + self.gamma_min  \
                                                                    for k in range(self.num_epochs)], \
                                                                    a_min=self.gamma_min, a_max=self.gamma_max) #decreasing schedule for gamma in [0,1]
        self.gamma_test = gamma_test

        init_mean = torch.eye(shape[-1], device=device) #init of running mean G_0
        init_var = torch.ones((1), device=device) #init of running var nu²_0
        
        self.min_std = 1e-3  # floor to avoid instability


        #register so they cound appear in model state dict (they don't require grad)
        self.register_buffer('running_mean', ManifoldTensor(init_mean, #running mean G_k
                                           manifold=self.manifold))
        self.register_buffer('running_var', init_var) #running var nu²_k
        self.register_buffer('running_mean_test', ManifoldTensor(init_mean, 
                                           manifold=self.manifold))
        self.register_buffer('running_var_test', init_var)


        if mean is not None: #G_\phi (for now not used)
            self.mean = mean
        else:
            if self.learn_mean:
                self.mean = ManifoldParameter(init_mean.clone(), manifold=self.manifold)
            else:
                self.mean = ManifoldTensor(init_mean.clone(), manifold=self.manifold)

        

        if std is not None: #\nu_\phi
            self.std = std
        else:
            if self.learn_std:
                # self.std = nn.parameter.Parameter(init_var.clone())

                init_std = 1.0 - self.min_std
                raw_std_init = torch.log(torch.exp(torch.tensor(init_std, device=device)) - 1.0)
                # raw_std_init = torch.randn(1)*0.1 #added 2
                noise = 1e-3 * torch.randn_like(raw_std_init) #added
                # raw_std = nn.Parameter(raw_std_init.expand_as(init_var))         
                raw_std = nn.Parameter(raw_std_init + noise) #added
                self.register_parameter('raw_std', raw_std)


            else:
                self.std = init_var.clone()

        self.rot_mat_manifold = Stiefel()
        if self.learn_rot:
            self.rot_mat = ManifoldParameter(self.rot_mat_manifold.origin(shape[-1], shape[-1], device=device), 
                                             manifold=self.rot_mat_manifold)
            self.register_parameter('rot_mat', self.rot_mat)

        
        else:
            self.rot_mat = self.rot_mat_manifold.origin(shape,device=device)

    def forward(self, X, epoch=None):
        #epoch only needed if in training mode
        if self.training:

            #estimate and update train running mean G_k
            Bk = self.manifold.karcher_flow(X)  #Compute batch mean B_k
            gamma_t = torch.tensor(self.gamma[epoch], device=X.device, dtype=X.dtype)
            self.running_mean =  self.manifold.interpolate(A = self.running_mean, B = Bk, gamma=gamma_t)

            #estimate and update train running var nu²_k :  
            var_Bk = (self.manifold.squared_distance(self.running_mean.expand_as(X), X)).mean() #single scalar
            self.running_var = (1-gamma_t)*self.running_var + gamma_t* var_Bk

            #estimate and update test running mean
            gamma_t_test = torch.tensor(self.gamma_test, device=X.device, dtype=X.dtype)
            self.running_mean_test = self.manifold.interpolate(A=self.running_mean_test, B=Bk, gamma=gamma_t_test)

            #estimate and update test running var
            var_Bk_test = (self.manifold.squared_distance(self.running_mean_test.expand_as(X), X)).mean() #single scalar
            self.running_var_test = (1-gamma_t_test)*self.running_var_test + gamma_t_test*var_Bk_test

            #use train running mean and train running var
            rm = self.running_mean.clone()
            rv = self.running_var.clone()

        else:
            #use test running mean G_k
            rm = self.running_mean_test
            #use test running var nu²_k :  
            rv = self.running_var_test

        #Normalize data: transport to Id and rescale the dispersion
        #handling std so it doesnt get negative
        self.std = torch.nn.functional.softplus(self.raw_std) + self.min_std

        if torch.isnan(self.std) or self.std < self.min_std:
            print("std problematic:", self.std)

        alpha = (self.std.to(X.device) / (rv.to(X.device) + self.eps)).sqrt()
        fromA_batch = rm.unsqueeze(0).expand(X.shape[0], -1, -1).clone()

        Xn = self.manifold.parallel_transp_to_id(fromA=fromA_batch, transportX=X) #transport to Id
        Xn = self.manifold._matrix_power(Xn, alpha) #rescale by std dev rate
        print("Xn requires_grad before rot_mat?", Xn.requires_grad)
        if self.learn_rot:
            Xn = torch.transpose(self.rot_mat, dim0=-2, dim1=-1) @ Xn @ self.rot_mat

        return Xn