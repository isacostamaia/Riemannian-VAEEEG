#%%
import numpy as np
import torch.nn as nn
import torch

from models.modules.spdbn import SPDBatchNorm

class VAE(nn.Module):

    def __init__(self, xVar, zVar, domain_keys, num_epochs, pz="EucStdGaussian", hidden_dim=16, device="cpu"):        
        super(VAE, self).__init__()

        #these objects contain infos about observation and latent variables
        self.xVar = xVar
        self.zVar = zVar

        self.pz = pz #Latent prior #TODO change prior to maybe an attribute choice of distribution class
        self.qz_x = self.zVar.distribution #Latent approximate posterior
        self.px_z = self.xVar.distribution #Observation likelihood

        self.device = device

        self.latent_dim = zVar.vec_shape

        self.num_epochs = num_epochs

        if self.xVar.batchNorm:
            self.spd_bn_layers = nn.ModuleDict()

            # xVar.batchNorm(shape=self.xVar.shape, num_epochs=self.num_epochs, device=self.device)

        #takes and outputs vectorized values
        self.vec_encoder = nn.Sequential(
            nn.Linear(self.xVar.vec_shape, hidden_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, self.latent_dim)
        
        #takes and outputs vectorized values
        self.vec_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, self.xVar.vec_shape),
            nn.LeakyReLU(0.2)
            )
        
        self.init_spd_bn_domains(domain_keys=domain_keys)

    def encode(self, x, domain_id, epoch=None):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input x has NaN or Inf!")

        #BatchNorm
        if self.xVar.batchNorm:
            spd_bn = self.get_spd_bn(domain_id) #get the layer specific to the domain
            x = spd_bn(x, epoch) #epoch arg only needed if in training mode

        #Project into Manifold's Tangent Space if necesary
        if self.xVar.manifold != "Euclidean": #Log Map
            man_basepoint = self.xVar.center_at.clone().repeat(x.shape[0],1,1).to(self.device) #add batch dimension
            x = self.xVar.manifold.log_map(man_basepoint, x) #logarithm map to tangent space
        
        #vectorize input
        x = self.xVar.vectorize(x) 

        #forward to vector encoder
        x = self.vec_encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        logvar = torch.clamp(logvar, min=-5.0, max=2.0)

        return mean, logvar

    def reparametrize(self, mean, logvar, K=1): 
        if self.qz_x != "EucGaussian":
            raise NotImplementedError("Only EucGaussian q(z|x) is supported.")
        
        std = torch.exp(0.5 * logvar)  #shape: batch_size, latent_dim

        if K == 1: #train mode
            eps = torch.randn_like(std)
            z = mean + std * eps  #shape: batch_size, latent_dim

        else: #evaluation mode (TODO: if used in train, add z.view(K * batch_size, latent_dim) before decoder and z.view(K ,batch_size, latent_dim) after)
            eps = torch.randn((K, *std.shape), device=std.device)  #(K, batch_size, latent_dim)
            z = mean.unsqueeze(0) + eps * std.unsqueeze(0)         #(K, batch_size, latent_dim)

        return z

    def decode(self, x): 
        """ Decoder output will be batch of xVar
          i.e. for xVar==SPDVar, mu_theta_zi.shape==(n_batches, n, n) """

        #forward to vector decoder
        f_theta_zi = self.vec_decoder(x)

        if torch.isnan(f_theta_zi).any() or torch.isinf(f_theta_zi).any():
            print("f_theta_zi has NaN or Inf!")

        
        #unvectorize
        f_theta_zi = self.xVar.unvectorize(f_theta_zi)

        #Project back to manifold if necessary
        if self.xVar.manifold != "Euclidean": #Exponential Map
            man_basepoint = self.xVar.center_at.clone().repeat(x.shape[0],1,1).to(self.device) #add batch dimension
            mu_theta_zi = self.xVar.manifold.expmap(man_basepoint, f_theta_zi)
        else:
            mu_theta_zi = f_theta_zi

        return mu_theta_zi

    def forward(self, x, domain_id, epoch=None, K=1):

        mean, logvar = self.encode(x, domain_id, epoch)

        z = self.reparametrize(mean, logvar, K)

        mu_theta_zi = self.decode(z)

        return{
            "mu_theta_zi": mu_theta_zi,
            "z1_mean_m1": mean, 
            "z1_logvar_m1": logvar, 
            "z1": z,} 
       
    def get_spd_bn(self, domain_id):
        key = np.unique(domain_id)
        if len(key)>1:
            raise ValueError("batch data contains more than a domain id.")
        key = str(key[0])
        if key not in self.spd_bn_layers:
            self.spd_bn_layers[key] = SPDBatchNorm(
                shape=self.xVar.shape,
                num_epochs=self.num_epochs,
                device=self.device
            )
        return self.spd_bn_layers[key]
    
    def init_spd_bn_domains(self, domain_keys):
        """
        domain_keys is a list with all the domain_ids that will be seen during training.
        """
        for key in domain_keys:
            self.spd_bn_layers[key] = SPDBatchNorm(shape=self.xVar.shape,
                                                num_epochs=self.num_epochs,
                                                learn_std=True,
                                                device=self.device)


    def unsupervised_loss(self, m1_varparams, x, dom_id, beta):
        #M1 reconstruction loss
        if self.training:
            # sigma_theta_zi = sigma_theta_zi = self.get_spd_bn(dom_id).running_var.clone().detach().sqrt().expand(m1_varparams["mu_theta_zi"].shape[0])  
            sigma_theta_zi = sigma_theta_zi = self.get_spd_bn(dom_id).running_var.clone().detach().expand(m1_varparams["mu_theta_zi"].shape[0])  
        else:
            sigma_theta_zi = self.get_spd_bn(dom_id).running_var_test.clone().detach().expand(m1_varparams["mu_theta_zi"].shape[0])
        
        # print("Decoder sigma_theta_zi: ", sigma_theta_zi.mean().item())
        
        px_z = self.px_z(mu=m1_varparams["mu_theta_zi"], 
                        sigma= sigma_theta_zi, 
                        manifold=self.xVar.manifold)
        lpx_z = px_z.log_prob(x) / np.array(x.shape[1:]).prod() #should have shape=batch_size

        #M1 regulariz loss -- KL term q(z1|x) vs p(z1)
        kl =  -0.5 * torch.sum(1+ m1_varparams["z1_logvar_m1"] - m1_varparams["z1_mean_m1"].pow(2) - m1_varparams["z1_logvar_m1"].exp(), dim = 1) / np.array(m1_varparams["z1"].shape[1:]).prod() #should have shape=batch_size
        kl = beta*kl #beta weight on KL term

        loss = -lpx_z + kl #batch_size
        diagnostics = { 
            "loss": loss,
            "recon": -lpx_z,
            "beta*kl": kl,
        }
        return loss, diagnostics


#### Version that learns logvar in decoder (stochastic decoder)

class VAE_sd(nn.Module):

    def __init__(self, xVar, zVar, domain_keys, num_epochs, pz="EucStdGaussian", hidden_dim=16, device="cpu"):        
        super(VAE, self).__init__()

        #these objects contain infos about observation and latent variables
        self.xVar = xVar
        self.zVar = zVar

        self.pz = pz #Latent prior #TODO change prior to maybe an attribute choice of distribution class
        self.qz_x = self.zVar.distribution #Latent approximate posterior
        self.px_z = self.xVar.distribution #Observation likelihood

        self.device = device

        self.latent_dim = zVar.vec_shape

        self.num_epochs = num_epochs

        if self.xVar.batchNorm:
            self.spd_bn_layers = nn.ModuleDict()

        #takes and outputs vectorized values
        self.vec_encoder = nn.Sequential(
            nn.Linear(self.xVar.vec_shape, hidden_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(hidden_dim, self.latent_dim)
        self.logvar_layer = nn.Linear(hidden_dim, self.latent_dim)
        
        #takes and outputs vectorized values
        self.vec_decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.LeakyReLU(0.2)
            )
        
        #decoder's tg space mean and logvar
        self.f_theta_zi_layer = nn.Linear(hidden_dim, self.xVar.vec_shape)
        self.logvar_theta_zi_layer = nn.Linear(hidden_dim, 1)

        #initialize SPDBN layers by domain
        self.init_spd_bn_domains(domain_keys=domain_keys)

    def encode(self, x, domain_id, epoch=None):
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("Input x has NaN or Inf!")

        #BatchNorm
        if self.xVar.batchNorm:
            spd_bn = self.get_spd_bn(domain_id) #get the layer specific to the domain
            x = spd_bn(x, epoch) #epoch arg only needed if in training mode

        #Project into Manifold's Tangent Space if necesary
        if self.xVar.manifold != "Euclidean": #Log Map
            man_basepoint = self.xVar.center_at.clone().repeat(x.shape[0],1,1).to(self.device) #add batch dimension
            x = self.xVar.manifold.log_map(man_basepoint, x) #logarithm map to tangent space
        
        #vectorize input
        x = self.xVar.vectorize(x) 

        #forward to vector encoder
        x = self.vec_encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        logvar = torch.clamp(logvar, min=-5.0, max=2.0)

        return mean, logvar

    def reparametrize(self, mean, logvar, K=1): 
        if self.qz_x != "EucGaussian":
            raise NotImplementedError("Only EucGaussian q(z|x) is supported.")
        
        std = torch.exp(0.5 * logvar)  #shape: batch_size, latent_dim

        if K == 1: #train mode
            eps = torch.randn_like(std)
            z = mean + std * eps  #shape: batch_size, latent_dim

        else: #evaluation mode (TODO: if used in train, add z.view(K * batch_size, latent_dim) before decoder and z.view(K ,batch_size, latent_dim) after)
            eps = torch.randn((K, *std.shape), device=std.device)  #(K, batch_size, latent_dim)
            z = mean.unsqueeze(0) + eps * std.unsqueeze(0)         #(K, batch_size, latent_dim)

        return z

    def decode(self, x): 
        """ Decoder output will be batch of xVar
          i.e. for xVar==SPDVar, mu_theta_zi.shape==(n_batches, n, n) """

        #forward to vector decoder
        x = self.vec_decoder(x)

        f_theta_zi = self.f_theta_zi_layer(x)
        logvar_theta_zi = self.logvar_theta_zi_layer(x)
        logvar_theta_zi = torch.clamp(logvar_theta_zi, min=-5.0, max=2.0)

        if torch.isnan(f_theta_zi).any() or torch.isinf(f_theta_zi).any():
            print("f_theta_zi has NaN or Inf!")

        #unvectorize
        f_theta_zi = self.xVar.unvectorize(f_theta_zi)

        #Project back to manifold if necessary
        if self.xVar.manifold != "Euclidean": #Exponential Map
            man_basepoint = self.xVar.center_at.clone().repeat(x.shape[0],1,1).to(self.device) #add batch dimension
            mu_theta_zi = self.xVar.manifold.expmap(man_basepoint, f_theta_zi)
        else:
            mu_theta_zi = f_theta_zi

        return mu_theta_zi, logvar_theta_zi
    
    def forward(self, x, domain_id, epoch=None, K=1):

        mean, logvar = self.encode(x, domain_id, epoch)

        z = self.reparametrize(mean, logvar, K)

        mu_theta_zi, logvar_theta_zi = self.decode(z)

        return{
            "mu_theta_zi": mu_theta_zi,
            "logvar_theta_zi": logvar_theta_zi,
            "z1_mean_m1": mean, 
            "z1_logvar_m1": logvar, 
            "z1": z,} 
           
    def get_spd_bn(self, domain_id):
        key = np.unique(domain_id)
        if len(key)>1:
            raise ValueError("batch data contains more than a domain id.")
        key = str(key[0])
        if key not in self.spd_bn_layers:
            self.spd_bn_layers[key] = SPDBatchNorm(
                shape=self.xVar.shape,
                num_epochs=self.num_epochs,
                device=self.device
            )
        return self.spd_bn_layers[key]
    
    def init_spd_bn_domains(self, domain_keys):
        """
        domain_keys is a list with all the domain_ids that will be seen during training.
        """
        for key in domain_keys:
            self.spd_bn_layers[key] = SPDBatchNorm(shape=self.xVar.shape,
                                                num_epochs=self.num_epochs,
                                                learn_std=True,
                                                device=self.device)


    def unsupervised_loss(self, m1_varparams, x, dom_id, beta):
        # #M1 reconstruction loss
        # if self.training:
        #     sigma_theta_zi = sigma_theta_zi = self.get_spd_bn(dom_id).running_var.clone().detach().sqrt().expand(m1_varparams["mu_theta_zi"].shape[0])  
        # else:
        #     sigma_theta_zi = self.get_spd_bn(dom_id).running_var_test.clone().detach().expand(m1_varparams["mu_theta_zi"].shape[0])
        
        sigma_theta_zi = torch.exp(m1_varparams["logvar_theta_zi"]*0.5).squeeze()

        # #compute a single (average) sigma in thebatch (since we only see a domain at a time; maybe this will be less noisy - apparently no)
        # sigma_theta_zi = sigma_theta_zi.mean(0).expand(m1_varparams["mu_theta_zi"].shape[0])        

        px_z = self.px_z(mu=m1_varparams["mu_theta_zi"], 
                        sigma= sigma_theta_zi, 
                        manifold=self.xVar.manifold,
                        learn_logvar=True)
        lpx_z = px_z.log_prob(x) / np.array(x.shape[1:]).prod() #should have shape=batch_size

        #M1 regulariz loss -- KL term q(z1|x) vs p(z1)
        kl =  -0.5 * torch.sum(1+ m1_varparams["z1_logvar_m1"] - m1_varparams["z1_mean_m1"].pow(2) - m1_varparams["z1_logvar_m1"].exp(), dim = 1) / np.array(m1_varparams["z1"].shape[1:]).prod() #should have shape=batch_size
        kl = beta*kl #beta weight on KL term

        loss = -lpx_z + kl #batch_size
        diagnostics = { #total loss is mean over the batch, but each loss is returned as the sum for plot purposes 
            "loss": loss,
            "recon": -lpx_z,
            "beta*kl": kl,
        }
        return loss, diagnostics