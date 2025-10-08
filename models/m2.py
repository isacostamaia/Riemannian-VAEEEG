import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import random, numpy as np, torch

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)  # or  torch.set_deterministic(True)

seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

import torch.nn as nn
import torch.nn.functional as F


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions import Categorical


class VAEM2(nn.Module):
    """
    Supervised VAE model as described in 
    """
    def __init__(self, z1_dim, z2_dim, y_dim, hidden_dim, device="cpu"):
        super().__init__()
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        self.py_logits = nn.Parameter(torch.zeros(self.y_dim), requires_grad=True)

        self.device = device

        self.encoder = nn.Sequential(

                                    )

        self.qy_z1_logits = nn.Linear(self.z1_dim, self.y_dim)
        self.cls_loss_fct = nn.CrossEntropyLoss(weight=torch.Tensor([0.2, 0.8]).to(self.device), reduction='none')
        # self.cls_loss_fct = nn.CrossEntropyLoss(reduction='none')

        #encoder of z2
        self.encoder = nn.Sequential(
                                nn.Linear(self.z1_dim+self.y_dim, self.hidden_dim),
                                nn.LeakyReLU(0.2)
                                )

        #mean and var of q(z2 | z1, y )
        self.qz2_mean_layer = nn.Linear(self.hidden_dim, self.z2_dim)
        self.qz2_logvar_layer = nn.Linear(self.hidden_dim, self.z2_dim)

        self.decoder = nn.Sequential(
                                    nn.Linear(self.y_dim+self.z2_dim, self.hidden_dim),
                                    nn.LeakyReLU(0.2),
        )

        #mean and var of p(z1 | z2, y )
        self.pz1_mean_layer = nn.Linear(self.hidden_dim, self.z1_dim)
        self.pz1_logvar_layer = nn.Linear(self.hidden_dim, self.z1_dim)

    def logp_y(self,y):
        return F.log_softmax(self.py_logits, dim=0)[y]
    
    def encode_y(self, z1):
        logits = self.qy_z1_logits(z1)
        return F.log_softmax(logits, dim=-1), logits

    def encode_z2(self, z1, y_onehot): 
        h = self.encoder(torch.cat([z1,y_onehot], dim=1)) #concatenate
        z2_mean = self.qz2_mean_layer(h)
        z2_logvar = self.qz2_logvar_layer(h)
        return z2_mean, z2_logvar

    def decode_z1(self, z2, y_onehot):
        h = self.decoder(torch.cat([z2,y_onehot], dim=1)) #concatenate
        z1_mean = self.pz1_mean_layer(h)
        z1_logvar = self.pz1_logvar_layer(h)
        return z1_mean, z1_logvar
    
    def reparametrize(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(self.device)      
        z = mean + torch.exp(logvar/2) * epsilon
        return z
    
    def forward(self, z1, y): #for labeled case, we don't obtain y
        y_onehot = F.one_hot(y, num_classes=self.y_dim).float().to(self.device) #true y
        z2_mean, z2_logvar = self.encode_z2(z1, y_onehot)
        z2 = self.reparametrize(z2_mean, z2_logvar)
        z1_mean, z1_logvar = self.decode_z1(z2, y_onehot)
        return {"z1_mean":z1_mean,
                "z1_logvar":z1_logvar,
                "z2_mean":z2_mean,
                "z2_logvar":z2_logvar} 
    
    def infer(self, z1_mean):
        logits = self.qy_z1_logits(z1_mean)
        return logits.argmax(dim=-1)
    
    def supervised_loss(self, z1_m1, z1_mean_m1, y, m2_varparams, alpha):
        """
        Inputs:
        z1: latent code from M latent space. Tensor of shape (batch_size, z1_dim)
        z1_mean: mean of M1 latent code before reparametrization. Tensor of shape (batch_size, z1_dim)
        y: tensor of class indexes with shape (batch_dim,)
        m2_varparams: dictionary with M2 forward outputs
        """

        z1_mean = m2_varparams["z1_mean"]     #obtained at  M2 decoder
        z1_logvar = m2_varparams["z1_logvar"] #obtained at M2 decoder
        z2_mean = m2_varparams["z2_mean"] #obtained from M2 encoder
        z2_logvar = m2_varparams["z2_logvar"] #obtained from M2 encoder

        #reconstruction term -- p_theta(z1 | z2, y) (in the article, p(x | z, y))
        # recon_loss = 0.5 * torch.sum(z1_logvar + ((z1 - z1_mean) ** 2) / torch.exp(z1_logvar)) / np.array(z1.shape[1:]).prod()
        const = z1_m1.size(1) * np.log(2 * np.pi)  # dimensionality times log 2π
        recon_loss = 0.5 * (
            const + torch.sum(z1_logvar + ((z1_m1 - z1_mean) ** 2) / torch.exp(z1_logvar), dim=1)
        ) / np.array(z1_m1.shape[1:]).prod() #should have shape=batch_size

        #regulariz loss -- KL term q(z2 | z1, y) vs p(z2) (in the article, KL term q(z | x, y) vs p(z))
        kl_z2 = -0.5 * torch.sum(1 + z2_logvar - z2_mean.pow(2) - z2_logvar.exp(), dim=1) / np.array(z2_mean.shape[1:]).prod() #should have shape=batch_size

        #(learnable) prior over y -- log p(y)
        # logp_y = F.log_softmax(m2.py_logits, dim=0)[y] #prior log probability of the actual label #should have shape=batch_size
        logp_y = self.logp_y(y)  #should have shape=batch_size #prior log probability of the actual label 

        #M2 classification loss -- q(y | z1) (in the article, q(y | x))
        probs_y, logits = self.encode_y(z1_mean_m1) #y_pred logits, only for training the classifier 
        # cls_loss = F.cross_entropy(logits, y, weight=torch.Tensor([0.2, 0.8]).to(m2.device))
        cls_loss = self.cls_loss_fct(logits, y) #should have shape=batch_size

        loss = recon_loss + kl_z2 -logp_y + alpha * cls_loss  #shape=batch_size

        diag = {"recon_loss":recon_loss,
                "kl_z2_loss": kl_z2,
                "yprior_loss": (-logp_y),
                "yclass_loss": (alpha * cls_loss)
                }
        return loss, diag


class VAEM2_CCVAE(nn.Module):
    """
    M2 is a supervised VAE that takes input z_1 and encodes it into a latent variable z = [z_bc, z_c],
    where z_c can generate an associate label y.    
    """
    def __init__(self, z1_dim, zbc_dim, zc_dim, y_dim, hidden_dim, device="cpu"):
        super().__init__()
        self.z1_dim = z1_dim #hidden var from M1 that is input to M1 encoder
        self.zc_dim = zc_dim #z_c -> latent related to y var
        self.zbc_dim = zbc_dim #z_{\c} -> latent unrelated to y vars
        self.y_dim = y_dim #number of classes
        self.hidden_dim = hidden_dim #hidden dimension in encoder and decoder intermediate layers
        # self.py_logits = nn.Parameter(torch.zeros(self.y_dim), requires_grad=False)
        self.device = device


        #encoder x -(Φ)- > z                with z = [z_{\c}, z_c] on the paper
        self.encoder = nn.Sequential(
                                nn.Linear(self.z1_dim, self.hidden_dim),
                                nn.LeakyReLU(0.2)
                                )
            #mean and var of q_Φ(z | x) on the paper, (q_Φ(z | z_1) for us)
        self.qz_mean_layer = nn.Linear(self.hidden_dim, self.zc_dim + self.zbc_dim)
        self.qz_logvar_layer = nn.Linear(self.hidden_dim, self.zc_dim + self.zbc_dim)

        #decoder z -(θ)- > x                with z = [z_{\c}, z_c] on the paper
        self.decoder = nn.Sequential(
                                    nn.Linear(self.zc_dim + self.zbc_dim, self.hidden_dim),
                                    nn.LeakyReLU(0.2),
        )
            #mean and var of p_θ(z1 | z2, y )
        self.pz1_mean_layer = nn.Linear(self.hidden_dim, self.z1_dim)
        self.pz1_logvar_layer = nn.Linear(self.hidden_dim, self.z1_dim)

        #y classifier q_φ(y|zc) logits
        self.q_y_zc_logits = nn.Linear(self.zc_dim, self.y_dim)

        #conditional prior params p_Ψ(z_c |y)
        self.cond_zc_prior_back = nn.Sequential(
                                    nn.Linear(self.y_dim, self.hidden_dim),
                                    nn.LeakyReLU(0.2),
        )
            #mean and var of p_Ψ(z_c |y)
        self.pzc_y_mean_layer = nn.Linear(self.hidden_dim, self.zc_dim)
        self.pzc_y_logvar_layer = nn.Linear(self.hidden_dim, self.zc_dim)

    def q_z_z1(self, z1):  #Approximate posterior: q_Φ(z | x) on the paper
        h = self.encoder(z1) # (batch_size,hidden_dim)
        z_mean = self.qz_mean_layer(h)
        z_logvar = self.qz_logvar_layer(h)
        return z_mean, z_logvar

    def reparametrize(self, mean, logvar):
        epsilon = torch.randn_like(logvar).to(self.device)      
        z = mean + torch.exp(logvar/2) * epsilon
        return z
    
    def p_z1_z(self, z): #Model likelihood: p_θ(x | z) on the paper, (p_θ(z1 | z) for us)
        h = self.decoder(z) # (batch_size,hidden_dim)
        z1_mean = self.pz1_mean_layer(h)
        z1_logvar = self.pz1_logvar_layer(h)
        return z1_mean, z1_logvar
    
    def log_q_y_zc(self, zc):  #Label predictive distribution: q_φ(y | z_c) on the paper 
        #should be used with unparametrized z_c
        logits = self.q_y_zc_logits(zc)
        return Categorical(logits=logits) 
    
    def p_zc_y(self, y): #Conditional prior: p_Ψ(z_c |y)
        h = self.cond_zc_prior_back(y)
        zc_mean = self.pzc_y_mean_layer(h)
        zc_logvar = self.pzc_y_logvar_layer(h)
        return zc_mean, zc_logvar
    
    def zbc_zc(self, z):
        "returns z = [z_{\c}, z_c]"
        return torch.split(z, [self.zbc_dim, self.zc_dim], dim = -1) #split over the last dimension

    def log_q_y_z1(self, z1, K=100): #q_{φ,Φ}(y |x) on the paper
        """
        Used for classifier loss.
        Monte Carlo estimation of
        q_{φ,Φ}(y |x) = \int q_φ(y | z_c) * q_Φ(z | x) dz on the paper
        
        """
        #encoder: infer z from z1 (infer z from x using q_Φ(z | x) on the paper)
            #dist params
        z_mean, z_logvar = self.q_z_z1(z1)
        if K!=1:
                #reparametrize K particles z_k
            z_mean_, z_logvar_ = z_mean.expand(K, -1, -1), z_logvar.expand(K, -1, -1) #(K, batch_size,zbc_dim+zc_dim) each
            z_ = self.reparametrize(z_mean_.contiguous().view(-1, self.zbc_dim+self.zc_dim), #(K*batch_size,zbc_dim+zc_dim)
                                    z_logvar_.contiguous().view(-1, self.zbc_dim+self.zc_dim)) 
        else: #don't reparametrize
            z_ = z_mean
        #split each z_k in z_ to get z_c. Forward to q_φ(y | z_c)
        _, zc_ = self.zbc_zc(z_) #(K*batch_size,zc_dim) 
        # logits = self.log_q_y_zc(zc_) #logits.shape = (K*batch_size,y_dim)

        return self.log_q_y_zc(zc_) #Categorical dist.
    
    def classifier_loss(self, z1, y, K=100):
        """
        Evaluates log q_{φ,Φ}(y |z1) (q_{φ,Φ}(y |x) on the paper) for a given y.
        Inputs:
        y:  tensor of indexes of shape (batch_size,)
        z1: original encoder's input tensor of shape (batch_size, z1_dim)
        """
        log_q_y_z1 = self.log_q_y_z1(z1, K) #logprob density params for 
        y = y.repeat(K) #tensor of shape (K*batch_size)
        logprob_ys_z = log_q_y_z1.log_prob(y)
        logprob_y_z = (logprob_ys_z.view(K, -1)) #(K, batch_size) 
        logprob_y_x = torch.logsumexp(logprob_y_z, dim = 0) - torch.log(torch.tensor(K, device=logprob_y_z.device)) #(batch_size)
        return -logprob_y_x
        
    def infer(self, z1):
        log_q_y_z1 = self.log_q_y_z1(z1, K=1) #logprob density params 
        pred = log_q_y_z1.logits.argmax(dim=-1)
        return pred


    def forward(self, z1, y, K=100): 
        """
        z1: (batch_size, z1_dim) tensor - hidden var from M1 that is input to M2 encoder
        y: (batch_size, y_dim) tensor - y_dim the number of class labels
        """
        #convert label indexes to one-hot encoding
        y_onehot = F.one_hot(y, num_classes=self.y_dim).float().to(z1.device) #groundtrue y

        #encode x -(Φ)- > z   with z = [z_{\c}, z_c] on the paper, (z1 -(Φ)- > z for us)
            #q_Φ(z | x) params, (q_Φ(z | z1) for us)
        z_mean, z_logvar = self.q_z_z1(z1) #(batch_size,zbc_dim+zc_dim) ok

        #reparametrize K particles z_k (Monte Carlo)
        z_mean_, z_logvar_ = z_mean.expand(K, -1, -1), z_logvar.expand(K, -1, -1) #(K, batch_size,zbc_dim+zc_dim) each
        z_ = self.reparametrize(z_mean_.contiguous().view(-1, self.zbc_dim+self.zc_dim), #(K*batch_size,zbc_dim+zc_dim)
                                z_logvar_.contiguous().view(-1, self.zbc_dim+self.zc_dim)) 

        #decode z -(θ)- > x   with z = [z_{\c}, z_c] on the paper, (z -(θ)- > z1 for us)
            #p_θ(x | z) on the paper (p_θ(z1 | z) for us)
        z1_mean, z1_logvar = self.p_z1_z(z_) #(K*batch_size,z1_dim) ok


        #split z = [z_{\c}, z_c]
        zbc_mean, zc_mean = self.zbc_zc(z_mean) #(batch_size,zbc_dim) and #(K, batch_size,zc_dim)
        zbc_logvar, zc_logvar = self.zbc_zc(z_logvar) #(batch_size,zbc_dim) and #(K, batch_size,zc_dim)

        #obtain z_c from y (latent variable generation) using conditional prior:
            # p_Ψ(z_c |y)
        zc_prior_mean, zc_prior_logvar = self.p_zc_y(y_onehot)  #(batch_size,y_dim) ok

        #obs: q_{φ,Φ}(y |x) and q_φ(y|z_c) will be evaluated directly in the loss function.

        return {
            "z_": z_, #K Monte Carlo particles
            "z_mean": z_mean, "z_logvar": z_logvar, 
            "zc_mean": zc_mean, "zc_logvar": zc_logvar,
            "zbc_mean": zbc_mean, "zbc_logvar": zbc_logvar,
            "z1_mean": z1_mean, "z1_logvar": z1_logvar,
            "zc_prior_mean": zc_prior_mean, "zc_prior_logvar": zc_prior_logvar,
        }

    def supervised_loss(self, z1_m1, z1_mean_m1, y, K, m2_varparams, alpha=1, eps=1e-6):
        """
        
        Inputs:

        z1: input tensor to the encoder (observation) of shape (batch_size, z1_dim)
        y:  label tensor with class indexes of shape (batch_size, )
        K: number of Monte Carlo particles (int)
        alpha: classification weight
        eps: min. value of std clamping
        """

        B = z1_m1.shape[0] #batch_size

        z1_mean_parts = m2_varparams["z1_mean"].view(K, B, -1)
        z1_logvar_parts = m2_varparams["z1_logvar"].view(K, B, -1)
        z1_m1_rep = z1_m1.expand(K, -1, -1) #(K,batch_size,zbc_dim+zc_dim)

        #(a) ok
            #log p_θ(x | z) on the paper (p_θ(z1 | z) for us)
            #takes inferred z and decodes into recosntructed z1
            #z1_mean and z1_logvar have shape (K*batch_size,z1_dim)
            #z1 has shape (batch_size, z1_dim)
            #reshape into Monte Carlo particles

        log_p_z1_z = Normal(z1_mean_parts, 
                            (torch.exp(0.5*z1_logvar_parts)).clamp(min=eps)
                            ).log_prob(z1_m1_rep).sum(dim=-1) #return per-dim log-prob. We should sum over them.
        #(K, batch_size)

        #(b) ok
            #p_Ψ(z |y) = p_Ψ(z_c |y)*p_Ψ(z_bc)
            #zc_prior_mean, zc_prior_logvar have shape (batch_size, zc_dim+zbc_dim)
            #z_ has shape (K*batch_size, zc_dim+zbc_dim)
        z_parts = m2_varparams["z_"].view(K, B, -1) #(K, batch_size, zc_dim+zbc_dim)
        zbc_parts, zc_parts = self.zbc_zc(z_parts) #shapes (K,B,zbc_dim) and (K,B,zc_dim)
        zc_prior_mean_rep = (m2_varparams["zc_prior_mean"]).expand(K, -1, -1) #(K,B,zc_dim)
        zc_prior_logvar_rep = (m2_varparams["zc_prior_logvar"]).expand(K, -1, -1) #(K,B,zc_dim)
                    #p_Ψ(z_c |y)
        log_p_zc_y = Normal(zc_prior_mean_rep, 
                            torch.exp(0.5*zc_prior_logvar_rep).clamp(min=eps)
                            ).log_prob(zc_parts).sum(dim=-1) #return per-dim log-prob. We should sum over them.
        #shape = (K, batch_size)

            #p_Ψ(z_bc) (here a std gaussian)
        log_p_zbc_y = Normal(torch.zeros_like(zbc_parts), 
                             torch.ones_like(zbc_parts)).log_prob(zbc_parts).sum(dim=-1) #return per-dim log-prob. We should sum over them.
        #shape = (K, batch_size)
        #sum of logprobs         
        log_p_z_y = log_p_zc_y + log_p_zbc_y

        #(c)
            #q_φ(y|z_c) 
            #zc has shape (batch_size, zc_dim)

        #     #Here z_c is used without reparametrizing, therefore no MC particles
        log_q_y_zc = self.log_q_y_zc(m2_varparams["zc_mean"]).log_prob(y)
        #expand to be the same shape as the other losses 
        log_q_y_zc = log_q_y_zc.expand(K, -1) #(K, batch_size)

        # zc_for_class = zc_parts.detach().contiguous().view(-1, self.zc_dim)  # (K*batch_size, zc_dim)
        # y_rep = y.expand(K, -1).contiguous().view(-1)  # (K*batch_size,)
        # log_q_y_zc = self.log_q_y_zc(zc_for_class).log_prob(y_rep).view(K,B)      # (K, batch_size)


        #(d) ok
            #q_Φ(z | x)
            #z_mean, z_logvar have shape (batch_size, zc_dim+zbc_dim)
            #z_ has shape (K*batch_size, zc_dim+zbc_dim)
        z_mean_rep = (m2_varparams["z_mean"]).expand(K, -1, -1) #(K, batch_size, zc_dim+zbc_dim)
        z_logvar_rep = (m2_varparams["z_logvar"]).expand(K, -1, -1) #(K, batch_size, zc_dim+zbc_dim)

        log_q_z_z1 = Normal(z_mean_rep, 
                            torch.exp(0.5*z_logvar_rep).clamp(min=eps)
                            ).log_prob(z_parts).sum(dim=-1) #return per-dim log-prob. We should sum over them.
        #(K, batch_size)

        #(e)
            #log q_{φ,Φ}(y |x), (log q_{φ,Φ}(y |z1) for us)
        minus_log_q_y_z1 = self.classifier_loss(z1_mean_m1, y, K=K) #always do Monte Carlo here (batch_size,)

        log_w = log_q_y_zc + minus_log_q_y_z1.expand(K, -1)
        log_w = torch.clamp(log_w, min=-50.0, max=50.0)
        w = torch.exp(log_w) #(K, batch_size)

        log_vi = log_p_z1_z + log_p_z_y - log_q_y_zc - log_q_z_z1 #(K, batch_size)

        weighted = (w*log_vi).mean(dim=0) #(batch_size,) (particles term mean)

        loss = alpha*minus_log_q_y_z1 - weighted #(batch_size, )

        loss = loss.mean()

        diagnostics = {
            "loss": loss.item(),
            "recon": -log_p_z1_z.mean(0), #mean over MC samples
            "-log_p_z_y": -log_p_z_y.mean(0),
            "log_q_y_zc": log_q_y_zc.mean(0),
            "log_q_y_z1": log_q_z_z1.mean(0),
            "class_loss": alpha*minus_log_q_y_z1
        }

        return loss, diagnostics