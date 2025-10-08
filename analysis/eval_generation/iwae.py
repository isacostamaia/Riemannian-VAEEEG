#%%
import os
import sys
from pathlib import Path

# Automatically get the root of the project (one level up from this script)
project_root = Path(__file__).resolve().parents[1]

#Add root to sys.path so you can import rvae, datasets, etc.
sys.path.append(str(project_root))

#Change working directory to root (for file paths)
os.chdir(project_root)


import numpy as np
import torch
import torch.nn.functional as F


def mult_stdGauss(z): #p(z) in M1
    return -0.5 * z.pow(2).sum(dim=-1) - 0.5 * z.size(1) * np.log(2 * np.pi)

def mult_Gauss(z, mean, logvar): #in M1
    var = torch.exp(logvar)
    return -0.5 * (((z - mean) ** 2) / var).sum(dim=-1) \
           - 0.5 * logvar.sum(dim=-1) \
           - 0.5 * z.size(-1) * np.log(2 * np.pi)

def inference_IWAE(model, data_loader, K=512):
    """
    Given a (trained) model and a (test) data_loader, 
    returns liwae_scores, the list of IWAE scores (lower bound of 
    the sample log-likelihood p(x_i) with p(x_i) >= lwiae_scores[i])
    """
    #thought for M1 alone
    model = model.to(model.device)
    model.eval()

    liwae_scores = []
    with torch.no_grad():
        for x, y, dom in data_loader:
            x = x.to(model.device)
            log_w_iks = []
            mean, logvar = model.encode(x,dom)
            for k in range(K):
                z = model.reparametrize(mean, logvar)
                mu_theta_zi = model.decode(z)
                #running test std dev (estimated in during batch norm)
                sigma_theta_zi = model.get_spd_bn(dom).running_var_test.clone().detach().sqrt().expand(mu_theta_zi.shape[0])
                px_z_dist = model.px_z(mu=mu_theta_zi,
                                sigma = sigma_theta_zi,
                                manifold = model.xVar.manifold)
                lpx_z = px_z_dist.log_prob(x)
                if model.pz == "EucStdGaussian":
                    lpz = mult_stdGauss(z)
                else:
                    raise NotImplementedError("Prior pz other than Euclidean StdGaussian hasn't been implemented yet")
                if model.qz_x == "EucGaussian":
                    lqz_x = mult_Gauss(z, mean, logvar)
                
                #unnormalized importance weights w_ik = px_z*pz/qz_x
                log_w_ik = lpx_z + lpz - lqz_x
                log_w_iks.append(log_w_ik.cpu()) 
            log_w_iks = torch.stack(log_w_iks) #shape = K, batch_size

            #w_i = (1/K) * \sum_k px_z*pz/qz_x
            log_w_i = torch.logsumexp(log_w_iks, dim=0) - np.log(K) #average over K IWAE samples, not over the batch
            
            #log-likelihood for a single data sample ln p(x_i) ~= ln(w_i)
            liwae_scores.extend(log_w_i.numpy()) #add batch_size elements
    return liwae_scores

def inference_IWAE_vec(model, data_loader, K=512):
    """
    Vectorized IWAE inference with batched K samples.
    Returns liwae_scores: list of IWAE lower bounds (one per sample).
    """
    model = model.to(model.device)
    model.eval()
    liwae_scores = []

    with torch.no_grad():
        for x, y, dom in data_loader:
            x = x.to(model.device)                   #shape= batch_size, N, N

            mean, logvar = model.encode(x, dom)      #shape= batch_size, latent_dim

            z = model.reparametrize(mean, logvar, K)  #shape= (K, batch_size, latent_dim)
            K_, B, D = z.shape  #(K, batch_size, latent_dim)

            #decoder: z -> mu_theta(z)
            z_flat = z.view(K * B, D)
            N = x.shape[-1]  #cvariance mat size
            mu_theta_zi = model.decode(z_flat).view(K, B, N, N)  #shape= (K, batch_size, N, N)

            #loglikelihood p(x | z)
            x_repeat = x.unsqueeze(0).expand(K, B, N, N)  # shape= (K, batch_size, N, N)
            sigma_theta_zi = model.get_spd_bn(dom).running_var_test.clone().detach().sqrt()
            sigma_theta_zi = sigma_theta_zi.expand(B).unsqueeze(0).expand(K, B)  #shape= (K, batch_size)

            px_z_dist = model.px_z(
                mu=mu_theta_zi, 
                sigma=sigma_theta_zi, 
                manifold=model.xVar.manifold
            )
            lpx_z = px_z_dist.log_prob(x_repeat)  #shape= (K, batch_size)

            #logprior log p(z)
            if model.pz == "EucStdGaussian":
                lpz = mult_stdGauss(z.view(-1, D)).view(K, B)
            else:
                raise NotImplementedError("Prior p(z) other than EucStdGaussian is not implemented.")

            #posterior log q(z|x)
            if model.qz_x == "EucGaussian":
                mean_rep = mean.unsqueeze(0).expand(K, B, D)
                logvar_rep = logvar.unsqueeze(0).expand(K, B, D)
                lqz_x = mult_Gauss(z, mean_rep, logvar_rep)  #shape= (K, batch_size)
            else:
                raise NotImplementedError("Posterior q(z|x) other than EucGaussian is not implemented.")

            #log w_ik = log p(x|z) + log p(z) - log q(z|x)
            log_w_ik = lpx_z + lpz - lqz_x  #shape= (K, batch_size)

            #IWAE = log(mean_k w_ik)
            log_w_i = torch.logsumexp(log_w_ik, dim=0) - np.log(K)  #shape= (batch_size,)

            liwae_scores.extend(log_w_i.cpu().numpy())
    
    return liwae_scores

def inference_xyIWAE_m1m2_vec(m1, m2, data_loader, K=512):
    """
    Vectorized IWAE bound for p(x,y) using M1+M2 inference with batched K samples.
    Returns liwae_scores: list of IWAE lower bounds (one per sample).
    """

    m1 = m1.to(m1.device)
    m2 = m2.to(m2.device)
    m1.eval()
    m2.eval()
    liwae_scores_m1 = []
    liwae_scores_m2 = []

    with torch.no_grad():
        for x, y, dom in data_loader:
            x = x.to(m1.device)                   #shape= batch_size, N, N
            
            ######## M1 ###########
            mean, logvar = m1.encode(x, dom)      #shape= batch_size, latent_dim

            z1 = m1.reparametrize(mean, logvar, K)  #shape= (K, batch_size, latent_dim)
            K_, B, D = z1.shape  #(K, batch_size, latent_dim)

            #decoder: z1 -> mu_theta(z1)
            z1_flat = z1.view(K * B, D)
            N = x.shape[-1]  #cvariance mat size
            m1_latent_params = m1.decode(z1_flat)

            if type(m1_latent_params) == tuple: #learning logvar in decoder
                mu_theta_zi = m1_latent_params[0].view(K, B, N, N)  #shape= (K, batch_size, N, N)
                logvar_theta_zi = m1_latent_params[1].view(K, B)  #shape= (K, batch_size)
                sigma_theta_zi = torch.exp(logvar_theta_zi*0.5).squeeze()
            else: #only decoder mean is learned
                mu_theta_zi =  m1_latent_params.view(K, B, N, N)
                sigma_theta_zi = m1.get_spd_bn(dom).running_var_test.clone().detach()
                sigma_theta_zi = sigma_theta_zi.expand(B).unsqueeze(0).expand(K, B)  #shape= (K, batch_size)

            #loglikelihood p(x | z)
            x_repeat = x.unsqueeze(0).expand(K, B, N, N)  # shape= (K, batch_size, N, N)

            px_z1_dist = m1.px_z(
                mu=mu_theta_zi, 
                sigma=sigma_theta_zi, 
                manifold=m1.xVar.manifold
            )
            lpx_z1 = px_z1_dist.log_prob(x_repeat)  #shape= (K, batch_size)

            #logprior log p(z)
            if m1.pz == "EucStdGaussian":
                lpz1 = mult_stdGauss(z1.view(-1, D)).view(K, B)
            else:
                raise NotImplementedError("Prior p(z) other than EucStdGaussian is not implemented.")

            #posterior log q(z|x)
            if m1.qz_x == "EucGaussian":
                mean_rep = mean.unsqueeze(0).expand(K, B, D)
                logvar_rep = logvar.unsqueeze(0).expand(K, B, D)
                lqz1_x = mult_Gauss(z1, mean_rep, logvar_rep)  #shape= (K, batch_size)
            else:
                raise NotImplementedError("Posterior q(z|x) other than EucGaussian is not implemented.")

            #log w_ik = log p(x|z) + log p(z) - log q(z|x)
            log_w_ik_m1 = lpx_z1 + lpz1 - lqz1_x  #shape= (K, batch_size)

            #IWAE = log(mean_k w_ik)
            log_w_i_m1 = torch.logsumexp(log_w_ik_m1, dim=0) - np.log(K)  #shape= (batch_size,)

            liwae_scores_m1.extend(log_w_i_m1.cpu().numpy())

            ######## M2 ###########
            #log p(y)
            logp_y = m2.logp_y(y).unsqueeze(0).expand(K, B) #prior log probability of the actual label #shape = K, batch_size

            #log q(z2k|z1k,y)   -- (q(zk|x,y) on the paper)
            y_rep = y.unsqueeze(0).expand(K, B) #shape = K,batch_size (class is encoded in a single digit)
            y_rep_onehot = F.one_hot(y_rep, num_classes=m2.y_dim).float().to(m2.device) #true y shape= K,batch_size,num_classes
            y_rep_onehot_flat = y_rep_onehot.view(K*B,-1)
            z2_mean_flat, z2_logvar_flat = m2.encode_z2(z1_flat, y_rep_onehot_flat)
            z2_mean, z2_logvar = z2_mean_flat.view(K, B, m2.z2_dim), z2_logvar_flat.view(K, B, m2.z2_dim) #shape = K, batch_size, z2_dim
            z2_flat = m2.reparametrize(z2_mean_flat, z2_logvar_flat)
            z2 = z2_flat.view(K, B, m2.z2_dim)
            lqz2_yz1 = mult_Gauss(z2, z2_mean, z2_logvar)  #shape  = K, batch_size

            #log p(z1|y,z2k)  --  (p(x|y,zk on the paper)
            z1_mean_flat, z1_logvar_flat = m2.decode_z1(z2_flat, y_rep_onehot_flat) #shape  = K*batch_size, D 
            z1_mean, z1_logvar = z1_mean_flat.view(K, B, D), z1_logvar_flat.view(K, B, D)
            lpz1_yz2 = mult_Gauss(z1,z1_mean,z1_logvar)  #shape  = K, batch_size

            #log p(z2)
            lpz2 = mult_stdGauss(z2.view(-1, m2.z2_dim)).view(K, B)  #shape = K, batch_size

            #log w_ik = log p(z1|y,z2k) + log p(z2k) +log p(y) - log q(z2k|z1k,y)
            log_w_ik_m2 = lpz1_yz2 + lpz2 + logp_y - lqz2_yz1  #shape= (K, batch_size)

            #IWAE = log(mean_k w_ik)
            log_w_i_m2 = torch.logsumexp(log_w_ik_m2, dim=0) - np.log(K)  #shape= (batch_size,)

            liwae_scores_m2.extend(log_w_i_m2.cpu().numpy())

    return liwae_scores_m1, liwae_scores_m2
