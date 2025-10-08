#%%
import torch
import torch.distributions as dist


class RiemannianNormal(dist.Distribution):
    arg_constraints = {'sigma': dist.constraints.positive}

    def __init__(self, mu, sigma, manifold, learn_sigma = False, validate_args = None):
        assert not torch.isnan(mu).any(), "mu contains Nan value(s)"
        assert not torch.isnan(sigma).any(), "sigma contains Nan value(s)"
        assert mu.shape[0] == sigma.shape[0], "mu and sigma have unaligned shape in batch dimension"
        
        self.mu = mu #mu_theta_zi for RiemGauss decoder
        self.sigma = sigma #sigma_theta_zi for RiemGauss decoder
        self.manifold = manifold
        batch_size = self.mu.size()
        self.N = mu.shape[-1] #mu is batch_size,N,N
        self.device = self.mu.device
        self.learn_sigma = learn_sigma

        super(RiemannianNormal, self).__init__(batch_size, validate_args=validate_args)

    def sample(self, shape=torch.Size()):
        #we dont need for now
        # with torch.no_grad():
        #     return self.rsample(shape)
        pass

    def log_prob(self, x_input):
        """
            log prob of Riemannian Gaussian
        """
        # mean = self.mu.expand(x_input.shape)
        squared_d = self.manifold.squared_distance(self.mu, x_input)

        eta= -1/(2*self.sigma.pow(2)) #see if we want to learn sigma_theta after

        log_prob = eta*squared_d

        if self.learn_sigma:
            log_Z_sigma_theta = log_z_beta(self.sigma, self.N) # + log(\omega_N) which is cte in sigma
            log_prob = log_prob - log_Z_sigma_theta
        
        assert squared_d.shape == self.sigma.shape or self.sigma.numel() == 1, \
        f"Shapes don't align: squared_d {squared_d.shape}, sigma {self.sigma.shape}"


        return log_prob

## Eq. 11 from
# Said, Salem, Simon Heuveline, and Cyrus Mostajeran. 
# "Riemannian statistics meets random matrix theory: 
# Toward learning from high-dimensional covariance matrices." 
# IEEE Transactions on Information Theory 69.1 (2022): 472-481.

cache_trilog_at_one = {}

def get_trilog_at_one(K):
    if K not in cache_trilog_at_one:
        k = torch.arange(1, K + 1, dtype=torch.float64)  
        cache_trilog_at_one[K] = float((1.0 / (k ** 3)).sum().item())
    return cache_trilog_at_one[K]

def trilog_fct(eta, K=1000):
    k = torch.arange(1, K + 1, device=eta.device, dtype=eta.dtype)
    terms = (eta.unsqueeze(-1) ** k) / (k ** 3) #shape batch_size, K
    return terms.sum(-1)

def phi_fct(xi, K=1000):
    trilog_exp = trilog_fct(torch.exp(-xi), K)
    trilog_one = torch.tensor(get_trilog_at_one(K), device=xi.device, dtype=xi.dtype)
    return xi / 6 - (trilog_exp - trilog_one) / (xi * xi)

def log_z_beta(sigma, N, beta=1):
    """
    sigma: std deviation
    N: dimension of NxN cov matrix 
    beta: 1 for real SPD matrices
          2 for complex SPD matrices
          4 for quaternion SPD matrices
    """
    t = N*sigma*sigma

    return N*N*0.5*beta*phi_fct(beta*t/2) #TODO confirm this with Salem