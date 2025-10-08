#%%
"""
Betaprime kernel and two-samples test implementation of 
"Invariant Kernels on the space of Complex Covariance Matrices", Said et.al, 2025
Code provided by the authors.
"""

import numpy as np
import math 
from decimal import Decimal, getcontext
import torch
############################################################################
#Modified implementation: to circumvent cone_gamma overflow for large N
#
#Obs: it seems that for big N "acceptance" has a (fixed, independently of the samples)
#     very high value. In this case, would the test always fail (null fail to be rejected)?
############################################################################
#precision
getcontext().prec = 100  

#compute special gamma function in log-space
def cone_log_gamma_decimal(N, z): #tested: check
    z = Decimal(z)
    log_2pi = Decimal(2 * math.pi).ln()

    log_prod = Decimal(N * (N - 1)) / 2 * log_2pi
    for k in range(N):
        log_gamma = Decimal(math.lgamma(float(z - k + 1)))
        log_prod += log_gamma

    return log_prod  # log(gamma)

#compute kernel evaluation in log-space
def betaprime(A, B): #tested: check
    N = A.shape[0]
    alpha = N

    if not N == B.shape[0]:
        raise ValueError('Matrices must have same size')
    if not alpha > N - 1:
        raise ValueError(f'alpha must be > {N - 1}')

    log_gamma = cone_log_gamma_decimal(N, 2*alpha)  # log(gamma)

    #slogdet to compute logs safely
    _, ldetA = np.linalg.slogdet(A)
    _, ldetB = np.linalg.slogdet(B)
    _, ldetAB = np.linalg.slogdet(A + B)

    log_dets = alpha * (ldetA + ldetB - 2 * ldetAB)
    log_result = float(log_gamma) + log_dets

    return np.exp(log_result)

#general upper bound derived analytically. can be achieved for any range of eigenvals
def max_betaprime(N, alpha): #tested: check
    log_gamma = cone_log_gamma_decimal(N, 2 * alpha)
    log_denom = Decimal(N * alpha) * Decimal(4).ln()
    log_result = log_gamma - log_denom
    return log_result.exp()

#in lemma, that is function h(z1,z2)
def aid_func(f, x1, y1, x2, y2):
    return f(x1, x2) - f(x1, y2) - f(x2, y1) + f(y1, y2)

def eff_kernel_test(func, level, sampleA, sampleB):  #tested: check
    m = int(np.floor(len(sampleA)/2))
    N = sampleA[0].shape[0]

    if not len(sampleB) == len(sampleA):
        raise ValueError('Samples must be of same size')

    #compute upper bound
    if func == betaprime:
        upper = max_betaprime(N, N)
    else:
        raise NotImplementedError("This kernel is not yet implemented.")

    #compute test statistic
    aid_sum = 0
    for i in range(m):
        aid_sum += aid_func(func, sampleA[2*i], sampleB[2*i], sampleA[2*i+1], sampleB[2*i+1])

    test_stat = aid_sum/m

    #compute acceptance region
    acceptance = np.sqrt(-np.log(level)) * (4*float(upper))/np.sqrt(m) 
    #obs. acceptance depends only on the level, the size of the data samples and the dimension of the mat

    #return test result
    if test_stat < acceptance:
        result = 0  # Fail to reject null hypothesis
    else:
        result = 1  # Reject null hypothesis

    return result, test_stat, acceptance, upper

def gen_block_mats(mat, N, l):
    """
    Returns all the lxl symmetric block-matrices within a NxN symmetric matrix mat
    N is the size of the (original) covariance matrix mat, l is the size of the block matrices
    """
    block_mats = torch.zeros((N-l+1,l,l))
    for i in range(N-l+1):
        block_mats[i,:,:] = mat[i:l+i, i:l+i]
    return block_mats

def gen_bblock_mats(mats, l):
    """
    Batched generation of block_matrices.
    Returns all the lxl symmetric block-matrices within each NxN symmetric matrix mat
    N is the size of the (original) covariance matrices, l is the size of the block matrices
    Inputs: 
        mat: BxNxN symmetric matrices
        l: size of new block matrices
    Output: 
        Bx(N-l+1)xlxl tensor with symmetric block matrices
    """
    B, N, _ = mats.shape
    block_mats = np.zeros((B, N-l+1,l,l))
    for i in range(N-l+1):
        block_mats[:, i,:,:] = mats[:, i:l+i, i:l+i]
    return block_mats

def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return torch.allclose(a, a.T, rtol=rtol, atol=atol)    

if __name__ == '__main__':
    from pyriemann.datasets import generate_random_spd_matrix

    np.random.seed(3)

    nsamples = 1000
    size = 32
    sampleA = np.array([generate_random_spd_matrix(n_dim=size) for _ in range(nsamples)])
    sampleB = np.array([generate_random_spd_matrix(n_dim=size) for _ in range(nsamples)])

    #test several block matrices sizes:
    for l in [2, 3, 4, 5, 6, 8, 10]:
        block_sampleA = gen_bblock_mats(sampleA, l=l).reshape(-1,l,l)
        block_sampleB = gen_bblock_mats(sampleB, l=l).reshape(-1,l,l)

        result, test_stat, acceptance, upper = eff_kernel_test(betaprime, level=0.05, sampleA=block_sampleA, sampleB=block_sampleB)
        print("blocks with l={} test: ".format(l))
        print("result: {}, acc/test_stat: {}, test_stat: {}, acceptance: {}, upper: {}".format(result, float(acceptance)/test_stat, test_stat, acceptance, upper))
        print("\n\n")
        
    #original test
    result, test_stat, acceptance, upper = eff_kernel_test(betaprime, level=0.05, sampleA=sampleA, sampleB=sampleB)
    print("original test: ")
    print("result: {}, acc/test_stat: {}, test_stat: {}, acceptance: {}, upper: {}".format(result, float(acceptance)/test_stat, test_stat, acceptance, upper))
    print("\n\n")
# %%
# ############################################################################
# #Original implementation: cone_gamma output might overflow for large N
# ############################################################################
# #compute special gamma function
# def cone_gamma(N, z):
#     log_prod = (N * (N - 1) / 2) * np.log(2 * np.pi)
#     for k in range(N):
#         log_prod += math.lgamma(z - k + 1)
#     return np.exp(log_prod)

# #compute kernel evaluation
# def betaprime(A,B):
#     print("In betaprime!")
#     print(A, B)
#     N = A.shape[0]
#     alpha = N
    
#     if not N == B.shape[0]:
#         raise ValueError('Matrices must have same size')
#     if not alpha > N-1:
#         raise ValueError(f'alpha must be > {N-1}')
    
#     gamma = cone_gamma(N, 2*alpha) #--> can cause overflow

#     '''
#     # Compute determinants
#     _ , ldetA = np.linalg.slogdet(A)
#     _ , ldetB = np.linalg.slogdet(B)
#     _ , ldetAB = np.linalg.slogdet(A + B)
     
#     dets = np.exp(alpha*(ldetA-2*ldetAB+ldetB))
#     '''

#     detA = np.linalg.det(A)
#     detB = np.linalg.det(B)
#     detAB = np.linalg.det(A+B)

#     dets = (detA * detB /(detAB)**2)**alpha
    
#     return gamma * dets

# #general upper bound derived analytically. can be achieved for any range of eigenvals
# def max_betaprime(N, alpha):
    
#     gamma = cone_gamma(N, 2*alpha)
#     return gamma/ (4**(N*alpha))

# #in lemma, that is funciton h(z1,z2)
# def aid_func(f,x1,y1,x2,y2):
#     return f(x1,x2) - f(x1,y2) - f(x2,y1) + f(y1,y2)

# ### test from lemma 14 in ibid. (computationally more efficient)
# def eff_kernel_test(func, level, sampleA, sampleB): # TODO: Extend it to non equal sample sizes
    
#     m = int(np.floor(len(sampleA)/2))
#     N = sampleA[0].shape[0]
    
#     if not len(sampleB)==len(sampleA):
#         raise ValueError('Samples must be of same size')

    
#     #compute upper bound
#     if func == betaprime:
#         upper = max_betaprime(N,N)
#     else:
#         raise NotImplementedError

    
#     #compute test statistic
#     aid_sum = 0
    
#     for i in range(m):
#         aid_sum += aid_func(func, sampleA[2*i], sampleB[2*i], sampleA[2*i+1], sampleB[2*i+1])
        
#     test_stat=aid_sum/m
#     #print(f'Test statistic: {test_stat:.2e}')
#     #compute acceptance region
#     acceptance = np.sqrt(-np.log(level)) * (4*upper)/np.sqrt(m) #for H_0: samples from same distribution
#     #print(f'Acceptance bound is {acceptance:.2e}.')
#     # TODO: Use the correct acceptance region for the linear version of the MMD_k test

    
#     #return test result
#     if test_stat < acceptance:
#         result = 0 #'Fail to reject null hypothesis'
#     else:
#         result = 1 #'Reject null hypothesis'
    
#     return result, test_stat, acceptance #check that thrm 15 yields same acceptance region

# if __name__ == '__main__':
#     from pyriemann.datasets import generate_random_spd_matrix

#     np.random.seed(3)

#     nsamples = 1000
#     size = 10
#     sampleA = np.array([generate_random_spd_matrix(n_dim=size) for _ in range(nsamples)])
#     sampleB = np.array([generate_random_spd_matrix(n_dim=size) for _ in range(nsamples)])

#     result_o, test_stat_o, acceptance_o = eff_kernel_test(betaprime, level=0.05, sampleA=sampleA, sampleB=sampleB)
