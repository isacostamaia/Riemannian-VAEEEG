import numpy as np

def frange_cycle_linear(n_iter, start=0.0, stop=1.0,  n_cycle=4, ratio=0.5):
    """
    Code for "Cyclical Annealing Schedule: A Simple Approach to Mitigating KL Vanishing",
    Fu et.al, 2019
    Creates annealing for beta in KL divergence
    """
    L = np.ones(n_iter) * stop
    period = n_iter/n_cycle
    step = (stop-start)/(period*ratio) # linear schedule

    for c in range(n_cycle):
        v, i = start, 0
        while v <= stop and (int(i+c*period) < n_iter):
            L[int(i+c*period)] = v
            v += step
            i += 1
    return L 

def monotonic_schedule(n_iter, start=0.0, stop=1.0, changes_at=0.5): 
    """
    Implementation of monotonic annealing for beta in KL divergence
    Proposed in "Generating Sentences from a Continuous Space",
    Bowman et. al, 2016
    """
    change_it = int(changes_at*n_iter)
    first_section = np.linspace(start, stop, change_it)
    second_section = [stop]*(n_iter-change_it)
    return np.array(list(first_section) + second_section)

def linear_annealing(n_iter, n_warmup=20):
    betas = np.linspace(0, 1, n_warmup).tolist()
    betas += [1.0] * (n_iter - n_warmup)
    return betas