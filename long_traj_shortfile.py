"""
Simulate long trajectories with Metropolis for different particle numbers and
compute their static structure factor.
"""

import os, datetime
import numpy as np
from joblib import Parallel, delayed

from basic_functions import compute_Sk, proposal_exclud_vol, run_Metropolis

n_sites = 100
delta = 5

seed = 1

sites = np.arange(n_sites)
list_n_particles = np.arange(delta, n_sites, delta)

rng = np.random.default_rng(seed)

n_particles = list_n_particles[1]

proposal_full = {'fun': proposal_exclud_vol, 'args': (n_sites, 5/n_particles)}  # n_sites, 0.3, 0.7)}
energy_function_full = {'fun': lambda x : 0, 'args': ()}

def run_and_compute_Sk(n_particles, sites, n_steps):

    print('n. particles: ', n_particles)
    
    x0 = rng.choice(np.shape(sites)[0], size=n_particles, replace=False)
    
    traj, ene, av_alpha = run_Metropolis(x0, n_steps=n_sites, proposal=proposal_full, energy_function=energy_function_full)
    
    k, Sk, Sk_list = compute_Sk(n_sites, n_particles, traj)

    return Sk

output = Parallel(n_jobs=3)(delayed(run_and_compute_Sk)(n, sites, 1000) for n in list_n_particles)

s = datetime.datetime.now()
date = s.strftime('%Y_%m_%d_%H_%M_%S')
path = 'Result_' + date

os.mkdir(path)

for i in range(len(output)):
    np.save(path + 'Sk_' + str(i), output[i])