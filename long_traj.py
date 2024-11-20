"""
Simulate long trajectories with Metropolis for different particle numbers and
compute their static structure factor.
"""

import os, datetime, sys
import pandas
import numpy as np
# from joblib import Parallel, delayed

from basic_functions import proposal_exclud_vol, energy_fun, run_Metropolis, compute_Sk

n_sites = int(sys.argv[1])
n_steps = int(sys.argv[2])

# choose between selecting a set of n_particles `if_delta = True` or just one `if_delta = False` (to parallelize)
if_delta = False

if if_delta: delta = int(sys.argv[3])
else: n_particles = int(sys.argv[3])

seed = int(sys.argv[4])
which_strategy = 1

#%% define: sites, n_particles, rng, proposal, energy

sites = np.arange(n_sites)

if if_delta: list_n_particles = np.arange(delta, n_sites, delta)

rng = np.random.default_rng(seed)

energy_function_full = {'fun': lambda x : 0, 'args': ()}

#%% mkdir and save input values

if if_delta:
    s = datetime.datetime.now()
    date = s.strftime('%Y_%m_%d_%H_%M_%S_%f')
else:
    date = 1996


path = '../Result_' + date

if not os.path.exists(path):
    os.mkdir(path)
else:
    assert not if_delta, 'possible overwriting'

input_values = {'n_sites': n_sites, 'n_steps': n_steps, 'delta': delta, 'seed': seed, 'strategy': which_strategy}
temp = pandas.DataFrame(list(input_values.values()), index=list(input_values.keys()), columns=[date]).T
temp.to_csv(path + '/input')

#%% repeat over different n. of particles: Metropolis and compute Sk

energies_pen = {'1': 5, '2': -7, '3': -2}
energy_function_full = {'fun': energy_fun, 'args': (energies_pen, n_sites)}

# if you want to parallelize with Parallel delayed
# def compute(sites, n_particles, delta, which_strategy, proposal_exclud_vol, energy_function_full, n_steps, path):

if not if_delta: list_n_particles = [n_particles]

for n_particles in list_n_particles:
    
    x0 = rng.choice(np.shape(sites)[0], size=n_particles, replace=False)

    n_sites = len(sites)

    # this value delta/(2*n_particles) for the probability that each particle moves
    # corresponds to setting a value of delta/2 for the av. n. of particles which actually move 
    proposal_full = {'fun': proposal_exclud_vol, 'args': (n_sites, delta/(2*n_particles), which_strategy)}  # n_sites, 0.3, 0.7)}

    traj, ene, av_alpha = run_Metropolis(x0, n_steps=n_steps, proposal=proposal_full, energy_function=energy_function_full)

    k, Sk = compute_Sk(n_sites, n_particles, traj)

    np.save(path + '/traj_' + str(n_particles), traj)
    np.save(path + '/Sk_' + str(n_particles), Sk)
