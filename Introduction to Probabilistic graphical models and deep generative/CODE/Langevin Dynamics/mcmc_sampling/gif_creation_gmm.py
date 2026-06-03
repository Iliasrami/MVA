import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import random
import math
import argparse
from tqdm import tqdm
plt.style.use('seaborn-poster')

import sys, os
import torch.nn as nn

local_path = '/Users/halvardbariller/Desktop/M2_MVA/_SEMESTER_1/PGM/Project/Score-matching-project-'
sys.path.append(local_path)


import dataset
from dataset import sampling, densities, scores, visualisation
# import score_matching
# from score_matching import toy_models, learning_objectives, score_visualisation
import mcmc_sampling
from mcmc_sampling import langevin, dynamics_plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


parser = argparse.ArgumentParser()
# parser.add_argument('--samples', type=str, required=True, help='Path to the .npy file with samples.')
# parser.add_argument('--start', type=int, required=True, help='Start frame number.')
# parser.add_argument('--n', type=int, required=True, help='Number of frames.')
parser.add_argument('--iterations', type=int, required=True, help='Number of iterations.')

args = parser.parse_args()



mus = [np.array([0, 0]), np.array([10,10])]
sigmas = [np.eye(2), np.eye(2)]
alphas = [0.5, 0.5]

step = 1e-2
iterations = args.iterations

_, samples = langevin.euler_maruyama_exact_scores(np.array([4,4]), scores.gaussian_mixture_score, 
                                                            step, iterations, mus, sigmas, alphas)

lim = 5

r = np.linspace(-lim, lim+10, 1000)
x, y = np.meshgrid(r, r)
z = np.vstack([x.flatten(), y.flatten()]).T

q0 = densities.gmm_density(z, mus, sigmas, alphas)

plt.rcParams["font.family"] = "serif"
fig, axn = plt.subplots(ncols=2, figsize=(15, 8))

axn[0].pcolormesh(x, y, q0.reshape(x.shape),
                           cmap='viridis')
axn[0].set_aspect('equal', adjustable='box')
axn[0].set_xlim([-lim, lim+10])
axn[0].set_ylim([-lim, lim+10])
axn[0].set_title('True Density')

cmap = matplotlib.cm.get_cmap('viridis')
bg = cmap(0.)
axn[1].set_facecolor(bg)
axn[1].set_aspect('equal', adjustable='box')
axn[1].set_xlim([-lim, lim+10])
axn[1].set_ylim([-lim, lim+10])
axn[1].set_title('Empirical Density')

fig.suptitle('Langevin Dynamics for GMM', fontsize=40)

line, = axn[0].plot([], [], lw=2, c='#f3c623')
scat = axn[0].scatter([], [], c='#dd2c00', s=150, marker='*')

def init():
    line.set_data([], [])
    return line,

def random_walk(i):
    i += 1
    if i <= 30:
        z = samples[:i]
    else:
        z = samples[i-30:i]
    line.set_data(z[:, 0], z[:, 1])
    scat.set_offsets(z[-1:])

    axn[1].clear()
    axn[1].set_aspect('equal', adjustable='box')
    axn[1].hist2d(samples[:i, 0], samples[:i, 1], cmap='viridis', rasterized=False, bins=150, density=True)
    axn[1].set_xlim([-lim, lim+10])
    axn[1].set_ylim([-lim, lim+10])
    axn[1].set_title('Empirical Density')
    return line, scat, #


from_frame = 1
upto_frame = len(samples) + from_frame
base_name = 'gmm_langevin'

anim = animation.FuncAnimation( fig = fig, blit=True, init_func=init, func = random_walk,
                                     interval = 10, frames=range(from_frame, upto_frame))

anim.save('gmm_langevin.gif', writer='imagemagick', dpi=200, 
          progress_callback = lambda i, n: print(f'Saving frame {i} of {n}') if i % 10 == 0 else None)

print('Done!')