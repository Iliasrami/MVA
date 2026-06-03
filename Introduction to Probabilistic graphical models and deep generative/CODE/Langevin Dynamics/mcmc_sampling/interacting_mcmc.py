import numpy as np
import torch
import matplotlib.pyplot as plt
import tqdm

from dataset import sampling, densities, scores, visualisation

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

def annealed_langevin(ncsn, step_size, noise_levels, n_steps, n_samples, **kwargs):
    """
    Annealed Langevin sampling.
    ----------
    Parameters:
    ncsn: Noise conditional score network
    step_size: Step size of the discretization (epsilon)
    noise_levels: Sequence of noise levels (sigma_1, ..., sigma_L)
    n_steps: Number of steps of the discretization at each noise level (T)
    n_samples: Number of samples to generate
    Returns:
    samples_hist[-1]: samples of the target distribution (sigma_L)
    samples_hist: history of the samples at each noise level (sigma_1, ..., sigma_L)
    """
    dim = 2
    if type(noise_levels) == list:
        L = len(noise_levels)
    elif type(noise_levels) == np.ndarray:
        L = noise_levels.shape[0]
    elif type(noise_levels) == torch.Tensor:
        L = noise_levels.shape[0]
        noise_levels = noise_levels.detach().cpu().numpy()
    else:
        raise ValueError('noise_levels must be a list or a numpy array')
    samples = np.zeros((n_samples, dim))
    # History of the samples at each noise level (L * n_samples * dim)
    samples_hist = np.zeros((L+1, n_samples, dim))
    data_type = kwargs['data_type']
    if data_type == 'gmm':
        x_0 = np.random.uniform(-5, 15, size=(n_samples, 2))
    elif data_type == 'star':
        x_0 = np.random.uniform(-3, 3, size=(n_samples, 2))
    samples_hist[0] = x_0
    for i in tqdm.tqdm(range(L)):
        # Annealing schedule for the step size
        alpha = step_size * noise_levels[i]**2 / noise_levels[-1]**2
        for j in range(n_steps):
            z = np.random.randn(*x_0.shape)
            idx_level = torch.tensor(np.ones((n_samples,)) * i).long().to(device)
            score_results = ncsn(torch.tensor(x_0, requires_grad=True).float().to(device), idx_level).detach().cpu().numpy()
            x_new = x_0 + alpha / 2 * score_results + np.sqrt(alpha) * z
            x_0 = x_new
        samples_hist[i+1] = x_0
    return samples_hist[-1], samples_hist






def transition_kernel(x, step_size, model):
    """
    Computes the transition kernel of the Gaussian distribution centered in x + step_size / 2 * score(x) 
    and covariance matrix step_size * I.
    --------------------
    Parameters:
    x : mean of the Gaussian distribution
    step_size : step size of the discretization
    model : score function of the target distribution
    Returns:
    x_prop : a candidate sample
    """
    d = x.shape[0]
    x_prop = x.copy()
    score_matching = model(torch.tensor(x_prop).float().to(device)).cpu().detach().numpy()
    x_prop = x_prop + step_size / 2 * score_matching + np.random.randn() * np.sqrt(step_size)
    return x_prop

def hasting_metropolis_step(z_O, step_size, model, temperature, transition_kernel, target_distribution):
    """
    Hasting-Metropolis filtering step.
    --------------------
    Parameters:
    z_O : initial point
    step_size : step size of the discretization
    model : score function of the target distribution
    temperature : temperature of the target distribution
    transition_kernel : proposal transition kernel
    target_distribution : function returning the log-density of the target distribution
    """
    z = z_O.copy()
    acceptance = 0
    # Proposal
    z_prop = transition_kernel(z, step_size, model)
    # Log-likelihood of the proposal
    log_alpha = min(0, (target_distribution(z_prop) - target_distribution(z)) / temperature)
    if np.random.rand() < np.exp(log_alpha):
        z = z_prop
        acceptance += 1
    return z, acceptance


def parallel_tempering(temperatures, max_iter, model, *args):
    """
    Parallel tempering algorithm.
    --------------------
    Parameters:
    z_O : initial point
    temperatures : temperatures of the chains
    max_iter : maximum number of iterations
    transition_kernel : proposal transition kernel
    target_distribution : function returning the log-density of the target distribution
    """

    z_0 = np.random.uniform(-5, 15, size=(2,))
    d = z_0.shape[0]
    n_chains = temperatures.shape[0]
    trajectories = np.zeros((max_iter, n_chains, d))
    mh_acceptances = np.zeros((max_iter, n_chains))
    swap_acceptances = np.zeros(max_iter)
    step_sizes = np.zeros(n_chains)
    for i in range(n_chains):
        step_sizes[i] = 0.25 * np.sqrt(temperatures[i])
    # Initializations
    trajectories[0] = z_0

    mus, sigmas, alpha = args

    def target_distribution(x, mus = mus, sigmas = sigmas, alpha = alpha):
        return densities.gmm_log_density(x, mus, sigmas, alpha)

    for n in range(1, max_iter):
        candidates = np.zeros((n_chains, d))
        # MH step for each chain
        for k in range(n_chains):
            candidates[k, :], mh_acceptances[n, k] = hasting_metropolis_step(trajectories[n-1, k], step_sizes[k], model,
                                                                             temperatures[k], transition_kernel, 
                                                                             target_distribution)
        
        # Temperatures selection
        i = np.random.randint(1, n_chains-1)
        if np.random.rand() < 0.5:
            j = i + 1
        else:
            j = i - 1 
        # Swap acceptance probability
        num = target_distribution(candidates[j]) / temperatures[i] + target_distribution(candidates[i]) / temperatures[j]
        den = target_distribution(candidates[i]) / temperatures[i] + target_distribution(candidates[j]) / temperatures[j]
        alpha = min(0, num - den)
        # Swap
        if np.random.rand() < np.exp(alpha):
            trajectories[n, i], trajectories[n, j] = candidates[j], candidates[i]
            swap_acceptances[n] = 1
        else:
            trajectories[n, i], trajectories[n, j] = candidates[i], candidates[j]

        remaining_chains = [k for k in range(n_chains) if k != i and k != j]
        # MH update for the remaining chains
        for k in remaining_chains:
            trajectories[n, k] = candidates[k]

    return trajectories, mh_acceptances, swap_acceptances

