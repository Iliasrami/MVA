from . import learning_objectives, toy_models

import matplotlib.pyplot as plt
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from dataset import scores


def plot_estimated_score_gmm(data, clusters, trained_model, mus, sigmas, alphas, difference=False,type=None, sigma_list=None):
    """
    Plots the estimated score of a Gaussian mixture model.
    ----------
    Parameters:
    data: the samples.
    clusters: the clusters of the samples.
    trained_model: the trained score network.
    mus: the means of the gaussians.
    sigmas: the covariance matrices of the gaussians.
    alphas: the mixing proportions of the gaussians.
    difference: if True, plots a heatmap of the difference between the true and estimated score w.r.t. l2 norm.
    type : whether we use anneal dsm or another loss 
    sigma_list : useful when we used anneal dsm because it's the tensor which contains the values of noise sigmas
    """

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    true_score = scores.gaussian_mixture_score(data, mus, sigmas, alphas)
    ax[0,0].scatter(data[:,0], data[:,1], s = 1, c=clusters, alpha=0.5)
    ax[0,0].quiver(data[:,0], data[:,1], true_score[:,0], true_score[:,1], 
                   np.linalg.norm(true_score, axis=1), color='red', alpha=0.5)
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    ax[0,0].set_title('True score for dataset')

    if type=='anneal_denoising_score_matching':
        # labels is the vector of indices corresponding to the sigmas to be selected for data perturbation
        labels = torch.randint(0, len(sigma_list), (data.shape[0],))
        estimated_scores = trained_model(torch.tensor(data).float().to(device),labels).cpu().detach().numpy()
    else:
        estimated_scores = trained_model(torch.tensor(data).float().to(device)).cpu().detach().numpy()
    ax[0,1].scatter(data[:,0], data[:,1], s = 1, c=clusters, alpha=0.5)
    ax[0,1].quiver(data[:,0], data[:,1], estimated_scores[:,0], estimated_scores[:,1], 
                 np.linalg.norm(estimated_scores, axis=1), color='red', alpha=0.5)
    ax[0,1].set_xlabel('x')
    ax[0,1].set_ylabel('y')
    ax[0,1].set_title('Estimated score for dataset')

    ##

    x_grid = np.linspace(-5, 15, 50)
    y_grid = np.linspace(-5, 15, 50)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    scores_grid = scores.gaussian_mixture_score(grid, mus, sigmas, alphas)

    if type=='anneal_denoising_score_matching':
        labels = torch.randint(0, len(sigma_list), (grid.shape[0],))
        estimated_scores_grid = trained_model(torch.tensor(grid, requires_grad=True).float().to(device),labels).detach().cpu().numpy()
    else:
        estimated_scores_grid = trained_model(torch.tensor(grid, requires_grad=True).float().to(device)).detach().cpu().numpy()
    

    ax[1,0].scatter(data[:,0], data[:,1], s = 1, c=clusters, alpha=0.5)
    ax[1,0].quiver(grid[:,0], grid[:,1], scores_grid[:,0], scores_grid[:,1], 
                 np.linalg.norm(scores_grid, axis=1), color='red', alpha=0.5, cmap='autumn_r')
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    ax[1,0].set_title('True score over grid')

    ax[1,1].scatter(data[:,0], data[:,1], s = 1, c=clusters, alpha=0.5)
    ax[1,1].quiver(grid[:,0], grid[:,1], estimated_scores_grid[:,0], estimated_scores_grid[:,1], 
                 np.linalg.norm(estimated_scores_grid, axis=1), color='red', alpha=0.5, cmap='autumn_r')
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')
    ax[1,1].set_title('Estimated score over grid')

    fig.suptitle('Gaussian mixture score')
    fig.tight_layout()
    plt.show()

    if difference:
        fig2, ax2 = plt.subplots(1, 1, figsize=(15, 6))
        difference = np.linalg.norm(scores_grid - estimated_scores_grid, axis=1)
        levels = np.linspace(0, 20, 100)
        ax2.contourf(x_grid, y_grid, difference.reshape(xx.shape), levels = levels, cmap='seismic')
        fig2.colorbar(ax2.contourf(x_grid, y_grid, difference.reshape(xx.shape), levels = levels, cmap='seismic'))
        ax2.scatter(data[:,0], data[:,1], s = 1, c='white', alpha=0.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Difference between true and estimated score')
        fig2.tight_layout()
        plt.show()






def plot_estimated_score_banana(data, trained_model, mu, sigma, difference,type=None,sigma_list=None):
    """
    Plots the estimated score of a banana-shaped distribution.
    ----------
    Parameters:
    data: the samples.
    trained_model: the trained score network.
    mu: the mean of the banana-shaped distribution.
    sigma: the covariance matrix of the banana-shaped distribution.
    difference: if True, plots a heatmap of the difference between the true and estimated score w.r.t. l2 norm.
    type : whether we use anneal dsm or another loss 
    sigma_list : useful when we used anneal dsm because it's the tensor which contains the values of noise sigmas
    """

    fig, ax = plt.subplots(2, 2, figsize=(15, 12))
    true_score = scores.score_banana(data, mu, sigma)
    ax[0,0].scatter(data[:,0], data[:,1], s = 1, alpha=0.5)
    ax[0,0].quiver(data[:,0], data[:,1], true_score[:,0], true_score[:,1], 
                   np.linalg.norm(true_score, axis=1), color = 'red', alpha=0.5)
    ax[0,0].set_xlabel('x')
    ax[0,0].set_ylabel('y')
    ax[0,0].set_title('True score for dataset')

    if type=='anneal_denoising_score_matching':

        #labels is the vector of indices corresponding to the sigmas to be selected for data perturbation
        labels = torch.randint(0, len(sigma_list), (data.shape[0],))
        estimated_scores = trained_model(torch.tensor(data).float().to(device),labels).cpu().detach().numpy()
    else:
        estimated_scores = trained_model(torch.tensor(data).float().to(device)).cpu().detach().numpy()
    ax[0,1].scatter(data[:,0], data[:,1], s = 1, alpha=0.5)
    ax[0,1].quiver(data[:,0], data[:,1], estimated_scores[:,0], estimated_scores[:,1], 
                 np.linalg.norm(estimated_scores, axis=1), color='red', alpha=0.5)
    ax[0,1].set_xlabel('x')
    ax[0,1].set_ylabel('y')
    ax[0,1].set_title('Estimated score for dataset')


    x_grid = np.linspace(np.min(data[:,0]), np.max(data[:,0]), 50)
    y_grid = np.linspace(np.min(data[:,1]), np.max(data[:,1]), 50)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    scores_grid = scores.score_banana(grid, mu, sigma)


    if type=='anneal_denoising_score_matching':
        labels = torch.randint(0, len(sigma_list), (grid.shape[0],))
        estimated_scores_grid = trained_model(torch.tensor(grid, requires_grad=True).float().to(device),labels).detach().cpu().numpy()
    else:
        estimated_scores_grid = trained_model(torch.tensor(grid, requires_grad=True).float().to(device)).detach().cpu().numpy()

    ax[1,0].scatter(data[:,0], data[:,1], s = 1, alpha=0.5)
    ax[1,0].quiver(grid[:,0], grid[:,1], scores_grid[:,0], scores_grid[:,1], 
                 np.linalg.norm(scores_grid, axis=1), color='red', alpha=0.5, cmap='autumn_r')
    ax[1,0].set_xlabel('x')
    ax[1,0].set_ylabel('y')
    ax[1,0].set_title('True score over grid')
    
    ax[1,1].scatter(data[:,0], data[:,1], s = 1, alpha=0.5)
    ax[1,1].quiver(grid[:,0], grid[:,1], estimated_scores_grid[:,0], estimated_scores_grid[:,1], 
                 np.linalg.norm(estimated_scores_grid, axis=1), color='red', alpha=0.5, cmap='autumn_r')
    ax[1,1].set_xlabel('x')
    ax[1,1].set_ylabel('y')
    ax[1,1].set_title('Estimated score over grid')

    fig.suptitle('Banana-shaped score')
    fig.tight_layout()
    plt.show()

    if difference:
        fig2, ax2 = plt.subplots(1, 1, figsize=(15, 6))
        difference = np.linalg.norm(scores_grid - estimated_scores_grid, axis=1)
        levels = np.linspace(0,40, 100)
        ax2.contourf(x_grid, y_grid, difference.reshape(xx.shape), levels = levels, cmap='seismic')
        fig2.colorbar(ax2.contourf(x_grid, y_grid, difference.reshape(xx.shape), levels = levels, cmap='seismic'))
        ax2.scatter(data[:,0], data[:,1], s = 1, c='white', alpha=0.5)
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_title('Difference between true and estimated score')
        fig2.tight_layout()
        plt.show()