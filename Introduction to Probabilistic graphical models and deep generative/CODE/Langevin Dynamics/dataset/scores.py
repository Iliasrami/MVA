import numpy as np
import torch

from . import densities

def gaussian_mixture_score(x, mu, sigmas, alphas):
    """
    Computes the score of a Gaussian mixture model.
    ----------
    Parameters:
    x: Points to evaluate the density at.
    mu: Means of the Gaussian components.
    var: Covariances of the Gaussian components.
    alphas: Mixing proportions of the Gaussian components.
    """
    if x.ndim == 1:    
        score = 0
        for i in range(len(mu)):
            term_exp = -0.5 * np.dot(np.dot((x - mu[i]), np.linalg.inv(sigmas[i])), (x - mu[i]).T)
            score += - alphas[i] / np.sqrt(np.linalg.det(sigmas[i])) * np.dot(np.linalg.inv(sigmas[i]),
                                                                               (x - mu[i])) * np.exp(term_exp)
        score /= densities.gmm_density(x, mu, sigmas, alphas) * 2 * np.pi
        return score
    else:
        scores = np.zeros(x.shape)
        for i in range(len(x)):
            for j in range(len(mu)):
                term_exp = -0.5 * np.dot(np.dot((x[i] - mu[j]), np.linalg.inv(sigmas[j])), (x[i] - mu[j]).T)
                scores[i] += - alphas[j] / np.sqrt(np.linalg.det(sigmas[j])) * np.dot(np.linalg.inv(sigmas[j]), 
                                                                                      (x[i] - mu[j])) * np.exp(term_exp)
            scores[i] /= densities.gmm_density(x[i], mu, sigmas, alphas) * 2 * np.pi
        return scores
    

    

def score_banana(x, mu, sigma, b=0.5):
    """
    Computes the score of the banana-shaped distribution.
    Parameters
    ----------
    x : The point at which to evaluate the density function.
    mu : The mean of the distribution.
    sigma : The covariance matrix of the distribution.
    b : The parameter of the distribution.
    """
    # x_transformed = np.copy(x)
    if x.ndim ==1:
        score = np.zeros(x.shape)
        score[0] = - (x[0] - mu[0]) / sigma[0,0] - 2 * b * x[0] * (x[1] + b * (x[0]**2 - sigma[0,0]) - mu[1]) / sigma[1,1]
        score[1] = - (x[1] + b * (x[0]**2 - sigma[0,0]) - mu[1]) / sigma[1,1]
        return score
    score = np.zeros(x.shape)
    for i in range(len(x)):
        score[i,0] = - (x[i,0] - mu[0]) / sigma[0,0] - 2 * b * x[i,0] * (x[i,1] + b * (x[i,0]**2 - sigma[0,0]) - mu[1]) / sigma[1,1]
        score[i,1] = - (x[i,1] + b * (x[i,0]**2 - sigma[0,0]) - mu[1]) / sigma[1,1]
    return score
    
    