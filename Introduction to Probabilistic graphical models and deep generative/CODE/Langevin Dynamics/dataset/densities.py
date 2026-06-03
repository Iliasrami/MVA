import numpy as np

def multivariate_gaussian_density(x, mu, sigma):
    """
    Computes the multivariate gaussian density function at x with mean mu and covariance sigma.
    ----------
    Parameters:
    x : numpy array of shape (n, 2)
        The point at which to evaluate the density function.
    mu : numpy array of shape (2,)
        The mean of the gaussian distribution.
    sigma : numpy array of shape (2, 2)
        The covariance matrix of the gaussian distribution.
    """
    if x.ndim == 1:
        return (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))) * np.exp(-0.5 * np.dot(np.dot((x - mu), 
                                                                                             np.linalg.inv(sigma)), (x - mu).T))
    else:
        density = np.zeros(x.shape[0])
        for i in range(len(x)):
            density[i] = (1 / (2 * np.pi * np.sqrt(np.linalg.det(sigma)))) * np.exp(-0.5 * np.dot(np.dot((x[i] - mu), 
                                                                                                         np.linalg.inv(sigma)), 
                                                                                                         (x[i] - mu).T))
        return density


def gmm_density(x, mu, sigmas, alphas):
    """
    Computes the unnormalized density of a Gaussian mixture model.
    ----------
    Parameters:
    x: Points to evaluate the density at. 
    mu: Means of the Gaussian components.
    var: Covariances of the Gaussian components.
    alphas: Mixing proportions of the Gaussian components.
    """
    density = 0
    for i in range(len(mu)):
        density += alphas[i] * multivariate_gaussian_density(x, mu[i], sigmas[i])
    return density

def gmm_log_density(x, mu, sigmas, alphas):
    """
    Computes the unnormalized log density of a Gaussian mixture model.
    ----------
    Parameters:
    x: Points to evaluate the density at. 
    mu: Means of the Gaussian components.
    var: Covariances of the Gaussian components.
    alphas: Mixing proportions of the Gaussian components.
    """
    return np.log(gmm_density(x, mu, sigmas, alphas))

def banana_density(x, mu, sigma, b=0.5):
    """
    Computes the unnormalized log density of the banana-shaped distribution.
    Parameters
    ----------
    x : The point at which to evaluate the density function.
    mu : The mean of the distribution.
    sigma : The covariance matrix of the distribution.
    b : The parameter of the distribution.
    """
    if x.ndim == 1:
        x_transformed = np.copy(x)
        x_transformed[1] = x[1] + b * (x[0]**2 - sigma[0,0])
        return multivariate_gaussian_density(x_transformed, mu, sigma)
    x_transformed = np.copy(x)
    x_transformed[:,1] = x[:,1] + b * (x[:,0]**2 - sigma[0,0])
    return multivariate_gaussian_density(x_transformed, mu, sigma)


def log_density_banana(x, mu, sigma, b=0.5):
    """
    Computes the unnormalized log density of the banana-shaped distribution.
    Parameters
    ----------
    x : The point at which to evaluate the density function.
    mu : The mean of the distribution.
    sigma : The covariance matrix of the distribution.
    b : The parameter of the distribution.
    """
    if x.ndim == 1:
        x_transformed = np.copy(x)
        x_transformed[1] = x[1] + b * (x[0]**2 - sigma[0,0])
        return np.log(multivariate_gaussian_density(x_transformed, mu, sigma))
    x_transformed = np.copy(x)
    x_transformed[:,1] = x[:,1] + b * (x[:,0]**2 - sigma[0,0])
    return np.log(multivariate_gaussian_density(x_transformed, mu, sigma))


