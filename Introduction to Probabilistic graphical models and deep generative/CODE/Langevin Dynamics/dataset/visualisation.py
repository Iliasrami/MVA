import numpy as np
import matplotlib.pyplot as plt

from . import scores


def plot_GMM(samples, clusters, mu, sigma, contours=True):
    """
    Plots the contours of the gaussians with parameters mu and sigma.
    Parameters
    ----------
    samples : The samples.
    clusters : The clusters of the samples.
    mu : The means of the gaussians.
    sigma : The covariance matrices of the gaussians.
    contours : If True, plots the contours of the gaussians.
    """
    assert len(mu) == len(sigma), 'mu and sigma must have the same length'
    assert samples.shape[1] == 2, 'samples must be 2-dimensional'

    if not contours:
        plt.scatter(samples[:,0], samples[:,1], c=clusters, alpha=0.1)
        for i in range(len(mu)):
            # plt.scatter(mu[:,0], mu[:,1], c='red', s=100, alpha=1)
            plt.scatter(mu[i][0], mu[i][1], c='red', s=100, alpha=1)

    if contours:
        for i in range(len(mu)):
            # Samples and means of each cluster
            x, y = samples[clusters == i].T
            plt.plot(x, y, 'o', alpha=0.5)
            plt.plot(mu[i][0], mu[i][1], 'x', color='red')

            # Contour plot of each cluster
            x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), np.linspace(min(y), max(y), 100))
            z_grid = np.empty(x_grid.shape)
            for j in range(x_grid.shape[0]):
                for k in range(x_grid.shape[1]):
                    x = np.array([x_grid[j, k], y_grid[j, k]])
                    z_grid[j, k] = np.exp(-0.5 * (x - mu[i]).T @ np.linalg.inv(sigma[i]) 
                                          @ (x - mu[i])) / (2 * np.pi * np.sqrt(np.linalg.det(sigma[i])))
            plt.contour(x_grid, y_grid, z_grid, levels=5, colors='black', alpha=1, linewidths=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    # Set the axis limits depending on the samples with some margin scaled to the size of the samples
    plt.xlim(min(samples[:,0]) - 0.1 * (max(samples[:,0]) - min(samples[:,0])), 
             max(samples[:,0]) + 0.1 * (max(samples[:,0]) - min(samples[:,0])))
    plt.ylim(min(samples[:,1]) - 0.1 * (max(samples[:,1]) - min(samples[:,1])), 
             max(samples[:,1]) + 0.1 * (max(samples[:,1]) - min(samples[:,1])))

    plt.title('Gaussian mixture')
    plt.tight_layout()
    plt.show()



def gmm_density_heatmap(density, *args):
    """
    Plots a heatmap of the unnormalized density function.
    ----------
    Parameters
    density : The density function.
    *args : The parameters of the density function.
    """
    mus, sigmas, alphas = args
    x_grid, y_grid = np.meshgrid(np.linspace(-5, 15, 100), np.linspace(-5, 15, 100))
    z_grid = np.empty(x_grid.shape)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x = np.array([x_grid[i, j], y_grid[i, j]])
            z_grid[i, j] = density(x, mus, sigmas, alphas)
    plt.contourf(x_grid, y_grid, z_grid, levels=100)
    plt.colorbar()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Unnormalized Gaussian mixture density')
    plt.show()


def gmm_score_plot(data, clusters, mus, sigmas, alphas):
    """
    Plots the score of the Gaussian mixture model.
    Parameters
    ----------
    data : The samples.
    clusters : The clusters of the samples.
    mus : The means of the gaussians.
    sigmas : The covariance matrices of the gaussians.
    alphas : The mixing proportions of the gaussians.
    """
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # Plot the score of the Gaussian mixture model
    scores_vec = scores.gaussian_mixture_score(data, mus, sigmas, alphas)
    ax[0].scatter(data[:,0], data[:,1], s = 1, c=clusters, alpha=0.5)
    ax[0].quiver(data[:,0], data[:,1], scores_vec[:,0], scores_vec[:,1], 
                 np.linalg.norm(scores_vec, axis=1), color='red', alpha=0.5)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Gaussian mixture score for dataset')

    # Plot the score over a grid
    x_grid = np.linspace(-5, 15, 50)
    y_grid = np.linspace(-5, 15, 50)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    scores_grid = scores.gaussian_mixture_score(grid, mus, sigmas, alphas)
    ax[1].scatter(data[:,0], data[:,1], s = 1, c=clusters, alpha=0.5)
    ax[1].quiver(grid[:,0], grid[:,1], scores_grid[:,0], scores_grid[:,1], 
                 np.linalg.norm(scores_grid, axis=1), color='red', alpha=0.5, cmap='autumn_r')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Gaussian mixture score over grid')

    fig.suptitle('Gaussian mixture score')
    fig.tight_layout()
    plt.show()


def plot_banana(banana_shaped_data):    
    plt.scatter(banana_shaped_data[:,0], banana_shaped_data[:,1], s = 1, alpha=0.5)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Banana-shaped distribution')
    plt.show()


def banana_density_heatmap(density, *args):
    """
    Plots a heatmap of the unnormalized density function.
    ----------
    Parameters
    density : The density function.
    *args : The parameters of the density function.
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    mu, sigma, b = args
    x_grid, y_grid = np.meshgrid(np.linspace(mu-4, mu+4, 100), np.linspace(mu-4, mu+4, 100))
    z_grid = np.empty(x_grid.shape)
    for i in range(x_grid.shape[0]):
        for j in range(x_grid.shape[1]):
            x = np.array([x_grid[i, j], y_grid[i, j]])
            z_grid[i, j] = density(x, mu, sigma, b)
    ax.contourf(x_grid, y_grid, z_grid, levels=100)
    fig.colorbar(ax.contourf(x_grid, y_grid, z_grid, levels=100), ax=ax)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Unnormalized banana-shaped density')
    plt.show()


def banana_score_plot(banana_shaped_data):
    
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    # Plot the score of the banana-shaped distribution
    scores_vec = scores.score_banana(banana_shaped_data, np.array([0, 0]), np.eye(2))
    ax[0].scatter(banana_shaped_data[:,0], banana_shaped_data[:,1], s = 1, alpha=0.5)
    ax[0].quiver(banana_shaped_data[:,0], banana_shaped_data[:,1], scores_vec[:,0], scores_vec[:,1], 
                 np.linalg.norm(scores_vec, axis=1), color='red', alpha=0.5)
    ax[0].set_xlabel('x')
    ax[0].set_ylabel('y')
    ax[0].set_title('Banana-shaped score for dataset')

    # Plot the score over a grid
    x_grid = np.linspace(np.min(banana_shaped_data[:,0]), np.max(banana_shaped_data[:,0]), 50)
    y_grid = np.linspace(np.min(banana_shaped_data[:,1]), np.max(banana_shaped_data[:,1]), 50)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.hstack([xx.reshape(-1, 1), yy.reshape(-1, 1)])
    scores_grid = scores.score_banana(grid, np.array([0, 0]), np.eye(2))
    ax[1].scatter(banana_shaped_data[:,0], banana_shaped_data[:,1], s = 1, alpha=0.5)
    ax[1].quiver(grid[:,0], grid[:,1], scores_grid[:,0], scores_grid[:,1], 
                 np.linalg.norm(scores_grid, axis=1), color='red', alpha=0.5, cmap='autumn_r')
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('y')
    ax[1].set_title('Banana-shaped score over grid')

    fig.suptitle('Banana-shaped score')
    fig.tight_layout()
    plt.show()


def plot_start(self, samples, clusters, contours=True):
    mu = self.Y_star
    sigma = self.covariances
    assert len(mu) == len(sigma), 'mu and sigma must have the same length'
    assert samples.shape[1] == 2, 'samples must be 2-dimensional'

    for i in range(len(mu)):
        cluster_samples = samples[clusters == i]

        # Skip plotting for empty clusters
        
        if len(cluster_samples) == 0:
            continue

        x, y = cluster_samples.T
        plt.scatter(x, y, alpha=0.5)

        if contours:
            plt.plot(mu[i][0], mu[i][1], 'x', color='red')
            x_grid, y_grid = np.meshgrid(np.linspace(min(x), max(x), 100), 
                                         np.linspace(min(y), max(y), 100))
            z_grid = np.empty(x_grid.shape)
            for j in range(x_grid.shape[0]):
                for k in range(x_grid.shape[1]):
                    x_point = np.array([x_grid[j, k], y_grid[j, k]])
                    z_grid[j, k] = np.exp(-0.5 * (x_point - mu[i]).T @ np.linalg.inv(sigma[i]) 
                                          @ (x_point - mu[i])) / (2 * np.pi * np.sqrt(np.linalg.det(sigma[i])))
            plt.contour(x_grid, y_grid, z_grid, levels=5, colors='black', alpha=1, linewidths=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian star discrete')
    plt.tight_layout()
    plt.show()


def plot_start_sans_cluster(self, samples, contours=True):
    mu = self.Y_star
    sigma = self.covariances
    assert len(mu) == len(sigma), 'mu and sigma must have the same length'
    assert samples.shape[1] == 2, 'samples must be 2-dimensional'

    # Plot all samples with the same color
    plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)

    if contours:
        for i in range(len(mu)):
            plt.plot(mu[i][0], mu[i][1], 'x', color='red')
            x_grid, y_grid = np.meshgrid(np.linspace(min(samples[:, 0]), max(samples[:, 0]), 100), 
                                         np.linspace(min(samples[:, 1]), max(samples[:, 1]), 100))
            z_grid = np.empty(x_grid.shape)
            for j in range(x_grid.shape[0]):
                for k in range(x_grid.shape[1]):
                    x_point = np.array([x_grid[j, k], y_grid[j, k]])
                    z_grid[j, k] = np.exp(-0.5 * (x_point - mu[i]).T @ np.linalg.inv(sigma[i]) 
                                          @ (x_point - mu[i])) / (2 * np.pi * np.sqrt(np.linalg.det(sigma[i])))
            plt.contour(x_grid, y_grid, z_grid, levels=5, colors='black', alpha=1, linewidths=0.5)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Gaussian mixture')
    plt.tight_layout()
    plt.show()




