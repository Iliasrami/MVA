import torch
import torch.autograd as autograd
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm


class ScoreMatching():
    def __init__(self, optimizer, loss_type, device,sigma=0.01,anneal_power=2):
        self.optimizer = optimizer
        self.device = device
        self.loss_type = loss_type
        self.loss_functions = {
            'implicit_score_matching': self.implicit_score_matching,
            'denoising_score_matching': self.denoising_score_matching,
            'sliced_score_matching': self.sliced_score_matching,
            'anneal_denoising_score_matching': self.anneal_dsm_score
        }
        self.sigma=sigma # sigmas to pertubate data in denoising score matching and also anneal dsm
        self.anneal_power=anneal_power #the factor to multiply lambda(sigma)=sigma^2 so by default anneal power=2

        if loss_type not in self.loss_functions:
            raise ValueError(f"Invalid loss type '{loss_type}'. Supported types are: {', '.join(self.loss_functions.keys())}")
        

    def get_jacobian(self, model, x):
        """
        Computes the Jacobian of the model w.r.t the data points x.
        ----------
        Parameters:
        model: score network (R^N -> R^N)
        x: data points (B, N)
        Returns:
        jacobian: the Jacobian of func w.r.t x (B, N, N)
        """
        B, N = x.shape
        y = model(x)
        jacobian_matrices = list()
        for i in range(N):
            v = torch.zeros_like(y)
            v[:, i] = 1.
            dy_i_dx = autograd.grad(y, x, grad_outputs=v, retain_graph=True, 
                                    create_graph=True, allow_unused=True)[0]  # shape [B, N]
            jacobian_matrices.append(dy_i_dx)
        jacobian = torch.stack(jacobian_matrices, dim=2).requires_grad_()
        return jacobian


    def implicit_score_matching(self, data, score_network):
        """
        Computes the implicit score matching loss for a given data batch.
        ----------
        Parameters:
        data: a batch of data points (B, N)
        score_network: a score network (R^N -> R^N)
        Returns:
        loss: the implicit score matching loss (scalar)
        """
        # Turn numpy array into tensor
        data = torch.tensor(data, requires_grad=True).float().to(self.device)
        # Compute the score
        score = score_network(data)
        # Compute the Jacobian and its trace
        jacobian_matrix = self.get_jacobian(score_network, data)
        jacobian_trace = torch.diagonal(jacobian_matrix, dim1=-1, dim2=-2).sum(dim=-1)
        # Compute the norm of the score estimation
        score_network_norm = 0.5 * torch.norm(score, dim=-1)**2
        # Compute the loss as empirical mean
        loss = torch.mean(jacobian_trace + score_network_norm, dim = -1)
        return loss

    def denoising_score_matching(self, data, model):
        """
         Computes the sliced score matching loss for a given data
          taking data and model as entries
          return denoising score matching loss
        """
        data=torch.Tensor(data).float()

        # data pertubation 
        perturbed_samples = data + torch.randn_like(data) * self.sigma
        # computation of the target and scores
        target = - 1 / (self.sigma** 2) * (perturbed_samples - data)
        scores = model(perturbed_samples)
        # reshape Tensor
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)

        # loss according to the formula
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1).mean(dim=0)
        return loss

    def sliced_score_matching(self, data, score_network):
        """
        Computes the sliced score matching loss for a given data batch of size B.
        ----------
        Parameters:
        data: a batch of data points (B, N)
        score_network: a score network (R^N -> R^N)
        Returns:
        loss: the sliced score matching loss (scalar)
        """
        # Turn numpy array into tensor
        data = torch.tensor(data, requires_grad=True).float().to(self.device)
        # Generate random vectors
        random_vectors = torch.randn_like(data)
        random_vectors /= torch.norm(random_vectors, dim=-1, keepdim=True)
        # Compute random projections
        score, projections = autograd.functional.jvp(score_network, data, random_vectors, create_graph=True)
        projections = projections * random_vectors
        # Compute the norm of the score estimation
        score_network_norm = 0.5 * torch.norm(score, dim=-1)**2
        # Compute the loss as empirical mean
        loss = torch.mean(torch.sum(projections, dim=-1) + score_network_norm, dim = -1)
        return loss
    
    def anneal_dsm_score(self,data,score_network,labels):

        """
                compute the anneal dsm score for a given batch of data with a given tensor of noise values

                return loss
        """
        # convert data to a tensor
        data=torch.Tensor(data).float()

        # extract the sigmas corresponding to the indices in labels and reshape it as (Batch, 1)
        used_sigmas = self.sigma[labels].view(data.shape[0], *([1] * len(data.shape[1:])))

        # pertub the origal data by adding the tensor of noises (used_sigmas)
        perturbed_datas = data + torch.randn_like(data) * used_sigmas
        # compute the target
        target = - 1 / (used_sigmas ** 2) * (perturbed_datas - data)
        # compute the score with our model, we give the labels since we need to take into account the noises used_sigmas
        #since labels is the vector of indices corresponding to the selected sigmas (used_sigmas)
        scores = score_network(perturbed_datas, labels)
        # reshape it 
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        # compute the loss accordind to the formula
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** self.anneal_power
        # return the loss 
        return loss.mean(dim=0)


    def compute_loss(self, data, model,labels=None):
        """
        Computes the loss for a given data batch based on the loss type defined in the constructor.
        ----------
        Parameters:
        data: a batch of data points (B, N)
        model: a score network (R^N -> R^N)
        Returns:
        loss: the loss (scalar)
        """
         # add the condition to use it when loss is anneal dsm
        if  self.loss_type=='anneal_denoising_score_matching':
            return self.loss_functions[self.loss_type](data, model,labels)
        else:
            return self.loss_functions[self.loss_type](data, model)
    
    def train_step(self, data, model):
        """
        Performs a single training step.
        ----------
        Parameters:
        data: a batch of data points (B, N)
        model: a score network (R^N -> R^N)
        Returns:
        loss: the loss (scalar)
        """
        
        self.optimizer.zero_grad()
        
         # add the condition to use it when loss is anneal dsm
        if  self.loss_type=='anneal_denoising_score_matching':

            data=torch.Tensor(data).float()
            # labels is the vector of indices corresponding to the sigmas to be selected for data perturbation
            labels = torch.randint(0, len(self.sigma), (data.shape[0],))
            loss = self.compute_loss(data, model,labels)
        else:
            loss = self.compute_loss(data, model)
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, data, model, epochs = 1000, verbose = True, plot = True):
        """
        Trains the model for a given number of epochs.
        ----------
        Parameters:
        data: a batch of data points (B, N)
        model: a score network (R^N -> R^N)
        epochs: number of epochs to train
        verbose: whether to print the loss during training
        plot: whether to plot the loss after training
        Returns:
        losses_hist: a list of the losses during training
        """
        losses_hist = []
        for i in tqdm(range(epochs)):
            loss = self.train_step(data, model)
            losses_hist.append(loss)
            if verbose and i % 100 == 0:
                print("Epoch: {}/{}, Loss: {}".format(i, epochs, loss))
        print("Training finished!")
        if plot:
            plt.plot(losses_hist)
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.show()



