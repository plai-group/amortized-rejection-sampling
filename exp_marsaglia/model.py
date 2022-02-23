import pyprob
from pyprob import Model
from pyprob.distributions import Normal, Uniform
import torch
import numpy as np


class GUMMarsaglia(Model):
    def __init__(self, prior_mean, prior_std, likelihood_std, num_obs):
        super().__init__(name='Gaussian with unknown mean (Marsaglia)')
        self.prior_mean = prior_mean
        self.prior_std = prior_std
        self.likelihood_std = likelihood_std
        self.num_obs = num_obs
        
    def marsaglia(self, mean, stddev):
        uniform = Uniform(-1, 1)
        s = 1
        while float(s) >= 1:
            pyprob.rs_start()
            x = pyprob.sample(uniform, name='base_x')
            y = pyprob.sample(uniform, name='base_y')
            s = x*x + y*y
        pyprob.rs_end()

        return mean + stddev * (x * torch.sqrt(-2 * torch.log(s) / s))

    def forward(self):

        mu = self.marsaglia(self.prior_mean, self.prior_std)
        likelihood = Normal(mu, self.likelihood_std)

        for i in range(self.num_obs):
            pyprob.observe(likelihood,
            name=f'obs_{i}')

        return mu

    def posterior_analytic(self, observe):
        mu, sigma = GUMPosteriorParameters(self.prior_mean, self.prior_std,
                                           self.likelihood_std, list(observe.values()))
        return Normal(mu, sigma)

    def evidence(self, observe):
        D = np.array(list(observe.values()))
        n = len(D)
        return GUMEvidence(D=D, mu_0=self.prior_mean, sigma_0=self.prior_std,
                           sigma=self.likelihood_std)


class GUMMarsagliaCollapsed(GUMMarsaglia):
    def __init__(self, prior_mean, prior_std, likelihood_std, num_obs):
        super().__init__(prior_mean, prior_std, likelihood_std, num_obs)
    
    def marsaglia(self, mean, stddev):
        x = pyprob.sample(Normal(0, 1), name='base_x')
        y = pyprob.sample(Normal(0, 1), name='base_y')
        return x


def GUMPosteriorParameters(mu_0, sigma_0, sigma, observed_list):
    '''
    Computes posterior parameters for a Gaussian Unknown Mean defined as follows:
        mu ~ N(mu_0, sigma_0)
        x ~ N(mu, sigma)
    Posterior for such model would be N(mu_n, sigma_n).
    This function returns mu_n and sigma_n
    Conjugate Bayesian analysis of the Gaussian distribution, Murphy. (https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
    '''
    n = len(observed_list)
    var_n = 1/(n/sigma**2 + 1/sigma_0**2)
    mu_n = var_n * (mu_0/sigma_0**2 + np.sum(observed_list)/sigma**2)
    return mu_n, np.sqrt(var_n)


def GUMEvidence(D, mu_0, sigma_0, sigma):
    '''
    D should be the observations represented as a numpy array.
    Conjugate Bayesian analysis of the Gaussian distribution, Murphy. (https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf)
    '''
    n = len(D)
    sigma_0_2 = sigma_0**2
    sigma_2 = sigma**2
    mu_0_2 = mu_0**2
    t1 = sigma / ((np.sqrt(2*np.pi)*sigma)**n * np.sqrt(n*sigma_0_2 + sigma_2))
    t2 = -((D@D)/(2*sigma_2) + mu_0_2/(2*sigma_0_2))
    t3 = (sigma_0_2*(n**2)*(np.mean(D)**2)/sigma_2 + sigma_2*mu_0_2/sigma_0_2 + 2*np.sum(D)*mu_0) / (2*(n*sigma_0_2 + sigma_2))
    return t1 * np.exp(t2 + t3)