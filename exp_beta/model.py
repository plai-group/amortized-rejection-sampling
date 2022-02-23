import pyprob
from pyprob import Model
from pyprob.distributions import Uniform, Beta, Bernoulli, Mixture
import numpy as np


class RejectionBetaBernoulli(Model):
    def __init__(self, beta, num_obs):
        super().__init__()
        self.beta = beta
        self.base = Uniform(0, 1)
        self.num_obs = num_obs
        self.perfect_proposal = False

    def sample_prior(self):
        while True:
            pyprob.rs_start()
            u_1 = pyprob.sample(self.base, name="base")
            upper_bound = (4 * u_1 * (1 - u_1))**(self.beta - 1)

            proposal = Mixture([Uniform(0, upper_bound), Uniform(0, 1)], probs=[0.99, 0.01]) if self.perfect_proposal == True else None
            u_2 = pyprob.sample(Uniform(0, 1), name="aux", proposal=proposal)
            
            if u_2 <= upper_bound:
                pyprob.rs_end()
                break
        return u_1
    
    def forward(self):
        x = self.sample_prior()
        likelihood = Bernoulli(x)
        for i in range(self.num_obs):
            pyprob.observe(likelihood, name=f"obs_{i}")
        return x

    def posterior_gt(self):
        alpha = self.beta
        beta = self.beta
        return Beta(alpha + self.num_obs, beta)

    def log_evidence(self):
        log_evidence = (
            Beta(self.beta, self.beta).log_prob(0.5) +\
            self.num_obs * np.log(0.5) - \
            self.posterior_gt().log_prob(0.5)
            ).item()
        return log_evidence


class BetaBernoulli(Model):
    def __init__(self, beta, num_obs):
        super().__init__()
        self.prior = Beta(beta, beta)
        self.num_obs = num_obs
    
    def forward(self):
        x = pyprob.sample(self.prior, name="prior")
        likelihood = Bernoulli(x)
        for i in range(self.num_obs):
            pyprob.observe(likelihood, name=f"obs_{i}")
        return x