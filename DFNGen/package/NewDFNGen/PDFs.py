import random
import math
import numpy as np
from numba import jit
from scipy.stats import vonmises

class PDFs:
    def __init__(self, distribution_type, params=None):
        self.distribution_type = distribution_type
        self.params = params

    def get_value(self):
        if self.distribution_type == "Fixed":
            return self.params["fixed_value"]
        elif self.distribution_type == "Uniform":
            return uniform(self.params["Lmin"], self.params["Lmax"])
        elif self.distribution_type == "Log-Normal":
            return logNormal(self.params["mu"], self.params["sigma"], self.params["Lmax"])
        elif self.distribution_type == "Negative power-law":
            return negativePowerLaw(self.params["alpha"],self.params["Lmin"], self.params["Lmax"])
        elif self.distribution_type == "Negative exponential":
            return negativeExponential(self.params["lambda"], self.params["Lmax"])
        elif  self.distribution_type == "Von-Mises":
            return  np.degrees(vonmisesImp(self.params["loc"], self.params["kappa"]))
        elif self.distribution_type == "Constant":
            return self.params["value"]
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

    def compute_mode(self):
        if self.distribution_type == "Fixed":
            return self.params["fixed_value"]
        elif self.distribution_type == "Uniform":
            return 0 #it doesnt have a mode so we consider the mode as zero for clculating the distance
        elif self.distribution_type == "Log-Normal":
            mode = np.exp(self.params["mu"] - self.params["sigma"] ** 2)
            return mode
        elif self.distribution_type == "Negative power-law":
            # The mode for negative power-law is typically the minimum value
            return self.params["Lmin"]
        elif self.distribution_type == "Negative exponential":
            # The mode for a negative exponential distribution is always the location parameter, which is zero for the standard form.
            return 0
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

@jit(nopython=True)
def uniform(min,max):
    return random.uniform(min,max)
@jit(nopython=True)
def logNormal( mu,sigma, max=None):
    #for plotting, remove the -mu
    val=random.lognormvariate(mu, sigma)
    while val>max:
        val= random.lognormvariate(mu, sigma)
    return val
#@jit(nopython=True)
def negativePowerLaw(alpha, min, max=None):
    if alpha <= 1:
        raise ValueError("Alpha must be greater than 1 for the distribution to be normalizable.")
    if max is None:
        max = np.inf
    cdf_min = 1 - min ** (-alpha + 1)
    cdf_max = 1 - max ** (-alpha + 1)
    u = np.random.rand() * (cdf_max - cdf_min) + cdf_min
    val = min / (1 - u) ** (1 / (alpha - 1))
    max_attempts = 1000
    attempts = 0
    while val > max and attempts < max_attempts:
        u = np.random.rand() * (cdf_max - cdf_min) + cdf_min
        val = min / (1 - u) ** (1 / (alpha - 1))
        attempts += 1
    if attempts >= max_attempts:
        raise ValueError("Unable to generate a value within the specified range after many attempts.")
    return val

def vonmisesImp(loc,kappa):
    return vonmises(loc=loc, kappa=kappa).rvs()

@jit(nopython=True)
def negativeExponential(lambdaf,max=None):
    val=-lambdaf * math.log(1.0 - random.random())
    while val>max:
        val= -lambdaf * math.log(1.0 - random.random())
    return val

