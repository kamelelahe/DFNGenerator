import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import vonmises

loc = 0.5 * np.pi  # circular mean
kappa = 1  # concentration
samples = vonmises(loc=loc, kappa=kappa).rvs()
print(samples)