import numpy as np
class AnalyticalPDFs:
    def __init__(self, distribution_type, params=None):
        self.distribution_type = distribution_type
        self.params = params

    def get_values(self, l_values):
        if self.distribution_type == "Uniform":
            return self.uniform(l_values, self.params["Lmin"], self.params["Lmax"])
        elif self.distribution_type == "Log-Normal":
            return self.logNormal(l_values, self.params["mu"], self.params["sigma"])
        elif self.distribution_type == "Negative power-law":
            return self.negativePowerLaw(l_values, self.params["alpha"])
        elif self.distribution_type == "Negative exponential":
            return self.negativeExponential(l_values, self.params["lambda"])
        else:
            raise ValueError(f"Unknown distribution type: {self.distribution_type}")

    def uniform(self, l_values, Lmin, Lmax):
        return [1 / (Lmax - Lmin) if Lmin <= l <= Lmax else 0 for l in l_values]

    def logNormal(self, l_values, mu, sigma):
        return [1 / (l * sigma * np.sqrt(2 * np.pi)) * np.exp(- (np.log(l) - mu)**2 / (2 * sigma**2)) for l in l_values]

    def negativePowerLaw(self, l_values, alpha):
        return [alpha * l**(-alpha) for l in l_values]

    def negativeExponential(self, l_values, lambdaf):
        return [lambdaf * np.exp(-lambdaf * l) for l in l_values]
