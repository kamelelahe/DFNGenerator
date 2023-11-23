from .PDFs import PDFs

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# Initialize parameters for each distribution
params_fixed = {"fixed_value": 5}
params_uniform = {"Lmin": 1, "Lmax": 10}
params_lognormal = {"mu": 0, "sigma": 1}
params_powerlaw = {"alpha": 2, "Lmin": 1}
params_exponential = {"lambda": 1}

# Create PDFs objects for each distribution
pdf_fixed = PDFs("Fixed", params_fixed)
pdf_uniform = PDFs("Uniform", params_uniform)
pdf_lognormal = PDFs("Log-Normal", params_lognormal)
pdf_powerlaw = PDFs("Negative power-law", params_powerlaw)
pdf_exponential = PDFs("Negative exponential", params_exponential)

# Generate samples
num_samples = 10000
samples_fixed = [pdf_fixed.get_value() for _ in range(num_samples)]
samples_uniform = [pdf_uniform.get_value() for _ in range(num_samples)]
samples_lognormal = [pdf_lognormal.get_value() for _ in range(num_samples)]
samples_powerlaw = [pdf_powerlaw.get_value() for _ in range(num_samples)]
samples_exponential = [pdf_exponential.get_value() for _ in range(num_samples)]


# Set style
sns.set_style("whitegrid")
sns.set_context("paper")  # or "paper" for smaller font sizes
# Plot histograms
fig, axs = plt.subplots(3, 2, figsize=(12, 14))
# Define a function to plot the histograms

def plot_hist(ax, data, title, color):
    sns.histplot(data, bins=50, kde=True, ax=ax, color=color, stat="density")
    ax.set_title(title, y=1)  # Adjust the y value as needed
    ax.set_ylabel('Density')
    ax.set_xlabel('Value')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

plot_hist(axs[0, 0], samples_fixed, 'Fixed Distribution', 'blue')
plot_hist(axs[0, 1], samples_uniform, 'Uniform Distribution', 'green')
plot_hist(axs[1, 0], samples_lognormal, 'Log-Normal Distribution', 'red')
plot_hist(axs[1, 1], samples_powerlaw, 'Negative Power-Law Distribution', 'purple')
plot_hist(axs[2, 0], samples_exponential, 'Negative Exponential Distribution', 'orange')

# Remove the empty subplot
axs[2, 1].axis('off')
# Adjust the spacing between subplots
plt.subplots_adjust(hspace=0.23)  # Adjust this value as needed

# If needed, further adjust the spacing between subplots
plt.subplots_adjust(hspace=0.23, wspace=0.2, top=0.95, bottom=0.05, left=0.07, right=0.95)

plt.savefig('myBeautifulPlot.png', format='png', dpi=300)
