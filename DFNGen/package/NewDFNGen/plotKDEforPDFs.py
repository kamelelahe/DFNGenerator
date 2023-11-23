import random
import math
import powerlaw
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from .PDFs import PDFs
from .analyticalsPDFs import AnalyticalPDFs

# Set style
sns.set_style("whitegrid")
sns.set_context("paper")

# Modify the plot_frequency_vs_length function
def plot_frequency_vs_length(ax, data, title, color, label=None, analytical=False):
    if analytical:
        ax.plot(data[0], data[1], color=color, lw=2, label=label)
    else:
        sns.kdeplot(data, ax=ax, color=color, lw=2, label=label)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2)
    ax.set_title(title, y=1.05)
    ax.set_ylabel('Normalized Fracture Frequency')
    ax.set_xlabel('Fracture Length')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if label:
        ax.legend()

# Plot histograms
fig, axs = plt.subplots(4, 2, figsize=(12, 24))

# Number of samples
num_samples = 10000

# L values for analytical plots
l_values = np.linspace(0.1, 10, 400)
colors = cm.Set1(np.linspace(0, 1, 1))
# Uniform
params_uniform = {"Lmin": 1, "Lmax": 10}
pdf_uniform = PDFs("Uniform", params_uniform)
samples_uniform = [pdf_uniform.get_value() for _ in range(num_samples)]
plot_frequency_vs_length(axs[0, 0], samples_uniform, 'Uniform Distribution KDE', colors[0])

pdf_uniform_analytical = AnalyticalPDFs("Uniform", params_uniform)
analytical_values_uniform = pdf_uniform_analytical.get_values(l_values)
plot_frequency_vs_length(axs[0, 1], (l_values, analytical_values_uniform), 'Uniform Distribution Analytical', colors[0], analytical=True)


# Log-Normal with different sigmas
sigmaRange=[0.25, 0.5, 1]
colors = cm.Set1(np.linspace(0, 1, len(sigmaRange)))
for i in range(len(sigmaRange)):
    params_lognormal = {"mu": 0, "sigma": sigmaRange[i], "loc": 0}
    pdf_lognormal = PDFs("Log-Normal", params_lognormal)
    samples_lognormal = [pdf_lognormal.get_value() for _ in range(num_samples)]
    label = f'sigma={sigmaRange[i]}'
    plot_frequency_vs_length(axs[1, 0], samples_lognormal,"Log-Normal KDE", colors[i], label=label)
for i in range(len(sigmaRange)):
    params_lognormal = {"mu": 0, "sigma": sigmaRange[i], "loc": 0}
    pdf_lognormal_analytical = AnalyticalPDFs("Log-Normal", params_lognormal)
    analytical_values_lognormal = pdf_lognormal_analytical.get_values(l_values)
    label = f'sigma={sigmaRange[i]}'
    plot_frequency_vs_length(axs[1, 1], (l_values, analytical_values_lognormal),"Log-Normal anlytical", colors[i], label=label, analytical=True)

# Power-law with different alphas
alphaRange = [1.5, 2, 2.5, 3]
colors = cm.Set1(np.linspace(0, 1, len(alphaRange)))
for i in range(len(alphaRange)):
    params_powerlaw = {"alpha": alphaRange[i], "Lmin": 1, "Lmax": 10}
    pdf_powerlaw = PDFs("Negative power-law", params_powerlaw)
    samples_powerlaw = [pdf_powerlaw.get_value() for _ in range(num_samples)]
    label = f'alpha={alphaRange[i]}'
    plot_frequency_vs_length(axs[2, 0], samples_powerlaw, "Negative power-law KDE", colors[i], label=label)
for i in range(len(alphaRange)):
    params_powerlaw = {"alpha": alphaRange[i], "Lmin": 1, "Lmax": 10}
    pdf_powerlaw_analytical = AnalyticalPDFs("Negative power-law", params_powerlaw)
    analytical_values_powerlaw = pdf_powerlaw_analytical.get_values(l_values)
    label = f'alpha={alphaRange[i]}'
    plot_frequency_vs_length(axs[2, 1], (l_values, analytical_values_powerlaw), "Negative power-law Analytical", colors[i], label=label, analytical=True)

# Exponential with different lambdas
lambdaRange = [0.5, 1, 2]
colors = cm.Set1(np.linspace(0, 1, len(lambdaRange)))
for i in range(len(lambdaRange)):
    params_exponential = {"lambda": lambdaRange[i]}
    pdf_exponential = PDFs("Negative exponential", params_exponential)
    samples_exponential = [pdf_exponential.get_value() for _ in range(num_samples)]
    label = f'lambda={lambdaRange[i]}'
    plot_frequency_vs_length(axs[3, 0], samples_exponential, "Negative exponential KDE", colors[i], label=label)
for i in range(len(lambdaRange)):
    params_exponential = {"lambda": lambdaRange[i]}
    pdf_exponential_analytical = AnalyticalPDFs("Negative exponential", params_exponential)
    analytical_values_exponential = pdf_exponential_analytical.get_values(l_values)
    label = f'lambda={lambdaRange[i]}'
    plot_frequency_vs_length(axs[3, 1], (l_values, analytical_values_exponential), "Negative exponential Analytical", colors[i], label=label, analytical=True)

# Adjust the layout
plt.tight_layout()
plt.subplots_adjust(hspace=0.3, wspace=0.2, top=0.95, bottom=0.05, left=0.07, right=0.95)
plt.savefig('PDFS-KDEAndAnalytical.pdf', format='pdf', dpi=300)
