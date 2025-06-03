# plotting an histogram of the output data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, gamma, lognorm, norm
from output import one_tag_in_and_out

# Generate the samples
distance_to_reader = 0.5
cycles = 10000
mean_cycle_samples = 1000
XZ = one_tag_in_and_out(distance_to_reader=distance_to_reader, cycles=cycles, mean_cycle_samples=mean_cycle_samples)

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'figure.figsize': (10, 6),
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'lines.linewidth': 2,
    'lines.markersize': 6,
    'grid.alpha': 0.7,
    'grid.linestyle': '--',
    'grid.color': 'gray',
    'axes.grid': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.2,
    'text.usetex': False,  # Use LaTeX for text
})

# Histogram noise signal
X_noise_dB = XZ[1][XZ[0]==0]
hist, bins = np.histogram(X_noise_dB, bins=200, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Fit and plot normal distribution
mu, sigma = np.mean(X_noise_dB), np.std(X_noise_dB)
pdf = norm.pdf(bin_centers, mu, sigma)
plt.plot(bin_centers, pdf, color='r', linestyle='--', label='Distribución normal ajustada')

plt.plot(bin_centers, hist, color='b', alpha=0.7, label='Densidad de probabilidad estimada')
plt.title('Histograma ruido (dBm)')
plt.xlabel('Potencia (dBm)')
plt.ylabel('Frequencia relativa')
#plt.tight_layout()
plt.legend(frameon=True)
plt.xlim(-120, -70)
plt.show()

# Histogram signal
X_signal_dB = XZ[1][XZ[0]==1]
hist, bins = np.histogram(X_signal_dB, bins=200, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2

# Fit and plot normal distribution
mu, sigma = np.mean(X_signal_dB), np.std(X_signal_dB)
pdf = norm.pdf(bin_centers, mu, sigma)
plt.plot(bin_centers, pdf, color='r', linestyle='--', label='Dist. normal ajustada')

plt.plot(bin_centers, hist, color='b', alpha=0.7, label='Densidad de prob. estimada')
plt.title('Histograma señal (dBm)')
plt.xlabel('Potencia (dBm)')
plt.ylabel('Frequencia relativa')
# plt.tight_layout()
plt.legend(frameon=True)
plt.show()

# Histogram signal and noise
hist, bins = np.histogram(XZ[1], bins=200, density=True)
bin_centers = (bins[:-1] + bins[1:]) / 2
plt.plot(bin_centers, hist, color='b', alpha=0.7, label='Den')
plt.title('Histogram samples (dBm)')
plt.xlabel('Sample value (dBm)')
plt.ylabel('Relative Frequency')
plt.tight_layout()
plt.show()

# We conclude is best to use the logarithm of the received power samples
# as it shows a more normal distribution, in reality is a chi-squared distribution