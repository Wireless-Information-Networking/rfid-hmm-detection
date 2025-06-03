# plotting an histogram of the output data
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2, gamma, lognorm
from output import one_tag_in_and_out

# Generate the samples
distance_to_reader = 0.5
cycles = 1000
mean_cycle_samples = 1000
XZ = one_tag_in_and_out(distance_to_reader=distance_to_reader, cycles=cycles, mean_cycle_samples=mean_cycle_samples)

# Histogram signal
X_noise_dB = XZ[1][XZ[0]==0]
plt.hist(X_noise_dB, bins=200, color='b', alpha=0.5, 
         label='Received power', density=True)
plt.title('Histograma ruido (dBm)')
plt.xlabel('Potencia (dBm)') 
plt.ylabel('Frequencia relativa')
plt.show()


# Histogram signal
X_signal_dB = XZ[1][XZ[0]==1]
plt.hist(X_signal_dB,bins=200, color='b', alpha=0.5, 
         label='Received power', density=True)
plt.title('Histograma señal (dBm)')
plt.xlabel('Potencia (dBm)') 
plt.ylabel('Frequencia relativa')
plt.show()

# Histogram signal and noise
plt.hist(XZ[1], bins=200, color='b', alpha=0.5, 
         label='Tag presence', density=True)
plt.title('Histogram samples (dBm)')
plt.xlabel('Sample value (dBm)') 
plt.ylabel('Relative Frequency')
plt.show()


# We conclude is best to use the logarithm of the received power samples
# as it shows a more normal distribution, in reality is a chi-squared distribution
