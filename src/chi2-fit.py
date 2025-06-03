# Create expected Maximization algorithm to determine the presence 
# of a tag in a signal

import numpy as np
from scipy import stats
from output import one_tag_in_and_out
import matplotlib.pyplot as plt

# Dataset
cycles = 100
mean_cycles = 1000
distance = 0.6

dataset = one_tag_in_and_out(cycles, mean_cycles, distance)

increase_factor = 1e8
noise = 10**(dataset[1][dataset[0] == 0]/10) * increase_factor
signal = 10**(dataset[1][dataset[0] == 1]/10) * increase_factor

# Fit to chisquare distribution
df_noise, loc_noise, scale_noise = stats.chi2.fit(noise)
df_signal, loc_signal, scale_signal = stats.chi2.fit(signal)

# Obtain distributions
definition = 100000
x_signal = np.linspace(np.min(signal),
                          np.max(signal),
                          definition)
chi2_signal = stats.chi2.pdf(x_signal, 
                             df_signal, 
                             loc_signal,
                             scale_signal)

x_noise = np.linspace(np.min(noise),
                          np.max(noise),
                          definition)

chi2_noise = stats.chi2.pdf(x_noise, 
                            df_noise, 
                            loc_noise,
                             scale_noise)

# Show estimated and histogram
plt.subplot(2, 1, 1)
plt.hist(signal, bins=100, density=True, label="samples")
plt.plot(x_signal, chi2_signal)
plt.title('Signal histogram vs chi2 best fit')

plt.subplot(2, 1, 2)
plt.hist(noise, density=True, label="samples")
plt.plot(x_noise, chi2_noise)
plt.title('Noise histogram vs chi2 best fit')
plt.show()

# When trying the best fit with chisquare 
# we don't get a necessarily good fit, 
# we will try with other distributions and find the one
# that fits best the data