# Create expected Maximization algorithm to determine the presence 
# of a tag in a signal

import numpy as np
from scipy import stats
from output import one_tag_in_and_out
import matplotlib.pyplot as plt

# Dataset
cycles = 1000
mean_cycles = 100
distance = 1

dataset = one_tag_in_and_out(cycles, mean_cycles, distance)
signal_dB = dataset[1][dataset[0] == 1]
noise_dB = dataset[1][dataset[0] == 0]

isdB = True

increase_factor = 1e8 # to avoid numerical error
noise = noise_dB if isdB else 10**(noise_dB/10) * increase_factor
signal = signal_dB if isdB else 10**(signal_dB/10) * increase_factor

# Fit to multiple distributions (signal)
# fit lognormal distribution
sigma, loc_log, scale_log = stats.lognorm.fit(signal)

# fit gamma distribution
shape, loc_gamma, scale_gamma = stats.gamma.fit(signal)

# fit beta distribution
a, b, loc_beta, scale_beta = stats.beta.fit(signal)

# Evaluate multiple distributions (signal)
# evaluate lognormal distribution
lognormal_test = stats.kstest(signal, stats.lognorm.cdf, args=(sigma, loc_log, scale_log))

# evaluate gamma distribution
gamma_test = stats.kstest(signal, stats.gamma.cdf, args=(shape, loc_gamma, scale_gamma))

# evaluate beta distribution
beta_test = stats.kstest(signal, stats.beta.cdf, args=(a, b, loc_beta, scale_beta))

if lognormal_test.statistic < gamma_test.statistic and lognormal_test.statistic < beta_test.statistic:
    print("Lognormal distribution is the best fit.")
elif gamma_test.statistic < lognormal_test.statistic and gamma_test.statistic < beta_test.statistic:
    print("Gamma distribution is the best fit.")
else:
    print("Beta distribution is the best fit.")

# For runs with distance under 0.5 we get that the best fit is the lognorm
# otherwise we get the gamma dist as best fit

# we generate the pdf for every fit to compare
definition = 100000
x_signal = np.linspace(np.min(signal),
                          np.max(signal),
                          definition)

lognorm_signal = stats.lognorm.pdf(x_signal, 
                             sigma, 
                             loc_log, 
                             scale_log)

gamma_signal = stats.gamma.pdf(x_signal,
                               shape,
                               loc_gamma,
                               scale_gamma)

beta_signal = stats.beta.pdf(x_signal,
                               a, b,
                               loc_beta,
                               scale_beta)

# Show estimated and histogram
plt.subplot(3,2,1)
plt.hist(signal, bins=100, density=True, label="samples")
plt.plot(x_signal, lognorm_signal)
plt.title('Signal histogram vs lognorm best fit')

plt.subplot(3,2,3)
plt.hist(signal, bins=100, density=True, label="samples")
plt.plot(x_signal, gamma_signal)
plt.title('Signal histogram vs gamma best fit')

plt.subplot(3,2,5)
plt.hist(signal, bins=100, density=True, label="samples")
plt.plot(x_signal, beta_signal)
plt.title('Signal histogram vs beta best fit')

# Fit to multiple distributions (noise)
# fit lognormal distribution
sigma, loc_log, scale_log = stats.lognorm.fit(noise)

# fit gamma distribution
shape, loc_gamma, scale_gamma = stats.gamma.fit(noise)

# fit beta distribution
a, b, loc_beta, scale_beta = stats.beta.fit(noise)

# Evaluate multiple distributions (noise)
# evaluate lognormal distribution
lognormal_test = stats.kstest(noise, stats.lognorm.cdf, args=(sigma, loc_log, scale_log))

# evaluate gamma distribution
gamma_test = stats.kstest(noise, stats.gamma.cdf, args=(shape, loc_gamma, scale_gamma))

# evaluate beta distribution
beta_test = stats.kstest(noise, stats.beta.cdf, args=(a, b, loc_beta, scale_beta))

if lognormal_test.statistic < gamma_test.statistic and lognormal_test.statistic < beta_test.statistic:
    print("Lognormal distribution is the best fit.")
elif gamma_test.statistic < lognormal_test.statistic and gamma_test.statistic < beta_test.statistic:
    print("Gamma distribution is the best fit.")
else:
    print("Beta distribution is the best fit.")

# we generate the pdf for every fit to compare
x_noise = np.linspace(np.min(noise),
                          np.max(noise),
                          definition)

lognorm_noise = stats.lognorm.pdf(x_noise, 
                             sigma, 
                             loc_log, 
                             scale_log)

gamma_noise = stats.gamma.pdf(x_noise,
                               shape,
                               loc_gamma,
                               scale_gamma)

beta_noise = stats.beta.pdf(x_noise,
                               a, b,
                               loc_beta,
                               scale_beta)

# Show estimated and histogram
plt.subplot(3,2,2)
plt.hist(noise, bins=100, density=True, label="samples")
plt.plot(x_noise, lognorm_noise)
plt.title('noise histogram vs lognorm best fit')

plt.subplot(3,2,4)
plt.hist(noise, bins=100, density=True, label="samples")
plt.plot(x_noise, gamma_noise)
plt.title('noise histogram vs gamma best fit')

plt.subplot(3,2,6)
plt.hist(noise, bins=100, density=True, label="samples")
plt.plot(x_noise, beta_noise)
plt.title('noise histogram vs beta best fit')

plt.show()

# The gamma distribution is an overall better estimate of the pdf when the data is in linear scale
# The beta distribution is the best distribution when the data is in dBm scale