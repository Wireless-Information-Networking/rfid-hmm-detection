# test multiple distributions using distfit

import numpy as np
import matplotlib.pyplot as plt
from output import one_tag_in_and_out
from distfit import distfit

# Set random seed
np.random.seed(42)

# Generate the samples
distance_to_reader = 1
cycles = 1000
mean_cycle_samples = 1000
example = one_tag_in_and_out(cycles, mean_cycle_samples, distance_to_reader)

# Fit a HMM model to obtain the most likely sequence
signal_dB = example[1][example[0] == 1]
noise_dB = example[1][example[0] == 0]

isdB = False

increase_factor = 1e8 # to avoid numerical error
noise = noise_dB if isdB else 10**(noise_dB/10) * increase_factor
signal = signal_dB if isdB else 10**(signal_dB/10) * increase_factor

# Fit to multiple distributions (signal)
dist = distfit(distr='full', n_jobs=24)
dist.fit_transform(signal )
print(dist.summary)
dist.plot()
dist.plot_summary()

# We observe that one of the best fits for a huge dataset is a 
# gamma distribution, already selected in the previous notebook