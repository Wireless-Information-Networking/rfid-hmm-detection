# Treat the samples as independent and identically distributed (i.i.d.) random variables
# and fit the best distribution to the train data to estimate the presence of a tag in the signal.

import numpy as np
from scipy import stats
from output import one_tag_in_and_out
import matplotlib.pyplot as plt

# Dataset
cycles = 1000
mean_cycles = 100
distance = 0.75

dataset = one_tag_in_and_out(cycles, mean_cycles, distance)

signal_dB = dataset[1][dataset[0] == 1]
noise_dB = dataset[1][dataset[0] == 0]

isdB = True

increase_factor = 1e8 # to avoid numerical error
x = dataset[1] if isdB else 10**(dataset[1]/10) * increase_factor
noise = noise_dB if isdB else 10**(noise_dB/10) * increase_factor
signal = signal_dB if isdB else 10**(signal_dB/10) * increase_factor

# Fit to beta distribution (signal)
a_signal, b_signal, loc_beta_signal, scale_beta_signal = stats.beta.fit(signal)

# Fit to beta distribution (noise)
a_noise, b_noise, loc_beta_noise, scale_beta_noise = stats.beta.fit(noise)


num_rounds = 100
accuracy = []
for i in range(num_rounds):
    # Generate a new dataset
    dataset = one_tag_in_and_out(cycles, mean_cycles, distance)
    isdB = True
    x = dataset[1] if isdB else 10**(dataset[1]/10) * increase_factor

    # Calculate the pdf for each value in the dataset for signal and noise
    prob_signal = stats.beta.pdf(x, a_signal, b_signal, 
                                loc=loc_beta_signal, scale=scale_beta_signal)
    prob_noise = stats.beta.pdf(x, a_noise, b_noise, 
                            loc=loc_beta_noise, scale=scale_beta_noise)

    # Combine both arrays by taking the highest value of the pdf for each index
    combined_pdf = np.maximum(prob_signal, prob_noise)

    # Create an array of the same length that takes either a 1 when the biggest pdf is from the signal and 0 otherwise
    labels = (prob_signal > prob_noise).astype(int)

    # Compute accuracy
    accuracy.append(np.mean(labels == dataset[0]))

# Print the average accuracy
print(f'Average accuracy: {np.mean(accuracy):.6f} ± {np.std(accuracy):.6f}')

# Plot the combined pdf and the labels
plt.subplot(3, 1, 1)
plt.plot(x, prob_noise, color='red', alpha=0.5, label='Noise', marker='o', linewidth=0.0)
plt.plot(x, prob_signal, color='blue', alpha=0.5, label='Signal', marker='o', linewidth=0.0)
plt.title('PDF of signal and noise')
plt.xlabel('Received power')
plt.ylabel('Probability density')
plt.legend()

plt.subplot(3, 2, 3)
plt.plot(dataset[1])
plt.scatter(range(len(dataset[1])), dataset[1], c=labels, cmap='viridis')
plt.title('RSSI with expected labels')
plt.xlabel('Samples')
plt.ylabel('Received power')

plt.subplot(3, 2, 4)
plt.plot(dataset[1])
plt.scatter(range(len(dataset[1])), dataset[1], c=dataset[0], cmap='viridis')
plt.title('RSSI with true labels')
plt.xlabel('Samples')
plt.ylabel('Received power')

plt.subplot(3, 2, 5)
plt.plot(labels)
plt.title('State estimation')
plt.xlabel('Samples')
plt.ylabel('Tag presence (estimated)')

plt.subplot(3, 2, 6)
plt.plot(dataset[0])
plt.title('State')
plt.xlabel('Samples')
plt.ylabel('Tag presence')

plt.show()

