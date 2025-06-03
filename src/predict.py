# Fit a HMM model to obtain the most likely sequence 
# of tags for the received power samples X.

import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from generate import X, Z
from generate import model

# Set random seed
np.random.seed(42)

# Define the model
# 2 states: {0: no tag, 1: tag}
# 1D Gaussian emissions: received power
estimator = hmm.GaussianHMM(n_components=2, 
                        covariance_type="full", 
                        n_iter=1000,
                        tol=1e-5)

# fit the model to the data
estimator.fit(X)

# generate new samples
X_new, Z_new = model.sample(1000)

# predict the most likely sequence of tags for the new samples
Z_new_hat =  estimator.predict(X_new)



# Plot the samples of the received power with different 
# colors for each state
plt.subplot(2,1,1)
plt.plot(X)
plt.scatter(range(len(X)), X, c=Z_new, cmap='viridis')
plt.title('Received power samples')
plt.xlabel('Samples')
plt.ylabel('Received power')

# Plot the true tags
plt.subplot(2,1,2)
plt.plot(X_new)
plt.scatter(range(len(X_new)), X_new, c=Z_new_hat, cmap='viridis')
plt.title('Received power samples')
plt.xlabel('Samples')
plt.ylabel('Received power')
plt.show()


