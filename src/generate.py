import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm

# Set random seed
np.random.seed(42)

# Define the model
# 2 states: {0: no tag, 1: tag}
# 1D Gaussian emissions: received power
model = hmm.GaussianHMM(n_components=2, covariance_type="full")

# equal probability of starting in either state
probabilityNoTag = 0.5
model.startprob_ = np.array([probabilityNoTag, 1 - probabilityNoTag]) 

# higher probability of staying in the same state
probabilityStay = 0.99 # jumps every 100 steps on average
model.transmat_ = np.array([[probabilityStay, 1 - probabilityStay], 
                            [1 - probabilityStay, probabilityStay]])

# mean and covariance of the two states
noiseMean = 0.0
tagMean = 1.0
noiseVariance = 1.0
model.means_ = np.array([[noiseMean], [tagMean]])
model.covars_ = np.tile(np.identity(1)*noiseVariance, (2, 1, 1))
print(model.get_params())

# get samples from the model
X, Z = model.sample(1000)

# Plot the samples of the received power with different colors for each state
plt.plot(X)
plt.scatter(range(len(X)), X, c=Z, cmap='viridis')
plt.title('Received power samples')
plt.xlabel('Samples')
plt.ylabel('Received power')
plt.show()