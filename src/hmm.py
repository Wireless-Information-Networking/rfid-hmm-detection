from hmmlearn import hmm
import numpy as np
import matplotlib.pyplot as plt
from output import one_tag_in_and_out

# Set random seed
# np.random.seed(42)

# Generate the samples
distance_to_reader = 0.6
cycles = 1000
mean_cycle_samples = 100
example = one_tag_in_and_out(cycles, mean_cycle_samples, distance_to_reader)

# Fit a HMM model to obtain the most likely sequence
# of tags for the received power samples X.
isdB = True
X = example[1,:].reshape(-1,1) if isdB else 10**(example[1,:]/10).reshape(-1,1)
Z = example[0,:].astype(int)

# Because we know when the tag is present, we can use this information
# to initialize the model
trans_one_to_zero = np.sum(np.abs(np.diff(Z) == -1))
trans_zero_to_one = np.sum(np.abs(np.diff(Z) == 1))

prob_zero_to_one = trans_zero_to_one / len(Z)
prob_one_to_zero = trans_one_to_zero / len(Z)

transmat = np.array([[1 - prob_zero_to_one, prob_zero_to_one],[prob_one_to_zero, 1 - prob_one_to_zero]])
means = np.array([[np.mean(X[Z == 0])], [np.mean(X[Z == 1])]])
covars = np.array([[np.var(X[Z == 0])], [np.var(X[Z == 1])]])

# Try multiple models to avoid local minimum
best_score = -np.inf
best_estimator = None
max_models = 2

for i in range(max_models):
    # Define the model
    # 2 states: {0: no tag, 1: tag}
    # 1D Gaussian emissions: received power
    # power is not gaussian, but we can use it as a first approximation
    print(f'Fitting model {i+1}/{max_models}')
    
    # Try inicializing the parameters of the model
    setInitParams = True
    init_params = 'st' if setInitParams else 'sctm'
    setParams = True
    params = 'st' if setParams else 'sctm'

    estimator = hmm.GaussianHMM(n_components=2,
                            implementation='log',
                            n_iter=1000,
                            tol=1e-10,
                            init_params=init_params,
                            params=params,
                            algorithm='viterbi'
                            )
    
    # our assumption is that the tag is always present
    # but we can start with a random distribution
    if setInitParams or setParams:
        # estimator.transmat_ = transmat
        estimator.means_ = means
        estimator.covars_ = covars

    estimator.fit(X)
    score = estimator.score(X)
    if score > best_score:
        print(f'New best score: {score}')
        best_score = score
        best_estimator = estimator

estimator = best_estimator

# Set up the plot style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'figure.figsize': (10, 8),
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 0.5,
    'lines.markersize': 2,
})

num_runs = 1
Z_hat = 0
accuracy = []
for i in range(num_runs):
    test = one_tag_in_and_out(cycles//100, mean_cycle_samples, distance_to_reader)
    X = test[1,:].reshape(-1,1) if isdB else 10**(test[1,:]/10).reshape(-1,1)
    Z = test[0,:].astype(int)
    Z_hat = estimator.predict(X)
    accuracy.append(np.mean(Z == Z_hat))

accuracy = np.array(accuracy)
mean_accuracy = np.mean(accuracy)
std_accuracy = np.std(accuracy)
print(f'Mean accuracy: {mean_accuracy*100:.6f}%')
print(f'Standard deviation of accuracy: {std_accuracy*100:.6f}%')

# plot the samples with different colors for each state
fig, axs = plt.subplots(2, 1, sharex=True)
pointSize = 10
# axs[0].plot(X, label='Received Power')
scatter0 = axs[0].scatter(range(len(X)), X, c=Z, cmap='viridis', label='Estado real', s=pointSize)
axs[0].set_title('Estado real de la etiqueta')
axs[0].set_ylabel('Potencia recibida (dBm)')
legend1 = axs[0].legend(handles=scatter0.legend_elements()[0], labels=['Ausente', 'Presente'], loc='lower left', frameon=True)
axs[0].add_artist(legend1)

# Z_hat = estimator.predict(X)
# axs[1].plot(X, label='Received Power')
scatter1 = axs[1].scatter(range(len(X)), X, c=Z_hat, cmap='viridis', label='Estado estimado', s=pointSize)
axs[1].set_title('Estado estimado de la etiqueta')
axs[1].set_xlabel('Muestras')
axs[1].set_ylabel('Potencia recibida (dBm)')
legend2 = axs[1].legend(handles=scatter1.legend_elements()[0], labels=['Ausente', 'Presente'], loc='lower left', frameon=True)
axs[1].add_artist(legend2)


#plt.tight_layout()
plt.show()