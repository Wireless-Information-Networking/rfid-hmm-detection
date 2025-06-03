import numpy as np
from matplotlib import pyplot as plt


# MLE
distanceMLE = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
mediaMLE = [0.999996, 0.99999075, 0.9999662, 0.976882, 0.9053966, 0.7948562, 0.689135, 0.6117456, 0.5628954, 0.5337818, 0.517103, 0.5088178, 0.5027318, 0.5007668]
mediaMLE = np.multiply(mediaMLE, 100)

# HMM
distanciaHMM = [0.1, 0.2, 0.3, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
mediaHMM = [99.9995468, 99.9995192, 99.9986584, 99.9691778, 99.8558968, 99.4814874, 98.2977078, 96.0705198, 88.9564854, 84.9226486, 65.693088, 56.1992112, 52.5768506, 50.5875992]

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
    'lines.markersize': 3,
})

# Plot the results with error bars
plt.plot(distanceMLE, mediaMLE, label='MLE', marker='o', linestyle='--')
plt.plot(distanciaHMM, mediaHMM, label='HMM', marker='^', linestyle='-')
plt.title('MLE vs HMM')
plt.xlabel('Distancia (m)')
plt.ylabel('Precisión (%)')
#plt.xticks(distancia)
plt.legend(frameon=True)
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()