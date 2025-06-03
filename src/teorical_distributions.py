import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2,lognorm,gamma

# Parámetros
k = 4 # grados de libertad
NOISE_POWER_dBm = -90 # UNKNOWN
c = 10**(NOISE_POWER_dBm/10)  # constante de escala

# Generar datos distribuidos chi-cuadrado
x = np.logspace(-11, -7, 10000)
x_db = 10*np.log10(x)
y = chi2.pdf(x, k)

# Generar la nueva distribución después de multiplicar por la constante c
y_new = gamma.pdf(x, k/2, scale = 2*c)  

# Graficar las distribuciones original y nueva
# plt.plot(x_db, y, label='Distribución Chi-Cuadrado Original')
plt.plot(x_db, y_new, label=f'Distribución Chi-Cuadrado multiplicada por {c}')
plt.xlabel('Señal [dB]')
plt.ylabel('Función de Densidad de Probabilidad')
plt.title('Densidad de probabilidad de la potencia del ruido blanco')
# plt.legend()
plt.grid(True)
plt.show()

# Parámetros
mu = 0  # media de la distribución lognormal
sigma = 1  # desviación estándar de la distribución lognormal
k = 2  # grados de libertad para la distribución chi-cuadrada

# Generar datos para las distribuciones lognormal y chi-cuadrada
x = np.linspace(0, 20, 1000)
lognormal_pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))
chi2_pdf = chi2.pdf(x, k)

# Convolución de las dos PDFs para obtener la PDF de su suma
sum_pdf = np.convolve(lognormal_pdf, chi2_pdf, mode='same')
sum_pdf /= np.trapz(sum_pdf, x)  # Normalizar la PDF

# Graficar las PDFs individuales y la suma
plt.plot(x, lognormal_pdf, label='Distribución Lognormal')
plt.plot(x, chi2_pdf, label='Distribución Chi-Cuadrado')
plt.plot(x, sum_pdf, label='Suma de Distribuciones Lognormal y Chi-Cuadrado')
plt.xlabel('x')
plt.ylabel('Función de Densidad de Probabilidad')
plt.title('PDF de la Suma de Distribuciones Lognormal y Chi-Cuadrado')
plt.legend()
plt.grid(True)
plt.show()
