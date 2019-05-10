import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import math
from plot_ex1 import plot_ex1, figure

mu0 = np.array([0, 0]).T
Sigma0 = np.array([[1, 0], [0, 1]])
mu1 = np.array([1, 1]).T
Sigma1 = np.array([[1, 0], [0, 1]])
phi = 0.5

# begin answer
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line', 1)


mu0 = 0
Sigma0 = 0
mu1 = 0
Sigma1 = 0
phi = 0

# begin answer
# end answer

plot_ex1(mu0, Sigma0, mu1, Sigma1, phi, 'Line (one side)', 2)
plt.show()