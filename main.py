import numpy as np
from scipy import signal

x = np.random.rand(3, 3)
y = np.ones((2, 2))
print(x)
print(y)
u = signal.fftconvolve(x, y, mode='same')
print(u)
