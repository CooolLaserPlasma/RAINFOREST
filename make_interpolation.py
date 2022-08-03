import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d

A = np.load("test_pattern.npy")

plt.imshow(A)
plt.show()

x = np.arange(0,200,1)
y = np.arange(0,200,1)

A_interp_fun = interp2d(x,y,A,kind='cubic')

y = np.linspace(0,200,2000)

A_interp = A_interp_fun(y,y)

plt.imshow(A_interp)
plt.show()

threshold = 0.7
A_binary = np.zeros(A_interp.shape)
A_binary[A_interp > threshold] = 1
A_binary[A_interp <= threshold] = 0

plt.imshow(A_binary)
plt.show()
