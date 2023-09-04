import numpy as np
import matplotlib.pyplot as plt
  
def sigmoid(z):
    return 1 / (1 + np.exp( - z))
  
x = np.arange(-10, 10, 0.1)
plt.plot(x , sigmoid(x))
plt.title('Visualization of the Sigmoid Function: f(x) = 1 / (1 + exp( - x))')

x1 = np.arange(-2, 2, 0.1)
plt.plot(x, np.sin(x))

plt.show()
