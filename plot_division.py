import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-6,6,100)

line_theta_1 = np.abs(np.tan(80*np.pi/180)*x)
line_theta_2 = np.abs(np.tan(70*np.pi/180)*x)

plt.plot(x, line_theta_1, label='theta = 80')
plt.plot(x, line_theta_2, label='theta = 70')

plt.gca().set_aspect('equal', adjustable='box')
plt.legend()
plt.show()