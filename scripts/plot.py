import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

PI = np.pi
def spiky_kernel(r, h):
    v = h - r
    s = 15.0 / (PI * pow(h, 6))
    return np.where(v < 1, v * v * v * s, 0)

def cubic_kernel(r, h):
    h1 = 1.0 / h
    q = r * h1

    fac = (1.0 / PI) * h1 * h1 * h1

    tmp2 = 2.0 - q

    val = np.where(q > 2.0, 0, np.where(q > 1.0, 0.25 * tmp2 * tmp2 * tmp2, 1 - 1.5 * q * q * (1 - 0.5 * q)))

    return val * fac


def calculate_distance(x, y, z):
    return np.sqrt(x**2 + y**2 + z**2)

x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
z = np.linspace(-1, 1, 100)
X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

# Calculate distance from the origin
distance = calculate_distance(X, Y, Z)

# Set the value of h (smoothing length)
h = 1.0

# Calculate kernel values
kernel_values = spiky_kernel(distance, h)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.set_title('3D Cubic Spline Kernel Function')

# Plot the surface
surf = ax.plot_surface(X[:,:,0], Y[:,:,0], kernel_values[:,:,50], cmap='viridis') # Using only the central layer of Z
fig.colorbar(surf)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()
