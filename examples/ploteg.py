import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d


x = np.linspace(-2, 2, 1000)
y = np.linspace(-2, 2, 1000)
X, Y = np.meshgrid(x, y)
xys = np.stack([X, Y], axis=-1).reshape([-1, 2])
vals = np.linalg.norm(xys, axis=1)
vals[np.where(vals >= 1)] = np.nan
# vals = np.reshape(vals,[ 1000, 1000])


fig = plt.figure()
ax = plt.axes(projection='3d')


# ax.plot3D(xys[:,0], xys[:,1], vals)
ax.contour3D(X, Y, np.reshape(vals,[ 1000, 1000]), 50, cmap='binary')
# ax.plot_surface(X, Y, np.reshape(vals,[ 1000, 1000]), rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
plt.show()