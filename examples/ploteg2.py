import sys
sys.path.append("..")

import numpy as np
import matplotlib.pyplot as plt

from benchmarks import PWA

# pwa = PWA()

x = np.linspace(-5, 5, 1000)
y = np.linspace(-5, 5, 1000)
X, Y = np.meshgrid(x, y)
xys = np.stack([X, Y], axis=-1).reshape([-1, 2])
vals = np.linalg.norm(xys, axis=1)

# for i in range(len(vals)):
# 	if not pwa.in_domain(xys[i]):
# 		vals[i] = np.nan
vals[np.where(vals>=1)] = np.nan

fig = plt.figure()
# ax = plt.axes(projection='3d')
ax = plt.axes()

im = ax.contourf(X, Y, np.reshape(vals,[ 1000, 1000]))
ax.scatter(xys[::300,0], xys[::300,1], s=1, c='red')
# ax.plot3D(xys[:,0], xys[:,1], vals)
# ax.contour3D(X, Y, np.reshape(vals,[ 1000, 1000]), 50, cmap='binary')
# ax.plot_surface(X, Y, np.reshape(vals,[ 1000, 1000]), rstride=1, cstride=1,
#                 cmap='viridis', edgecolor='none')
fig.colorbar(im)
plt.show()