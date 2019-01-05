import math
import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from math import pi
from scipy.interpolate import griddata


def func(x, y):
    return x*(1-x)*np.cos(4*np.pi*x) * np.sin(4*np.pi*y**2)**2


def circ(center=(0, 0), r=50, n=100):
    return [
        (
            center[0]+(math.cos(2 * pi / n * x) * r),  # x
            center[1] + (math.sin(2 * pi / n * x) * r)  # y

        ) for x in xrange(0, n + 1)]


grid_x, grid_y = np.mgrid[0:1:1000j, 0:1:1000j]

points = [[0.25, 0.5]]
values = [-1]

points += circ((0.25, 0.5), 0.2, 25)
values += [0 for x in points][:-1]

grid_z1 = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)

points = [[0.75, 0.5]]
values = [1]

points += circ((0.75, 0.5), 0.5, 25)
values += [0 for x in points][:-1]

grid_z2 = griddata(points, values, (grid_x, grid_y), method='linear', fill_value=0)

grid_zs = np.add(grid_z1, grid_z2)
grid_za = np.add(np.abs(grid_z1), np.abs(grid_z2))

grid_zo = np.subtract(grid_za, np.abs(grid_zs))

# from pprint import pprint
# for i in grid_z2:
#     pprint(i)

cdict_rbg = [(1, 0, 0), (0, 0, 1), (0, 1, 0)]
cdict_rg = [(1, 0, 0), (1, 1, 1), (0, 1, 0)]
cdict_b = [(1, 1, 1), (0, 0, 1)]

cmap_rbg = matplotlib.colors.LinearSegmentedColormap.from_list('cmap_rbg', cdict_rbg, N=256)
cmap_rg = matplotlib.colors.LinearSegmentedColormap.from_list('cmap_rg', cdict_rg, N=256)
cmap_b = matplotlib.colors.LinearSegmentedColormap.from_list('cmap_b', cdict_b, N=256)


# plt.imshow(grid_zo.T, extent=(0, 1, 0, 1), cmap=cmap_b)
plt.imshow(grid_zo.T, extent=(0, 1, 0, 1), cmap=plt.get_cmap('Greys_r'))
# plt.imshow(grid_zs.T, extent=(0, 1, 0, 1), cmap=cmap_rg)
plt.title('Cubic')
plt.gcf().set_size_inches(6, 6)
plt.show()
