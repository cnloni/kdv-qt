#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def get_rootname(path):
    head, tail = os.path.split(path)
    root, ext = os.path.splitext(tail)
    return root

def get_prop(path):
    root = get_rootname(path)
    p = root.split("_")
    r = [int(p[1]), float(p[2]), float(p[3])]
    return r

npyfiles = [
    'kdv_256_1e-05_0.npy',
    'kdv_256_1e-05_1.npy',
    'kdv_256_1e-05_3.6.npy',
]
N, dt, T = get_prop(npyfiles[0])

rootname = get_rootname(sys.argv[0])
pngfile = rootname + '.png'

fig = plt.figure(figsize=[4.8, 3.6])
ax = fig.add_subplot(111)
ax.set_title(rootname)
ax.axis([0, N - 1, -1.1, 2.9])
x = range(N)

for npyfile in npyfiles:
    d = np.load(npyfile)
    ax.plot(x, d)

plt.show()

#plt.savefig(pngfile);
