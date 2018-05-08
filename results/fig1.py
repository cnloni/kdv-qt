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
    return p[1:]


npyfiles = sys.argv[1:]
N, dt, T = get_prop(npyfiles[0])

rootname = get_rootname(sys.argv[0])
pngfile = rootname + '.png'

fig = plt.figure(figsize=[4.8, 3.6])
ax = fig.add_subplot(111)
ax.set_title(rootname)
ax.axis([0, int(N) - 1, -1.1, 2.9])
x = range(int(N))
labels = []

for npyfile in npyfiles:
    d = np.load(npyfile)
    N, dt, T = get_prop(npyfile)
    labels.append('T='+T)
    ax.plot(x, d)

ax.legend(labels)
plt.show()
