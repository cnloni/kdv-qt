#! /usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def get_prop(filename):
    head, tail = os.path.split(filename)
    root, ext = os.path.splitext(tail)
    p = root.split("_")
    r = [int(p[1]), float(p[2]), float(p[3])]
    return r

npyfiles = [
    'kdv_256_1e-05_1_CPU.npy',
    'kdv_256_1e-05_3.6_CPU.npy',
]
N, dt, T = get_prop(npyfile)

root = 'fig1'
pngfile = root + '.png'

d = np.load(npyfile)

print("file=",npyfile)
print("N  = ", N)
print("dt = ", dt)
print("T  = ", T)

fig = plt.figure(figsize=[4.8, 3.6])
ax = fig.add_subplot(111)
ax.axis([0, N - 1, -1.1, 2.9])
x = range(N)
ax.plot(x, d)
ax.set_title('Fig.1')
plt.show()

#plt.savefig(pngfile);
