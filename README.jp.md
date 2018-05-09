# kdv-qt

![kdv-qt](https://github.com/cnloni/kdv-qt/raw/master/results/fig1.png)

## Prerequisites
This software is verified under following environments
+ CUDA 9.1
+ Debian 9.4

## Installing
```bash
# Compiling
$ make

# create the figure displayed above
$ make fig1
```

## Usage
```bash
$ bin/kdv-qt -N 256 -d 1e-5 -T 10
```

## Files
```
kdv-qt/
+-- LICENSE
+-- README.md
+-- makefile
+-- results/
|   +-- fig1.png
|   +-- fig1.py
+-- src/
    +-- CnlArray.h
    +-- KdV.cu
    +-- KdV.h
    +-- KdVDevice.cu
    +-- gb-kdv.cpp
```

### results/
Results are stored in this directory.

### src/
The source directory including \*.cu, \*.cpp and \*.h files.

## Copyright
see ./LICENCE

Copyright (c) 2018 HIGASHIMURA Takenori <oni@cynetlab.com>
