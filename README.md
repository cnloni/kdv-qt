# kdv-qt
<<<<<<< HEAD
Solve the KdV equation with CUDA

![kdv-qt](https://github.com/cnloni/kdv-qt/raw/master/results/fig1.png)

## Description
This project is for the article ['Solve the KdV equaton with CUDA' (in Japanese)](https://qiita.com/cnloni/items/a9826dc961401dbd2e08) at the Qiita site. Executable form (kdv-qt) provides the spacial pattern of the numerical solution at the final time T. Almost all parameters of the KdV equation follows the paper written by N. J. Zabusky and M. D. Kruska[(1)].  

## Required
This software is verified under the following environments
+ CUDA 9.1
+ Debian 9.4

To create the figure, required are
+ Python 3+
+ MatplotLib 2+
+ NumPy

## Install
```bash
# compile
$ make

# create the figure displayed above
$ make fig1
```

## Usage
```bash
$ bin/kdv-qt -N <the value of N> -d <the value of dt> -T < the value of T>
```
where N is the number of spacial units, dt the time interval of the calculation, and T the final time.

## Files
```
kdv-qt/
+-- LICENSE
+-- README.md
+-- makefile
+-- bin/
|   +-- kdv-qt   
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

### bin/
Automatically created to store the executable form (kdv-qt).

### results/
Results are stored in this directory.

### src/
The source directory including \*.cu, \*.cpp and \*.h files.

## License
see ./LICENCE

Copyright (c) 2018 HIGASHIMURA Takenori <oni@cynetlab.com>

[(1)]: https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.15.240
=======

## Copyright
see ./LICENCE

Copyright (c) 2018 HIGASHIMURA Takenori <oni@cynetlab.com>
>>>>>>> refs/remotes/eclipse_auto/master
