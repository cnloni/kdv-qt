#! /bin/bash
DATA=results/kdv-qt.dat
rm -f ${DATA}
Release/kdv-qt -N 256 -d 1e-5 -T 10
Release/kdv-qt -N 512 -d 1e-5 -T 10
Release/kdv-qt -N 1024 -d 1e-5 -T 10
Release/kdv-qt -N 256 -d 5e-6 -T 10
Release/kdv-qt -N 512 -d 5e-6 -T 10
Release/kdv-qt -N 1024 -d 5e-6 -T 10
