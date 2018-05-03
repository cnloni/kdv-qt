#! /bin/bash
DATA=results/kdv-qt.dat
rm -f ${DATA}
for i in {1..10};do
bin/kdv-qt 256 1e-5 10 >>${DATA}
bin/kdv-qt 512 1e-5 10 >>${DATA}
bin/kdv-qt 1024 1e-5 10 >>${DATA}
bin/kdv-qt 256 1e-6 10 >>${DATA}
bin/kdv-qt 512 1e-6 10 >>${DATA}
bin/kdv-qt 1024 1e-6 10 >>${DATA}
done
