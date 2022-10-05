#!/bin/bash
# Script to repeat 10 times the leave-one-out cross-validation and compute AUC-ROC values.

suffix=""

for repeat in $(seq 2 10)
do
#for n_comp in 3 4 5 6 7 8 9 10 20 30 40 50 60 70 80 90 100
#do
../EXEC_master_script.py -n_comp $n_comp -xtp 'A2A,ACHE,AR,CATL,DHFR,EPHB4,GBA,GR,HIV1RT,JNK2,MDM2,MK2,PARP-1,PPARG,SARS-HCoV,SIRT2,TPA,TP' >& ../../sqm-ml_data/execution_dir${suffix}/results_${n_comp}D-UMAP_repeat${repeat}.log
#../EXEC_master_script.py -xtp 'A2A,ACHE,AR,CATL,DHFR,EPHB4,GBA,GR,HIV1RT,JNK2,MDM2,MK2,PARP-1,PPARG,SARS-HCoV,SIRT2,TPA,TP' >& ../../sqm-ml_data/execution_dir${suffix}/results_repeat${repeat}.log
#done
done