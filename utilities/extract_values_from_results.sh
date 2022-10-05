#!/bin/bash

# to extract the SQM-ML AUC-ROC values
logfile=SQM-ML_weights_results.log
egrep "XTEST|model =" $logfile | perl -p -e "s/XTEST: \[(.*)\]$/\1/" | perl -p -e "s/.* model = ([0-9.]+) DOR .*/\1/" | perl -p -e "s/\'\n/\' /" | perl -p -e "s/\'//g"

# to extract Glide scores
egrep "Generating|AUC-ROC" $logfile | perl -p -e "s/.*nofusion_r_i_docking_score = ([0-9.]+) .*$/\1/" | perl -p -e "s/Generating features for ([A-Za-z0-9-]+)\n/\1 /" | awk '{print $2}'
