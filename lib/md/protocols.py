from lib.global_fun import *
from lib.global_vars import CONSSCORTK_THIRDPARTY_DIR

############################################ OLD PROTOCOLS ##########################################
#     if exhaustiveness == 0:  # run 100 ps and just save the last frame for SQM/COSMO scoring.
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=100
# csp=100
# pocketps=100
# pocketsp=100
# bps=80
# mps=48
# msp=6
#     """
#
#     if exhaustiveness == 1: # run 100 ps, but use the last 48 ps for analysis
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=100
# csp=1
# pocketps=50
# pocketsp=1
# bps=80
# mps=48
# msp=6
# """
#
#     if exhaustiveness == -1:    # run 200 ps, but use the last 48 ps for analysis
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=200
# csp=1
# pocketps=50
# pocketsp=1
# bps=80
# mps=48
# msp=6
# """
#
#     elif exhaustiveness == 2:
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=1000
# csp=10
# pocketps=500
# pocketsp=10
# bps=800
# mps=480
# msp=30
# """
#
#     elif exhaustiveness == -2:
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=1500
# csp=10
# pocketps=1000
# pocketsp=10
# bps=800
# mps=480
# msp=30
# """
#
#     elif exhaustiveness == 3:   # run 2.5 ns, but use the last 2 ns for analysis
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=2500
# csp=25
# pocketps=2000
# pocketsp=25
# bps=2000
# mps=2000
# msp=50
# """
#
#     elif exhaustiveness == -3:  # run 5 ns, but use only the last 2 ns for analysis (recommended for poorly aligned ligands)
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=5000
# csp=25
# pocketps=2000
# pocketsp=25
# bps=2000
# mps=2000
# msp=50
# """
#
#     elif exhaustiveness == 4:   # run 5 ns, but use the last 4 ns for analysis
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=5000
# csp=50
# pocketps=4000
# pocketsp=50
# bps=4000
# mps=4000
# msp=100
# """
#
#     elif exhaustiveness == -4:  # run 7 ns, but use only the last 4 ns for analysis (recommended for poorly aligned ligands)
#         parameters = """
# igb=5
# mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
# ff='ff14SB'
# cms=1000
# cps=7000
# csp=50
# pocketps=4000
# pocketsp=50
# bps=4000
# mps=4000
# msp=100
# """
####################################################################################################



protocol_parameters_dict = {}

# Run only 10 minimization steps and save only the coordinates.
protocol_parameters_dict["nomin"] = """
igb=5
mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
ff='ff14SB'
etolratio=0
cms=10
pmdcms=0
cps=0
csp=10
pocketps=10
pocketsp=10
bps=8
mps=2
msp=2
rmask=None
wcut=20
    """

# Run only 5000 steps minimization and save only the coordinates.
protocol_parameters_dict["onlymin"] = """
igb=5
mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
ff='ff14SB'
etolratio=5e-5
cms=5000
pmdcms=0
cps=0
csp=100
pocketps=100
pocketsp=100
bps=80
mps=48
msp=6
rmask=None
wcut=20
    """
# Run 100 ps MD + 5000 steps post-MD minimization and save only the coordinates after pre-MD minimization,
# upon the end of MD, and after post-MD minimization.
protocol_parameters_dict["100ps"] = """
igb=5
mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
ff='ff14SB'
etolratio=5e-5
cms=5000
pmdcms=5000
cps=100
csp=100
pocketps=100
pocketsp=100
bps=80
mps=48
msp=6
rmask=None
wcut=20
    """
# Run 1 ns MD + 5000 steps post-MD minimization and save only the coordinates after pre-MD minimization,
# upon the end of MD, and after post-MD minimization.
protocol_parameters_dict["1ns"] = """
igb=5
mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
ff='ff14SB'
etolratio=5e-5
cms=5000
pmdcms=5000
cps=100
csp=1000;   # <== CHANGE ME
pocketps=1000
pocketsp=1000
bps=800
mps=480
msp=60
rmask=None
wcut=20
    """
# Run 5 ns MD + 5000 steps post-MD minimization (etolratio=5e-5 will probaly stop it earlier) and save only the coordinates
# after pre-MD minimization, upon the end of MD, and after post-MD minimization.
protocol_parameters_dict["5ns"] = """
igb=5
mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
ff='ff14SB'
etolratio=5e-5
cms=5000
pmdcms=5000
cps=5000
csp=100;   # <== CHANGE ME
pocketps=5000
pocketsp=5000
bps=1600
mps=960
msp=120
rmask=None
wcut=20
    """

# Run 10 ns MD + 5000 steps post-MD minimization (etolratio=5e-5 will probaly stop it earlier) and save only the coordinates
# after pre-MD minimization, upon the end of MD, and after post-MD minimization.
protocol_parameters_dict["10ns"] = """
igb=5
mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
ff='ff14SB'
etolratio=5e-5
cms=5000
pmdcms=5000
cps=10000
csp=100;   # <== CHANGE ME
pocketps=0
pocketsp=10000
bps=0
mps=10000
msp=100
rmask=None
wcut=20
    """

protocol_parameters_dict["1microsec"] = """
igb=5
mdigb=0 ; # do in vacuo MD, otherwise the waters nucleate
ff='ff14SB'
etolratio=5e-5
cms=5000
pmdcms=5000
cps=1000000
csp=1000;   # <== CHANGE ME
pocketps=0
pocketsp=1000000
bps=0
mps=1000000
msp=1000
rmask=None
wcut=20
    """

protocol_parameters_dict["onlymin_flex"] = protocol_parameters_dict["onlymin"].replace("rmask=None", "rmask='flex'")
protocol_parameters_dict["100ps_flex"] = protocol_parameters_dict["100ps"].replace("rmask=None", "rmask='flex'")
protocol_parameters_dict["1ns_flex"] = protocol_parameters_dict["1ns"].replace("rmask=None", "rmask='flex'")
protocol_parameters_dict["5ns_flex"] = protocol_parameters_dict["5ns"].replace("rmask=None", "rmask='flex'")
protocol_parameters_dict["10ns_flex"] = protocol_parameters_dict["10ns"].replace("rmask=None", "rmask='flex'")
protocol_parameters_dict["1microsec_flex"] = protocol_parameters_dict["1microsec"].replace("rmask=None", "rmask='flex'")

def write_MD_scripts(pdb, protocol, GPU_allocation_list="(1,2,1)", MMGBSA_ARGS=[]):

    ColorPrint("Writing MD scripts.", "BOLDGREEN")
    if protocol == 'nomin': MMGBSA_ARGS = ['onlytopol', '']
    pc_gpusThreads_dict = {}
    for pc, gpucount_threads_jobloadratio in enumerate(replace_alt(GPU_allocation_list, ['),(', '(', ')'], ' ').split()):
        gpucount, threads, jobload_ratio = gpucount_threads_jobloadratio.split(',')
        pc_gpusThreads_dict[str(pc)] = [" ".join([str(g) for g in range(int(gpucount))]), threads]

    for pc in pc_gpusThreads_dict.keys():
        compound_file = "unfinished_molnames.PC%s.list" % pc
        out = '''#!/bin/bash

##~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ SET GLOBAL PARAMETERS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##
%s
jobs_per_GPU=%s
GPUs=(%s)	; # array with the device IDs
receptor=%s
## ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~##

# copy mol2 files to skip the cumbersome semi-qm calculations
run_protein_MD.py -r ${receptor} -ms 1000 -etolratio $etolratio -ps 0 -ssp 0 -igb $igb -ff $ff -exe 'openmm' \\
#-mol2  \\
#-frcmod  \\
#-leaprc  \\

# Initialize all 'active_jobs' files to 0
for gpu in ${GPUs[@]};
do
echo 0 > active_job_count.GPU${gpu}
done

# WRITE A FILE WITH THE COMMANDS TO RUN MD FOR EACH COMPOUND IN PARALLEL
for mol2 in $(cat %s);
do
molname=$(awk '{if(NR==2){print(tolower($0))}}' $mol2);
MOLNAME=$(awk '{if(NR==2){print $0}}' $mol2);
[ -e ${MOLNAME}/complex_forscoring_frm.1.min.pdb ] && continue;
#basemolname=$(perl -p -e 's/_pose[0-9]+//' <<< $molname)    # molname without the _pose[0-9] suffix
#chargemol2=$(ls charges/bcc/${basemolname}.bcc.mol2)   # one charge file for all poses of the same structvar
chargemol2=$(ls charges/bcc/${molname}.bcc.mol2)    # a separate charge file for each pose

# Save the command line for this compound to a file
echo """
while true;
do

job_completed=false;

for gpu in ${GPUs[@]};
do

active_jobs=\$(head -1 active_job_count.GPU\${gpu});

if [ \$active_jobs -lt $jobs_per_GPU ];
then
let \"active_jobs++\";
echo \"\$active_jobs\" > active_job_count.GPU\${gpu};
run_MMGBSA.py \
-device \$gpu \
-ff $ff \
-r $receptor \
-l $mol2 \
-rp receptor_MD/protein.prmtop \
-rn receptor_MD/protein.prod_GB.nc \
-etolratio $etolratio \
-rmask $rmask \
-wcut $wcut \
-cms $cms \
-cps $cps \
-csp $csp \
-pocketps $pocketps \
-pocketsp $pocketsp \
-igb $igb \
-mdigb $mdigb \
-cf $chargemol2 \
-lps 0 \
-bps $bps \
-mps $mps \
-msp $msp \
-exe 'openmm' \
-clean \
-onlymd %s;
active_jobs=\$(head -1 active_job_count.GPU\${gpu});
let \"active_jobs--\";
echo \"\$active_jobs\" > active_job_count.GPU\${gpu};
job_completed=true;
break	;
fi;

done;

[ \$job_completed == true ] && break;
sleep 5	;

done;
""" | perl -p -e "s/\\n/ /g"
echo ""


## SOME EXTRA run_MMGBSA.py arguments
#-onlytopol \
#-onlymmgbsa \
#-mol2  \
#-frcmod  \
#-leaprc  

done > commands.PC%s.list


## LAUNCH PARALLEL EXECUTION ON GPUS
N=`bc <<< "$jobs_per_GPU * ${#GPUs[@]}"`	; # number of threads
%s/parallel -j $N < commands.PC%s.list


# FINALLY, CLEAN DEBRIS
rm active_job_count.GPU*

    ''' % (protocol_parameters_dict[protocol],
           pc_gpusThreads_dict[pc][1],
           pc_gpusThreads_dict[pc][0],
           pdb,
           compound_file,
           " ".join(["-%s %s" % (a,v) for a,v in zip(MMGBSA_ARGS[0::2], MMGBSA_ARGS[1::2])]),
           pc,
           CONSSCORTK_THIRDPARTY_DIR,
           pc)

        script_name = "calc_FE.PC%s.sh" % pc

        with open(script_name, 'w') as f:
            f.write(out)

    run_commandline("chmod 777 calc_FE*.sh")