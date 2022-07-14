import os
import sys

import numpy as np

# Import ConsScorTK libraries
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )  # import the top level package directory
from . import ConsScoreTK_Statistics


def write_plumed_shape_input(molname, atom_groups, NOT_MASS_WEIGHTED='off'):
    
    # random.seed() # initialize seed according to system time, otherwise all plumed input filenames will have the same random integer
    #infname = "plumed_input."+str(random.randint(0,1000000000))
    infname = "plumed_input."+molname
    f = open(infname, 'w')
    f.write("""
GROUP ATOMS="""+",".join(map(str, atom_groups[0]))+""" LABEL=al
GROUP ATOMS="""+",".join(map(str, atom_groups[1]))+""" LABEL=hp
GROUP ATOMS="""+",".join(map(str, atom_groups[2]))+""" LABEL=ar
GROUP ATOMS="""+",".join(map(str, atom_groups[3]))+""" LABEL=ac
GROUP ATOMS="""+",".join(map(str, atom_groups[4]))+""" LABEL=do

GYRATION TYPE=RADIUS ATOMS=al LABEL=rg_al
GYRATION TYPE=RADIUS ATOMS=hp LABEL=rg_hp
GYRATION TYPE=RADIUS ATOMS=ar LABEL=rg_ar
GYRATION TYPE=RADIUS ATOMS=ac LABEL=rg_ac
GYRATION TYPE=RADIUS ATOMS=do LABEL=rg_do

GYRATION TYPE=TRACE ATOMS=al LABEL=tr_al
GYRATION TYPE=TRACE ATOMS=hp LABEL=tr_hp
GYRATION TYPE=TRACE ATOMS=ar LABEL=tr_ar
GYRATION TYPE=TRACE ATOMS=ac LABEL=tr_ac
GYRATION TYPE=TRACE ATOMS=do LABEL=tr_do

GYRATION TYPE=GTPC_1 ATOMS=al LABEL=gtpc1_al
GYRATION TYPE=GTPC_1 ATOMS=hp LABEL=gtpc1_hp
GYRATION TYPE=GTPC_1 ATOMS=ar LABEL=gtpc1_ar
GYRATION TYPE=GTPC_1 ATOMS=ac LABEL=gtpc1_ac
GYRATION TYPE=GTPC_1 ATOMS=do LABEL=gtpc1_do

GYRATION TYPE=GTPC_2 ATOMS=al LABEL=gtpc2_al
GYRATION TYPE=GTPC_2 ATOMS=hp LABEL=gtpc2_hp
GYRATION TYPE=GTPC_2 ATOMS=ar LABEL=gtpc2_ar
GYRATION TYPE=GTPC_2 ATOMS=ac LABEL=gtpc2_ac
GYRATION TYPE=GTPC_2 ATOMS=do LABEL=gtpc2_do

GYRATION TYPE=GTPC_3 ATOMS=al LABEL=gtpc3_al
GYRATION TYPE=GTPC_3 ATOMS=hp LABEL=gtpc3_hp
GYRATION TYPE=GTPC_3 ATOMS=ar LABEL=gtpc3_ar
GYRATION TYPE=GTPC_3 ATOMS=ac LABEL=gtpc3_ac
GYRATION TYPE=GTPC_3 ATOMS=do LABEL=gtpc3_do

GYRATION TYPE=ASPHERICITY ATOMS=al LABEL=asph_al
GYRATION TYPE=ASPHERICITY ATOMS=hp LABEL=asph_hp
GYRATION TYPE=ASPHERICITY ATOMS=ar LABEL=asph_ar
GYRATION TYPE=ASPHERICITY ATOMS=ac LABEL=asph_ac
GYRATION TYPE=ASPHERICITY ATOMS=do LABEL=asph_do

GYRATION TYPE=ACYLINDRICITY ATOMS=al LABEL=acyl_al
GYRATION TYPE=ACYLINDRICITY ATOMS=hp LABEL=acyl_hp
GYRATION TYPE=ACYLINDRICITY ATOMS=ar LABEL=acyl_ar
GYRATION TYPE=ACYLINDRICITY ATOMS=ac LABEL=acyl_ac
GYRATION TYPE=ACYLINDRICITY ATOMS=do LABEL=acyl_do

GYRATION TYPE=KAPPA2 ATOMS=al LABEL=K2_al
GYRATION TYPE=KAPPA2 ATOMS=hp LABEL=K2_hp
GYRATION TYPE=KAPPA2 ATOMS=ar LABEL=K2_ar
GYRATION TYPE=KAPPA2 ATOMS=ac LABEL=K2_ac
GYRATION TYPE=KAPPA2 ATOMS=do LABEL=K2_do

GYRATION TYPE=RGYR_3 ATOMS=al LABEL=g3_al
GYRATION TYPE=RGYR_3 ATOMS=hp LABEL=g3_hp
GYRATION TYPE=RGYR_3 ATOMS=ar LABEL=g3_ar
GYRATION TYPE=RGYR_3 ATOMS=ac LABEL=g3_ac
GYRATION TYPE=RGYR_3 ATOMS=do LABEL=g3_do

GYRATION TYPE=RGYR_2 ATOMS=al LABEL=g2_al
GYRATION TYPE=RGYR_2 ATOMS=hp LABEL=g2_hp
GYRATION TYPE=RGYR_2 ATOMS=ar LABEL=g2_ar
GYRATION TYPE=RGYR_2 ATOMS=ac LABEL=g2_ac
GYRATION TYPE=RGYR_2 ATOMS=do LABEL=g2_do

GYRATION TYPE=RGYR_1 ATOMS=al LABEL=g1_al
GYRATION TYPE=RGYR_1 ATOMS=hp LABEL=g1_hp
GYRATION TYPE=RGYR_1 ATOMS=ar LABEL=g1_ar
GYRATION TYPE=RGYR_1 ATOMS=ac LABEL=g1_ac
GYRATION TYPE=RGYR_1 ATOMS=do LABEL=g1_do
""")
    
    f.write("PRINT ARG=rg_al,tr_al,gtpc1_al,gtpc2_al,gtpc3_al,asph_al,acyl_al,K2_al,g3_al,g2_al,g1_al,\
rg_hp,tr_hp,gtpc1_hp,gtpc2_hp,gtpc3_hp,asph_hp,acyl_hp,K2_hp,g3_hp,g2_hp,g1_hp,\
rg_ar,tr_ar,gtpc1_ar,gtpc2_ar,gtpc3_ar,asph_ar,acyl_ar,K2_ar,g3_ar,g2_ar,g1_ar,\
rg_ac,tr_ac,gtpc1_ac,gtpc2_ac,gtpc3_ac,asph_ac,acyl_ac,K2_ac,g3_ac,g2_ac,g1_ac,\
rg_do,tr_do,gtpc1_do,gtpc2_do,gtpc3_do,asph_do,acyl_do,K2_do,g3_do,g2_do,g1_do STRIDE=1 FILE=shape_"+molname)
    f.close()
    
    return infname


def execute_plumed_driver(*args):
    ConsScoreTK_Statistics.run_commandline("plumed driver "+" ".join(list(args)), logname="shape.log",
                                           return_out=False, error_keywords=[])
    
    
def parse_plumed_output(outfname="shape"):
    
    with open(outfname, 'r') as f:
        for line in f:
            wordlist = line.split()
            if wordlist[0] == '0.000000':
                feature_vector = [float(w) for w in wordlist][1:]
                break
    return np.array(feature_vector)
    

def get_shape_vector(molfile, atom_groups, onlyshape=False):
    
    print("DEBUG: atom_groups=", atom_groups)
    
    molname = molfile.replace(".pdb", "")
    plumed_input = write_plumed_shape_input(molname, atom_groups)
    execute_plumed_driver("--mf_pdb", molfile, "--plumed", plumed_input)
    shape_vector = parse_plumed_output(outfname="shape_"+molname)
    
    names = ['rg_al', 'tr_al', 'gtpc1_al', 'gtpc2_al', 'gtpc3_al', 'asph_al', 'acyl_al', 'K2_al', 'g3_al', 'g2_al', 'g1_al', 'rg_hp', 'tr_hp',
             'gtpc1_hp', 'gtpc2_hp', 'gtpc3_hp', 'asph_hp', 'acyl_hp', 'K2_hp', 'g3_hp', 'g2_hp', 'g1_hp', 'rg_ar', 'tr_ar', 'gtpc1_ar',
             'gtpc2_ar', 'gtpc3_ar', 'asph_ar', 'acyl_ar', 'K2_ar', 'g3_ar', 'g2_ar', 'g1_ar', 'rg_ac', 'tr_ac', 'gtpc1_ac', 'gtpc2_ac',
             'gtpc3_ac', 'asph_ac', 'acyl_ac', 'K2_ac', 'g3_ac', 'g2_ac', 'g1_ac', 'rg_do', 'tr_do', 'gtpc1_do', 'gtpc2_do', 'gtpc3_do',
             'asph_do', 'acyl_do', 'K2_do', 'g3_do', 'g2_do', 'g1_do']
    for group, gname in zip(atom_groups, ['al', 'hp', 'ar', 'ac', 'do']):
        if len(group) <= 3:
            # null_indices = [i for i in range(shape_vector.shape[0]) if 'gtpc1_'+gname in names[i] or 'gtpc2_'+gname in names[i]
            #                 or 'gtpc3_'+gname in names[i] or 'g1_'+gname in names[i] or 'g2_'+gname in names[i] or 'g3_'+gname in names[i]]
            null_indices = [i for i in range(shape_vector.shape[0]) if '_'+gname in names[i]]
            for i in null_indices:
                shape_vector[i] = -1.0
    
    if onlyshape:
        return shape_vector[:11]
    else:
        return shape_vector
