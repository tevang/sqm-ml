from lib.global_fun import run_commandline

def write_gen_lig_params(fname, lig_charge):
    contents = """
#!/bin/bash

grep LIG input.pdb > 01-ligand_starting_geometry.pdb;
mkdir 01-ligand_amber_params/;
cd 01-ligand_amber_params;
/home/tevang/Programs/amber20_src_static/bin/antechamber -i ../01-ligand_starting_geometry.pdb -fi pdb -o ligand.mol2 -fo mol2 -c gas -dr n -nc %i; 
/home/tevang/Programs/amber20_src_static/bin/parmchk2 -a Y -s 2 -i ligand.mol2 -f mol2 -o LIG.frcmod;
cd ../;
cat > 01-ligand_leaprc << EOF
source leaprc.gaff2
loadamberparams ../01-ligand_amber_params/LIG.frcmod
LIG=loadmol2 ../01-ligand_amber_params/ligand.mol2
source leaprc.protein.ff19SB
loadamberparams /home/tevang/rezac/cuby4/interfaces/amber/data/amberff03_pm6/BCC.frcmod
loadamberparams /home/tevang/rezac/cuby4/interfaces/amber/data/amberff03_pm6/BNC.frcmod
loadamberparams /home/tevang/rezac/cuby4/interfaces/amber/data/amberff03_pm6/BCB.frcmod
loadamberprep /home/tevang/rezac/cuby4/interfaces/amber/data/amberff03_pm6/BCC.in
loadamberprep /home/tevang/rezac/cuby4/interfaces/amber/data/amberff03_pm6/BNC.in
loadamberprep /home/tevang/rezac/cuby4/interfaces/amber/data/amberff03_pm6/BCB.in
loadamberprep /home/tevang/rezac/cuby4/interfaces/amber/data/amberff03_pm6/BCG.in
source leaprc.water.tip3p
EOF
    """ % lig_charge
    with open(fname, 'w') as f:
        f.write(contents)

    run_commandline("chmod 777 " + fname, verbose=False)