"""
This file contains setting for the normal QOS on Aurum.
"""

import uuid

from lib.global_fun import *
from lib.global_vars import CONSSCORTK_THIRDPARTY_DIR

class Aurum():

    def __init__(self):
        pass

    @staticmethod
    def write_SQM_Eint_cuby4_script(fname, pdb, method="pm6", solvent="cosmo2", trunc_cutoff=0,
                                    SQM_selection=None, amberhome=None, SQM_protein_charge=None):
        """Does not support SQM optimization of the whole binding pocket, only of the ligand."""

        lig_charge, rec_charge = MD_Analysis().read_charges_from_pdb(pdb)

        if method.startswith("pm") and not SQM_selection:
            Aurum.__write_PM67_scoring_cuby4_script(fname, lig_charge, rec_charge, method, solvent)
        if method.startswith("pm") and SQM_selection:
            Aurum.__write_PM67_opt_scoring_cuby4_script(fname, lig_charge, rec_charge, method, solvent,
                                                        SQM_selection=SQM_selection, amberhome=amberhome)
        elif method.startswith("dftb"):
            Aurum.__write_DFTB_Eint_cuby4_script(fname, lig_charge, rec_charge, method, solvent,
                                                 trunc_cutoff)

    def __write_DFTB_Eint_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6", solvent="cosmo2",
                                       trunc_cutoff=0):
        # TODO: complete this function's body
        pass

    def __write_PM67_scoring_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6", solvent="cosmo2"):

        if method == "pm6":
            corrections_comment = ""
        elif method == "pm7":
            corrections_comment = "#"

        out = """queue_submit: yes
queue_submit: no
queue_name: cpu
queue_jobname: %s_%s
queue_parallel: 1
cuby_threads: 1
queue_walltime_hrs: 1

job: multistep
steps:
  - %s_int

multistep_result_eval: |

  %s_Eint = steps['%s_int'].energy
  %s_a = steps['%s_int'].subsystems[:a].energy
  %s_b = steps['%s_int'].subsystems[:b].energy
  %s_ab = steps['%s_int'].subsystems[:ab].energy

  puts "%s_Eint:            #{'%%.2f' %% %s_Eint}"
  puts "%s_a:               #{'%%.2f' %% %s_a}"
  puts "%s_b:               #{'%%.2f' %% %s_b}"
  puts "%s_ab:              #{'%%.2f' %% %s_ab}"

shared_charge: &charge
  charge: %i

shared_charge_rec: &rec_charge
  charge: %i

shared_mopac: &mopac_setup
  job: interaction
  cuby_threads: 1
  interface: mopac
  existing_calc_dir: read_results
  method: %s
  solvent_model: %s
  mopac_mozyme: yes
%s  modifiers: dispersion3, h_bonds4, x_bond
%s  modifier_dispersion3:
%s    d3_hh_fix_version: 2
%s  modifier_h_bonds4:
%s    h_bonds4_skip_acceptor:
%s      - OS
%s    h_bonds4_pt_corr: 18
  print: energy_decomposition, subsystem_results, timing
  geometry: input.pdb
  mopac_setpi:
    - "%%atomtype(S:O2N1C1); %%atomtype(O:S)"
  mopac_setcharge:
    "%%atomtype(N:S1H1)": "-"
  mopac_keywords: LET NSPA=92

  molecule_a:
    selection: ":LIG"
    <<: *charge
    mopac_setpi:
      - "%%atomtype(S:O2N1C1); %%atomtype(O:S)"
    mopac_setcharge:
      "%%atomtype(N:S1H1)": "-"

  molecule_b:
    selection: "%%not(:LIG)"
    <<: *rec_charge
    mopac_setpi:
      - "%%atomtype(S:O2N1C1); %%atomtype(O:S)"
    mopac_setcharge:
      "%%atomtype(N:S1H1)": "-"

calculation_%s_int:
  <<: *mopac_setup
        """ % tuple([method] + [solvent] * 18 + [lig_charge, rec_charge, method, solvent] + [corrections_comment]*7 + [solvent])

        with open(fname, 'w') as f:
            f.writelines(out)


    def __write_PM67_opt_scoring_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6", solvent="cosmo2",
                                                  SQM_selection=":LIG", amberhome=os.environ.get('AMBERHOME')):

            SQM_selection = SQM_selection.replace("LIG", "UNK")  # cuby4 in this script renames the ligand to UNK

            if method == "pm6":
                corrections_comment = ""
            elif method == "pm7":
                corrections_comment = "#"

            out = """queue_submit: no
queue_name: cpu
queue_jobname: %s_%s_OptScore
queue_parallel: 1
cuby_threads: 1
queue_walltime_hrs: 2

#===============================================================================
# USER INPUT
#===============================================================================
#
# Initial complex geometry. MUST CONTAIN TER RECORDS AT LEAST AFTER RESIDUES WITH 'OXT' ATOMS (C-term caps)
shared_complex_geometry: &complex_geometry
  geometry: input.pdb

# Ligand charge
shared_ligand_charge: &ligand_charge
  charge: %i

# Receptor charge
shared_receptor_charge: &receptor_charge
  charge: %i

# Regions to be optimized in QM part
shared_flex_h_lig: &flex_h_lig
  optimize_region: "%s"


#===============================================================================
# END OF USER INPUT
#===============================================================================

#===============================================================================
# Steps of the workflow
#===============================================================================
job: multistep
steps:
  - 01a_save_ligand_geometry
  - 01b_save_receptor_geometry
  - 02_ligand_parameters
  - 03_ligand_pdb_file
  - 04_complex_pdb_file
  - 05_complex_pdb_file_renumber
  - 06_qmmm_optimization
  - 07_sqm_final_interaction

#===============================================================================
# Global Cuby options:
#===============================================================================
print: timing
job_cleanup: yes

#===============================================================================
# Common method setup
#===============================================================================

# Method used for all SQM calculations
shared_sqm_setup: &sqm_setup
  interface: mopac
  method: %s
%s  mopac_corrections: d3h4x
  mopac_setcharge:
    "%%atomtype(N:S1H1)": "-"
    "%%atomtype(N:S1*1)&%%within(2;%%atomtype(S:O2N1*1))": "-"
    "%%atomtype(N:C2)&%%within(1.5;%%atomtype(C:N1O1*1))&%%within(1.5;%%atomtype(C:H1*2))": "-"
  mopac_setpi:
    - "%%atomtype(S:O2N1C1); %%atomtype(O:S)"

# Solvent model for the SQM structure optimization and preparation (ONLY 'COSMO' IS SUPPORTED)
shared_sqm_solvent: &sqm_solvent
  solvent_model: cosmo

# Solvent model for the SQM calculation of final single-point energies
shared_sqm_solvent_singlepoint: &sqm_solvent_singlepoint
  solvent_model: %s

# Method for MM calculations, with the ligand parameters generated on the fly
shared_amber_setup: &amber_setup
  interface: amber
  amber_leaprc: 02-ligand_leaprc
  amber_amberhome: %s

#===============================================================================
# Code evaluating the results
#===============================================================================

multistep_result_eval: |

  results = Results.from_yaml_file("07-interaction_energy.yaml")
  Eint = results.energy
  a = results.subsystems[:a].energy
  a_eE = results.subsystems[:a].energy_components[:electronic_energy]
  a_dipole = results.subsystems[:a].multipoles[:dipole].value
  b = results.subsystems[:b].energy
  b_eE = results.subsystems[:b].energy_components[:electronic_energy]
  b_dipole = results.subsystems[:b].multipoles[:dipole].value
  ab = results.subsystems[:ab].energy
  ab_eE = results.subsystems[:ab].energy_components[:electronic_energy]
  ab_dipole = results.subsystems[:ab].multipoles[:dipole].value

  puts "electronic_ligandE_bound:  	#{'%%.2f' %% a_eE}"
  puts "electronic_proteinE_bound:  	#{'%%.2f' %% b_eE}"
  puts "electronic_complexE:  		#{'%%.2f' %% a_eE}"

  puts "dipole_ligand_bound:  		#{'%%s' %% a_dipole}"
  puts "dipole_protein_bound:  		#{'%%s' %% b_dipole}"
  puts "dipole_complex:  		#{'%%s' %% ab_dipole}"

  puts "Eint:            		#{'%%.2f' %% Eint}"
  puts "ligandE_bound:   		#{'%%.2f' %% a}"
  puts "proteinE_bound:  		#{'%%.2f' %% b}"
  puts "complexE:        		#{'%%.2f' %% ab}"

#===============================================================================
# Individual steps
#===============================================================================
calculation_01a_save_ligand_geometry:
  step_title: "01a - Save ligand geometry"
  job: geometry
  geometry_action: none
  <<: *complex_geometry
  selection: ":LIG"
  geometry_write_format: xyz
  geometry_write: 01-ligand_starting_geometry.xyz

calculation_01b_save_receptor_geometry:
  step_title: "01b - Save receptor geometry"
  job: geometry
  geometry_action: none
  <<: *complex_geometry
  selection: "%%not(:LIG)"
  geometry_write_format: pdb
  geometry_write: 01-receptor_starting_geometry.pdb

calculation_02_ligand_parameters:
  step_title: "02 - Generating AMBER parameters for ligand"
  skip_step_if_file_found: 02-ligand_leaprc
  job: ff_params_amber
  amber_amberhome: %s
  amberparm_dir: 02-ligand_amber_params
  amberparm_leaprc: 02-ligand_leaprc
  amberparm_leaprc_paths: relative
  geometry: 01-ligand_starting_geometry.xyz
  amberparm_ligand_ff: gaff2
  amberparm_protein_ff: ff19sb, cuby_caps, tip3p
  <<: *ligand_charge

calculation_03_ligand_pdb_file:
  step_title: "03 - preparing PDB file with ligand"
  job: geometry
  geometry: 02-ligand_amber_params/ligand.pdb
  geometry_update_coordinates: 01-ligand_starting_geometry.xyz
  geometry_action: none
  geometry_write_format: pdb
  pdb_extra_columns: yes
  geometry_write: 03-original_ligand.pdb

calculation_04_complex_pdb_file:
  step_title: "04 - preparing PDB file of protein--ligand complex"
  job: geometry
  geometry_action: cat
  geometry: 03-original_ligand.pdb
  geometry2: 01-receptor_starting_geometry.pdb
  geometry_write_format: pdb
  geometry_write: 04-protein--ligand_temp.pdb

calculation_05_complex_pdb_file_renumber:
  step_title: "05 - renumbering PDB file of protein--ligand complex"
  job: geometry
  geometry_action: pdb_renumber
  geometry_write_format: pdb
  geometry: 04-protein--ligand_temp.pdb
  geometry_write: 05-protein--ligand.pdb

calculation_06_qmmm_optimization:
  step_title: "06 - SQM/MM optimization of ligand"
  skip_step_if_file_found: 06-qmmm_optimized_complex.pdb
  job: optimize
  geometry: 05-protein--ligand.pdb
  optimize_print: steps_as_dots, final_energy
  interface: qmmm
  qmmm_core: ":UNK"
  qmmm_qmregion_file: ""
  <<: *flex_h_lig
  optimizer: lbfgs
  opt_quality: 0.5
  maxcycles: 800
  history_freq: 1
  history_file: 06-qmmm_optimization_history.xyz
  restart_file:  06-qmmm_optimized_complex.pdb
  calculation_qm:
    <<: *sqm_setup
    <<: *sqm_solvent
    <<: *ligand_charge

  calculation_mm:
    <<: *amber_setup
    solvent_model: igb7

calculation_07_sqm_final_interaction:
  step_title: "07 - Final SQM interaction energy calculations"
  skip_step_if_file_found: 07-interaction_energy.yaml
  job: interaction
  geometry: 06-qmmm_optimized_complex.pdb
  write_results_yaml: 07-interaction_energy.yaml
  mopac_mozyme: yes
  <<: *sqm_setup
  <<: *sqm_solvent_singlepoint
  molecule_a:
    selection: ":UNK"
    <<: *ligand_charge
  molecule_b:
    selection: "%%not(:UNK)"
    <<: *receptor_charge
            """ % (method, solvent, lig_charge, rec_charge, SQM_selection, method, corrections_comment,
                   solvent, amberhome, amberhome)

            with open(fname, 'w') as f:
                f.writelines(out)


    @staticmethod
    def write_SQM_COSMO_submit_script(solvent, workdir, structdir):
        """

        :param solvent:
        :param workdir:
        :param structdir: useless, just for compatibility with the other similar function
        :return:
        """
        signature = str(uuid.uuid4())
        out = """#!/bin/sh 
## Script for Aurum. Ean node provides 36 cores. I run a single job on a single node spread all over it's cores with GNU's 'parallel' script.
#SBATCH --job-name=score
#SBATCH -N 1 -n 1 -c 36
#SBATCH --mem-per-cpu=2600M
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH --output=SQM-%%J.out 
#SBATCH --error=SQM-%%J.err   
#SBATCH --time=12:00:00

export PATH=/home1/rezac/cuby4:/opt/uochb/soft/spack/20201015/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/ruby-2.7.1-zrckiakitfdlxo32rdprxkk2xj6clp4c/bin/:$PATH

# tar and copy all the input files to /dev/shm
cd %s ; 
tar cfh /dev/shm/command_${i}_input_folders.tar -T command_${i}_input_folders.list ; 
cp command_${i}.list /dev/shm/ ;
cd /dev/shm ;
tar xf command_${i}_input_folders.tar ;
rm command_${i}_input_folders.tar ;

# launch the calculations
timeout 11.9h srun -c 1 -n 1 --qos=12 %s/parallel --tmpdir /dev/shm -j36 < command_${i}.list

# At the end copy all remaining finished jobs and clean the /dev/sh dir
cd /dev/shm;
for d in $(find . -wholename "*/*/_finished_SQM" -exec dirname {} \;) ;
do
rm -rf ${d}/_finished_SQM ${d}/cosmo*.yaml ${d}/input.pdb ${d}/0*;
echo $d ;
done > finished_moldirs.%s.list ;
tar cfh finished_moldirs.%s.tar -T finished_moldirs.%s.list ;
mv finished_moldirs.%s.tar %s/ ;
rm -rf *

# To submit the job:
# sbatch --export=ALL,i=23 submit.sh
        """ % (workdir, CONSSCORTK_THIRDPARTY_DIR, signature, signature, signature, signature, workdir)

        with open("%s/submit_%s.sh" % (workdir, solvent), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/submit_%s.sh" % (workdir, solvent))

        untar = """#!/bin/bash
# Simple script to untar sequentially all the result archives upon completion of the jobs on Aurum nodes

for f in $(ls *.tar)
do
tar xf $f
rm $f
done
    """
        with open("%s/untar_finished_moldirs.sh" % (workdir), 'w') as f:
            f.writelines(untar)
        run_commandline("chmod 777 %s/untar_finished_moldirs.sh" % (workdir))

    @staticmethod
    def write_geom_opt_cuby4_script(fname, molname, formal_charge, method="pm7", solvent='cosmo', opt_quality=0.1,
                                    atomic_charges="mulliken"):
        """

        :param fname:
        :param mol2:
        :param net_charge: net charge of the molecule
        :return:
        """
        if solvent == "cosmo":
            solvent_line = "solvent_model: cosmo"
        elif solvent == "gas-phase":
            solvent_line = "#solvent_model: cosmo"

        if method == "pm6":
            corrections_comment = ""
        elif method == "pm7":
            corrections_comment = "#"

        out = """queue_submit: yes
queue_submit: no
queue_name: cpu
queue_jobname: GeomOpt_%s
queue_parallel: 1
cuby_threads: 1
queue_walltime_hrs: 1

job: multistep
steps: optimize, charges

geometry: input.mol2

calculation_common: 
  interface: mopac
  method: %s
  %s
%s  modifiers: dispersion3
%s  modifier_dispersion3:
%s    d3_hh_fix_version: 2
  # Set the charge from commandline if it is not zero, e.g.:
  # --*:charge -1
  charge: %i
  mopac_keywords: NSPA=92

calculation_optimize:
  job: optimize
  optimizer: lbfgs
  geometry: parent_block
  restart_file: optimized.mol2
  optimize_print: steps_as_dots, statistics
  opt_quality: %f
%s  modifiers: h_bonds4, x_bond
%s  modifier_h_bonds4:
%s    h_bonds4_skip_acceptor:
%s      - OS
%s    h_bonds4_pt_corr: 18

calculation_charges:
  geometry: optimized.mol2
  job: atomic_charges
  atomic_charges: %s
  atomic_charges_write_format: mol2
  atomic_charges_write: optgeom_charges.mol2
        """ % tuple([molname] + [method, solvent_line] + [corrections_comment]*3 + [formal_charge, opt_quality] +
                    [corrections_comment]*5 + [atomic_charges])

        with open(fname, 'w') as f:
            f.writelines(out)    \

    @staticmethod
    def write_vibrational_analysis_cuby4_script(fname, molname, formal_charge, method="pm7", solvent='gas-phase', opt_quality=0.1,
                                    atomic_charges="mulliken"):
        """
        You must use the same solvent model for both geometry optimization and Hessian matrix calculation.

        :param fname:
        :param molname:
        :param formal_charge: net charge of the molecule
        :param method:
        :param solvent: recommended for vibrational analysis is 'gas-phase' as COSMO introduces numerical inaccuracies
                        (the potential energy is not continuous and thus not differentiable in every point).
        :param opt_quality:
        :param atomic_charges:
        :return:
        """
        if solvent == "cosmo":
            solvent_line = "solvent_model: cosmo"
        elif solvent == "gas-phase":
            solvent_line = "#solvent_model: cosmo"

        if method == "pm6":
            corrections_comment = ""
        elif method == "pm7":
            corrections_comment = "#"

        out = """queue_submit: yes
queue_submit: no
queue_name: cpu
queue_jobname: Vibration_%s
queue_parallel: 1
cuby_threads: 1
queue_walltime_hrs: 1

job: multistep
steps: optimize, vibration

geometry: input.mol2

calculation_common: 
  method: %s
  %s
%s  modifiers: dispersion3
%s  modifier_dispersion3:
%s    d3_hh_fix_version: 2
  # Set the charge from commandline if it is not zero, e.g.:
  # --*:charge -1
  charge: %i
  mopac_keywords: NSPA=92

calculation_optimize:
  job: optimize
  interface: mopac
  optimizer: lbfgs
  geometry: parent_block
  restart_file: optimized.mol2
  optimize_print: steps_as_dots, statistics
  opt_quality: %f
%s  modifiers: h_bonds4, x_bond
%s  modifier_h_bonds4:
%s    h_bonds4_skip_acceptor:
%s      - OS
%s    h_bonds4_pt_corr: 18

calculation_vibration:
  geometry: optimized.mol2
  job: vibrational_analysis
  interface: numerical_hessian
  geometry: optimized.xyz # Created in teh first step

  calculation: # Calculation of gradients used to construct Hessian
    interface: mopac

        """ % tuple([molname] + [method, solvent_line] + [corrections_comment]*3 + [formal_charge, opt_quality] +
                    [corrections_comment]*5 + [atomic_charges])

        with open(fname, 'w') as f:
            f.writelines(out)

    @staticmethod
    def write_geom_opt_submit_script(workdir):
        workdir = os.path.abspath(workdir)
        signature = str(uuid.uuid4())
        out = """#!/bin/sh 
## Script for Aurum. Ean node provides 36 cores. I run a single job on a single node spread all over it's cores with GNU's 'parallel' script.
#SBATCH --job-name=geomopt
#SBATCH -N 1 -n 1 -c 36
#SBATCH --mem-per-cpu=2600M
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH --output=SQM-%%J.out 
#SBATCH --error=SQM-%%J.err   
#SBATCH --time=12:00:00

export PATH=/home1/rezac/cuby4:/opt/uochb/soft/spack/20201015/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/ruby-2.7.1-zrckiakitfdlxo32rdprxkk2xj6clp4c/bin/:$PATH

# tar and copy all the input files to /dev/shm
cd %s ; 
tar cfh /dev/shm/command_${i}_input_folders.tar -T command_${i}_input_folders.list ; 
cp command_${i}.list /dev/shm/ ;
cd /dev/shm ;
tar xf command_${i}_input_folders.tar ;
rm command_${i}_input_folders.tar ;

# Launch the calculations
timeout 11.9h srun --qos=normal %s/parallel --tmpdir /dev/shm -j36 < command_${i}.list

sleep 10 

# At the end copy all remaining finished jobs and clean the /dev/sh dir
echo "Copying all remaining finished jobs and cleaning the /dev/sh dir."
cd /dev/shm;
for d in $(find . -wholename "*/optgeom_charges.mol2" -exec dirname {} \;) ;
do
echo "Tarring finished job files from folder $d"
rm ${d}/_finished_SQM ${d}/*.yaml ${d}/input.mol2 ;
echo $d >> final_finished_moldirs.${i}.list;
done;
for d in $(find . -wholename "*/_finished_SQM" -exec dirname {} \;) ;
do
echo "Tarring finished job files from folder $d"
rm ${d}/_finished_SQM ${d}/*.yaml ${d}/input.mol2 ;
echo $d >> final_finished_moldirs.${i}.list;
done;
tar cfh final_finished_moldirs.${i}.tar -T final_finished_moldirs.${i}.list ;
mv final_finished_moldirs.${i}.tar %s/ ;
cat final_finished_moldirs.${i}.list | xargs -i sh -c 'rm -rf {}' ;
echo "The following moldirs have been left in /dev/shm before deleting everything:"
ls -lt *
rm -rf *

# To submit the job:
# sbatch --export=ALL,i=23 submit.sh
                """ % (workdir, CONSSCORTK_THIRDPARTY_DIR, workdir)

        with open("%s/submit.sh" % (workdir), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/submit.sh" % (workdir))

        untar = """#!/bin/bash
# Simple script to untar sequentially all the result archives upon completion of the jobs on Aurum nodes

for f in $(ls *.tar)
do
tar xf $f
rm $f
done
            """
        with open("%s/untar_finished_moldirs.sh" % (workdir), 'w') as f:
            f.writelines(untar)
        run_commandline("chmod 777 %s/untar_finished_moldirs.sh" % (workdir))

    @staticmethod
    def write_vibrations_analysis_submit_script(workdir):
        workdir = os.path.abspath(workdir)
        signature = str(uuid.uuid4())
        out = """#!/bin/sh 
## Script for Aurum. Ean node provides 36 cores. I run a single job on a single node spread all over it's cores with GNU's 'parallel' script.
#SBATCH --job-name=vibration
#SBATCH -N 1 -n 1 -c 36
#SBATCH --mem-per-cpu=2600M
#SBATCH -p cpu
#SBATCH --qos=normal
#SBATCH --output=SQM-%%J.out 
#SBATCH --error=SQM-%%J.err   
#SBATCH --time=12:00:00

export PATH=/home1/rezac/cuby4:/opt/uochb/soft/spack/20201015/opt/spack/linux-centos8-skylake_avx512/gcc-8.3.1/ruby-2.7.1-zrckiakitfdlxo32rdprxkk2xj6clp4c/bin/:$PATH

# tar and copy all the input files to /dev/shm
cd %s ; 
tar cfh /dev/shm/command_${i}_input_folders.tar -T command_${i}_input_folders.list ; 
cp command_${i}.list /dev/shm/ ;
cd /dev/shm ;
tar xf command_${i}_input_folders.tar ;
rm command_${i}_input_folders.tar ;

# Launch the calculations
timeout 11.9h srun --qos=normal %s/parallel --tmpdir /dev/shm -j36 < command_${i}.list

sleep 10 

# At the end copy all remaining finished jobs and clean the /dev/sh dir
echo "Copying all remaining finished jobs and cleaning the /dev/sh dir."
cd /dev/shm;
for d in $(find . -wholename "*/optgeom_charges.mol2" -exec dirname {} \;) ;
do
echo "Tarring finished job files from folder $d"
rm ${d}/_finished_SQM ${d}/*.yaml ${d}/input.mol2 ;
echo $d >> final_finished_moldirs.${i}.list;
done;
for d in $(find . -wholename "*/_finished_SQM" -exec dirname {} \;) ;
do
echo "Tarring finished job files from folder $d"
rm ${d}/_finished_SQM ${d}/*.yaml ${d}/input.mol2 ;
echo $d >> final_finished_moldirs.${i}.list;
done;
tar cfh final_finished_moldirs.${i}.tar -T final_finished_moldirs.${i}.list ;
mv final_finished_moldirs.${i}.tar %s/ ;
cat final_finished_moldirs.${i}.list | xargs -i sh -c 'rm -rf {}' ;
echo "The following moldirs have been left in /dev/shm before deleting everything:"
ls -lt *
rm -rf *

# To submit the job:
# sbatch --export=ALL,i=23 submit.sh
                """ % (workdir, CONSSCORTK_THIRDPARTY_DIR, workdir)

        with open("%s/submit.sh" % (workdir), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/submit.sh" % (workdir))

        untar = """#!/bin/bash
# Simple script to untar sequentially all the result archives upon completion of the jobs on Aurum nodes

for f in $(ls *.tar)
do
tar xf $f
rm $f
done
            """
        with open("%s/untar_finished_moldirs.sh" % (workdir), 'w') as f:
            f.writelines(untar)
        run_commandline("chmod 777 %s/untar_finished_moldirs.sh" % (workdir))
