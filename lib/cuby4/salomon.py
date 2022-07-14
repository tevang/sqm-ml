import os.path

from lib.global_fun import *
from lib.global_vars import CONSSCORTK_THIRDPARTY_DIR
from lib.MD_Analysis_functions import MD_Analysis
from lib.cuby4.pdb_header_functions import get_sqm_region_properties_from_header
from lib.cuby4.write_shell_scripts import write_gen_lig_params

class Salomon():

    def __init__(self):
        pass

    @staticmethod
    def write_SQM_Eint_cuby4_script(fname, pdb, method="pm6", solvent="cosmo2",
                                    trunc_cutoff=0,
                                    SQM_selection=None, amberhome=None):

        lig_charge, rec_charge = MD_Analysis().read_charges_from_pdb(pdb)

        if method.startswith("pm") and not SQM_selection:
            Salomon.__write_PM67_scoring_cuby4_script(fname, lig_charge, rec_charge, method, solvent)
        if method.startswith("pm") and SQM_selection == ':LIG':
            Salomon.__write_PM67_lig_opt_scoring_cuby4_script(fname, lig_charge, rec_charge, method, solvent,
                                                              SQM_selection=SQM_selection, amberhome=amberhome)
        if method.startswith("pm") and SQM_selection.startswith('residues within '):
            selected_cutoff = int(SQM_selection.split()[2])
            for cutoff, charge, resid_selection in get_sqm_region_properties_from_header(pdb):
                if cutoff == selected_cutoff:
                    break
            Salomon.__write_PM67_protlig_opt_scoring_cuby4_script(fname, lig_charge, rec_charge,
                                                                  method, solvent,
                                                                  sqm_region_selection=resid_selection,
                                                                  sqm_region_charge=charge,
                                                                  amberhome=amberhome)
        elif method.startswith("dftb"):
            Salomon.__write_DFTB_Eint_cuby4_script(fname, lig_charge, rec_charge, method, solvent,
                                                 trunc_cutoff)

    def __write_DFTB_Eint_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6", solvent="cosmo2",
                                       trunc_cutoff=0):
        out = """queue_submit: yes
queue_jobname: dftb_large
queue_qsub_options: -A OPEN-23-26
queue_name: qprod
queue_walltime_hrs: 24
queue_extra_commands: "module load mkl"
queue_parallel: 24

job: multistep
steps:
#  - fragmentation
  - dftb3d3h5_int

print: energy_decomposition, subsystem_results, timing

multistep_result_eval: |
  dftb3d3h5_eint = steps['dftb3d3h5_int'].energy
  dftb3d3h5_a = steps['dftb3d3h5_int'].subsystems[:a].energy
  dftb3d3h5_b = steps['dftb3d3h5_int'].subsystems[:b].energy
  dftb3d3h5_ab = steps['dftb3d3h5_int'].subsystems[:ab].energy
  puts "dftb3d3h5_eint:            #{'%%.2f' %% dftb3d3h5_eint}"
  puts "dftb3d3h5_a:               #{'%%.2f' %% dftb3d3h5_a}"
  puts "dftb3d3h5_b:               #{'%%.2f' %% dftb3d3h5_b}"
  puts "dftb3d3h5_ab:              #{'%%.2f' %% dftb3d3h5_ab}"

shared_charge: &charge
  charge: %i

shared_charge_rec: &rec_charge
  charge: %i

shared_mopac: &frag_setup
  # TRUNCATE THE PROTEIN
  job: energy
  interface: qmmm
  geometry: input.pdb

  qmmm_auto_fragmentation: peptide_backbone

  qmmm_core: "%%within(8;:LIG)"
  qmmm_peptide_frag_add_pro: yes

  calculation_qm:
    interface: void
  calculation_mm:
    interface: void

  qmmm_qmregion_file: "input.pdb"
  qmmm_geometry_only: yes
  
shared_mopac: &mopac_setup
  job: interaction
  cuby_threads: 1
  existing_calc_dir: read_results
  
  # DFTB3-D3H5 setup
  interface: dftb
  method: scc-dftb3
  dftb_slko_set: 3ob-3-1
  dftb_e_temp: 300
  dftb_h5: yes
  dftb_d3: yes
  dftb_d3_hhrep: yes
  d3_damping: :zero
  d3_sr6: 1.25
  d3_alpha6: 29.61
  d3_s8: 0.49
  parallel: 24
  # Requires latest executable
  dftbplus_exe: /home/tevang/rezac/bin/DFTB+github/dftbplus/dftb+

  print: energy_decomposition, subsystem_results, timing
  geometry: input.pdb

  molecule_a:
    selection: ":LIG"
    <<: *charge

  molecule_b:
    selection: "%%not(:LIG)"
    <<: *rec_charge

#calculation_fragmentation:
#  <<: *frag_setup

calculation_dftb3d3h5_int:
  <<: *mopac_setup
        """ % (lig_charge, rec_charge)

        with open(fname, 'w') as f:
            f.writelines(out)

    def __write_PM67_scoring_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6", solvent="cosmo2"):

        # On Salomon we don't need any type of parallelization, since we have linear scaling in PM6 (mozyme)
        # simply it does not help. Secondly you're going to use your own, custom queue submission file. Cuby4 will
        # anyways be called by the node computer by that.
        if method == "pm6":
            corrections_comment = ""
        elif method == "pm7":
            corrections_comment = "#"

        out = """job: multistep
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
  mopac_mozyme: yes
  solvent_model: %s
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
        """ % tuple([solvent] * 17 + [lig_charge, rec_charge, method, solvent] + [corrections_comment]*7 + [solvent])

        with open(fname, 'w') as f:
            f.writelines(out)

    def __write_PM67_lig_opt_scoring_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6", solvent="cosmo2",
                                                  SQM_selection=":LIG", amberhome=os.environ.get('AMBERHOME')):

        write_gen_lig_params(os.path.dirname(fname) + '/gen_lig_param.sh', lig_charge)

        if method == "pm6":
            corrections_comment = ""
        elif method == "pm7":
            corrections_comment = "#"

        out = """
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
  - 01_shell_commands
  - 02_qmmm_optimization
  - 03_sqm_final_interaction

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
  amber_leaprc: 01-ligand_leaprc
  amber_amberhome: %s

#===============================================================================
# Code evaluating the results
#===============================================================================

multistep_result_eval: |
  
  results = Results.from_yaml_file("03-interaction_energy.yaml")
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
calculation_01_shell_commands:
  step_title: "01 - Generating AMBER parameters for ligand"
  skip_step_if_file_found: 01-ligand_leaprc
  job: shell_script
  shell_commands: "./gen_lig_param.sh"

calculation_02_qmmm_optimization:
  step_title: "02 - SQM/MM optimization of ligand"
  skip_step_if_file_found: 02-qmmm_optimized_complex.pdb
  job: optimize
  <<: *complex_geometry
  optimize_print: steps_as_dots, final_energy
  interface: qmmm
  qmmm_core: ":LIG"
  qmmm_qmregion_file: ""
  <<: *flex_h_lig
  optimizer: lbfgs
  opt_quality: 0.1
  maxcycles: 800
  history_freq: 1
  history_file: 02-qmmm_optimization_history.xyz
  restart_file:  02-qmmm_optimized_complex.pdb
  calculation_qm:
    <<: *sqm_setup
    <<: *sqm_solvent
    <<: *ligand_charge

  calculation_mm:
    <<: *amber_setup
    solvent_model: igb7

calculation_03_sqm_final_interaction:
  step_title: "03 - Final SQM interaction energy calculations"
  skip_step_if_file_found: 03-interaction_energy.yaml
  job: interaction
  geometry: 02-qmmm_optimized_complex.pdb
  write_results_yaml: 03-interaction_energy.yaml
  mopac_mozyme: yes
  <<: *sqm_setup
  <<: *sqm_solvent_singlepoint
  molecule_a:
    selection: ":LIG"
    <<: *ligand_charge
  molecule_b:
    selection: "%%not(:LIG)"
    <<: *receptor_charge
                """ % (lig_charge, rec_charge, SQM_selection, method, corrections_comment,
                       solvent, amberhome)

        with open(fname, 'w') as f:
            f.writelines(out)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def __write_PM67_protlig_opt_scoring_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6",
                                                      solvent="cosmo2",
                                                      amberhome=os.environ.get('AMBERHOME'),
                                                      sqm_region_selection="",
                                                      sqm_region_charge=0):

        write_gen_lig_params(os.path.dirname(fname) + '/gen_lig_param.sh', lig_charge)

        if method == "pm6":
            corrections_comment = ""
        elif method == "pm7":
            corrections_comment = "#"

        out = """
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

# QM region charge
shared_pocket_charge: &pocket_charge
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
  - 01_shell_commands
  - 02_qmmm_optimization
  - 03_sqm_final_interaction

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
  amber_leaprc: 01-ligand_leaprc
  amber_amberhome: %s

#===============================================================================
# Code evaluating the results
#===============================================================================

multistep_result_eval: |
  
  results = Results.from_yaml_file("03-interaction_energy.yaml")
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
calculation_01_shell_commands:
  step_title: "01 - Generating AMBER parameters for ligand"
  skip_step_if_file_found: 01-ligand_leaprc
  job: shell_script
  shell_commands: "./gen_lig_param.sh"

calculation_02_qmmm_optimization:
  step_title: "02 - SQM/MM optimization of ligand"
  skip_step_if_file_found: 02-qmmm_optimized_complex.pdb
  job: optimize
  <<: *complex_geometry
  optimize_print: steps_as_dots, final_energy
  interface: qmmm
  qmmm_auto_fragmentation: peptide_backbone
  qmmm_auto_run: yes 
  qmmm_core: "%s"
  qmmm_qmregion_file: ""
  <<: *flex_h_lig
  optimizer: lbfgs
  opt_quality: 0.5
  maxcycles: 800
  history_freq: 1
  history_file: 02-qmmm_optimization_history.xyz
  restart_file:  02-qmmm_optimized_complex.pdb
  calculation_qm:
    mopac_mozyme: yes
    <<: *sqm_setup
    <<: *sqm_solvent
    <<: *pocket_charge

  calculation_mm:
    <<: *amber_setup
    solvent_model: igb7

calculation_03_sqm_final_interaction:
  step_title: "03 - Final SQM interaction energy calculations"
  skip_step_if_file_found: 03-interaction_energy.yaml
  job: interaction
  geometry: 02-qmmm_optimized_complex.pdb
  write_results_yaml: 03-interaction_energy.yaml
  mopac_mozyme: yes
  <<: *sqm_setup
  <<: *sqm_solvent_singlepoint
  molecule_a:
    selection: ":LIG"
    <<: *ligand_charge
  molecule_b:
    selection: "%%not(:LIG)"
    <<: *receptor_charge
                """ % (lig_charge, rec_charge, sqm_region_charge, sqm_region_selection, method,
                       corrections_comment, solvent, amberhome, sqm_region_selection)

        with open(fname, 'w') as f:
            f.writelines(out)

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @staticmethod
    def write_SQM_COSMO_submit_script(solvent, workdir, structdir):

        out = """#!/bin/bash
#PBS -A OPEN-23-26
#PBS -q qprod
#PBS -N SCORE
#PBS -l select=1:ncpus=128,walltime=48:00:00

# for cuby4
source /home/tevang/cubyconfig.sh

# the index of the numtasks file is give as input argument
[ -z "$PARALLEL_SEQ" ] && { ml parallel ; exec parallel -a $PBS_O_WORKDIR/numtask_files/numtasks_$findex $0 ; }

# change to scratch directory
SCR=/ramdisk/$PBS_JOBID/$PARALLEL_SEQ   ; # for production runs (you cannot see the files)
#SCR=/scratch/work/user/$USER/$PBS_JOBID/$PARALLEL_SEQ  ; # for troubleshooting (you can see the files)
mkdir -p $SCR ; cd $SCR || exit

# get individual task from tasklist with index from PBS JOB ARRAY and index form Parallel
IDX=$(sed -n "${PARALLEL_SEQ}p" $PBS_O_WORKDIR/numtask_files/numtasks_${findex})
TASK=$(sed -n "${IDX}p" $PBS_O_WORKDIR/tasklist)
[ -z "$TASK" ] && exit

# copy input file and executable to scratch
cp $TASK/%s.yaml .   # I always put cal.yaml files to a common source directory. Then do sed for charge: XXX to ligand charge, charge: YYY for receptor charge.
cp $TASK/input.pdb .      # It can be also cp $PBS_O_WORKDIR/$TASK/input.pdb $PBS_O_WORKDIR/$TASK/cosmo2.yaml .   If you prepare you yaml file in advance, use this step.
cp $TASK/gen_lig_param.sh .      

# execute the calculation
echo "Launching job $IDX: $TASK"
cuby4 %s.yaml > LOG 
if [ $(grep "Interaction energy:" LOG | wc -l) -eq 1 ]
then
rm -rf 0[1345678]* job* 02-qmmm_optimization_history.xyz
fi

# copy output file to submit directory
cp -r * $TASK/
echo $TASK >> $PBS_O_WORKDIR/finished_tasklist
exit
        """ % (solvent, solvent)

        with open("%s/submit_%s.sh" % (workdir, solvent), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/submit_%s.sh" % (workdir, solvent))

    @staticmethod
    def write_SQM_COSMO_local_submit_script(scorefun, workdir):
        out = """#!/bin/sh

CPUS=14
export PATH=/home/tevang/rezac/cuby4:$PATH

for mol in $(ls %s)
do
[ ! -e "%s/$mol/%s/input.pdb" ] && continue ;
echo -n "cd %s/$mol/%s/ ; " 
echo -n "cuby4 %s.yaml >& LOG ; "
echo -n "cd ../../../ ; "
echo ""
done > commands.list

%s/parallel -j $CPUS < commands.list

        """ % (workdir, workdir, scorefun, workdir, scorefun, scorefun, CONSSCORTK_THIRDPARTY_DIR)

        with open("%s/submit_%s_local.sh" % (workdir, scorefun), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/submit_%s_local.sh" % (workdir, scorefun))

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

        out = """job: multistep
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
  mopac_keywords: LET NSPA=92

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
        """ % tuple([method, solvent_line] + [corrections_comment]*3 + [formal_charge, opt_quality] + [corrections_comment]*5 + [atomic_charges])

        with open(fname, 'w') as f:
            f.writelines(out)

    @staticmethod
    def write_geom_opt_submit_script(workdir):
        workdir = os.path.abspath(workdir)
        out = """#!/bin/bash
#PBS -A OPEN-23-26
#PBS -q qprod
#PBS -N GEOMOPt
#PBS -l select=1:ncpus=128,walltime=48:00:00

# for cuby4
source /home/tevang/rezac/cubyconfig.sh

# the index of the numtasks file is give as input argument
[ -z "$PARALLEL_SEQ" ] && { ml parallel ; exec parallel -a $PBS_O_WORKDIR/numtask_files/numtasks_$findex $0 ; }

# change to scratch directory
SCR=/ramdisk/$PBS_JOBID/$PARALLEL_SEQ   ; # for production runs (you cannot see the files)
#SCR=/scratch/work/user/$USER/$PBS_JOBID/$PARALLEL_SEQ  ; # for troubleshooting (you can see the files)
mkdir -p $SCR ; cd $SCR || exit

# get individual task from tasklist with index from PBS JOB ARRAY and index form Parallel
IDX=$(sed -n "${PARALLEL_SEQ}p" $PBS_O_WORKDIR/numtask_files/numtasks_${findex})
TASK=$(sed -n "${IDX}p" $PBS_O_WORKDIR/tasklist)
[ -z "$TASK" ] && exit

# copy input file and executable to scratch
cp $TASK/opt_charges.yaml .   # I always put cal.yaml files to a common source directory. Then do sed for charge: XXX to ligand charge, charge: YYY for receptor charge.
molname=$(basename $TASK)
rm $TASK/input.mol2
ln -s ${TASK}/../molfiles/${molname}.mol2 $TASK/input.mol2 ; # to fix broken symlink created on another machine
cp $TASK/input.mol2 .      # It can be also cp $PBS_O_WORKDIR/$TASK/input.pdb $PBS_O_WORKDIR/$TASK/cosmo2.yaml .   If you prepare you yaml file in advance, use this step.

# execute the calculation
echo "Launching job $IDX: $TASK"
cuby4 opt_charges.yaml > LOG 

# if job succeeded, copy only the output file to submit directory, otherwise the whole folder
mkdir $TASK/RESULTS
if [ -e ./optgeom_charges.mol2 ]
then
cp -r ./optgeom_charges.mol2 $TASK/RESULTS
else
cp -r * $TASK/RESULTS
fi
echo $TASK >> $PBS_O_WORKDIR/finished_tasklist
exit
                """

        with open("%s/submit.sh" % (workdir), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/submit.sh" % (workdir))