from lib.global_fun import *

class Gallium():

    def __init__(self):
        pass

    @staticmethod
    def write_SQM_Eint_cuby4_script(fname, pdb, method="pm6", solvent="cosmo2",
                                    trunc_cutoff=0):
        """OUTDATED!"""
        lig_charge, rec_charge = MD_Analysis().read_charges_from_pdb(pdb)

        if method.startswith("pm"):
            Gallium.__write_PM67_Eint_cuby4_script(fname, lig_charge, rec_charge, method, solvent)
        elif method.startswith("dftb"):
            Gallium.__write_DFTB_Eint_cuby4_script(fname, lig_charge, rec_charge, method, solvent,
                                                   trunc_cutoff)

    def __write_DFTB_Eint_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6", solvent="cosmo2",
                                       trunc_cutoff=0):
        # TODO: complete this function's body
        pass

    def __write_PM67_Eint_cuby4_script(fname, lig_charge=0, rec_charge=0, method="pm6", solvent="cosmo2"):
        # TODO: fix PM7.
        if method == "pm6":
            corrections_comment = ""
        elif method == "pm7":
            corrections_comment = "#"

        out = """queue_submit: yes
queue_name: gq*
queue_jobname: %s_%s
queue_parallel: 1
cuby_threads: 1

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
        """ % tuple([method] + [solvent] * 18 + [lig_charge, rec_charge, method, solvent] + [corrections_comment]*7 + [solvent])

        with open(fname, 'w') as f:
            f.writelines(out)

    @staticmethod
    def write_SQM_COSMO_submit_script(solvent, workdir, structdir):
        out = """#!/bin/sh

export PATH=/home/rezac/cuby4:$PATH
WORKDIR=$(realpath $(dirname $0))
# Set the ABSOLUTE PATH to structure directory if you run calculations on a different computer
STRUCTURE_DIR=%s

for molname in $(ls ${WORKDIR})
do
[ -e ${WORKDIR}/${molname}/%s/RESULTS/LOG ] && [ $(grep "Interaction energy:" ${WORKDIR}/${molname}/%s/RESULTS/LOG | wc -l) -gt 0 ] && continue ;
[ ! -e ${WORKDIR}/${molname}/cosmo ] && continue ;
cd ${WORKDIR}/${molname}/cosmo
rm -rf *_ RESULTS/
complex=$(basename $(readlink input.pdb))
rm input.pdb
ln -s ${STRUCTURE_DIR}/${complex} input.pdb
complex=$(basename $(realpath input.pdb))
rm input.pdb
ln -s ${STRUCTURE_DIR}/${complex} input.pdb
cuby4 %s.yaml
cd ../../../

while [ $(qstat -u $USER | wc -l) -gt 1000 ]
do
sleep 10
done

done
        """ % (structdir, solvent, solvent, solvent)

        with open("%s/submit_%s.sh" % (workdir, solvent), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/submit_%s.sh" % (workdir, solvent))

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
queue_name: gq*
queue_jobname: GeomOpt_%s
queue_parallel: 1
cuby_threads: 1

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
        """ % tuple([molname] + [method, solvent_line] + [corrections_comment]*3 + [formal_charge, opt_quality] +
                    [corrections_comment]*5 + [atomic_charges])

        with open(fname, 'w') as f:
            f.writelines(out)

    @staticmethod
    def write_geom_opt_submit_script(workdir):
        workdir = os.path.abspath(workdir)
        out = """#!/bin/sh
# Launch this script using 'screen' command:
#   screen -S GeomOpt_pose_PM7 ./submit.sh

export PATH=/home/rezac/cuby4:$PATH
WORKDIR=$(realpath $(dirname $0))
# RE-creating sym-links deals with directories that were created on another computer and then coppied to Gallium.

for molname in $(cat ${WORKDIR}/unfinished_molnames.list)
do
[ -e ${WORKDIR}/${molname}/RESULTS/optgeom_charges.mol2 ] && continue ;
cd ${WORKDIR}/${molname}
rm input.mol2
ln -s ${WORKDIR}/molfiles/${molname}.mol2 input.mol2
cuby4 opt_charges.yaml;
cd ../../

while [ $(qstat -u $USER | wc -l) -gt 1000 ]
do
sleep 10
done

done
            """

        with open("%s/submit.sh" % (workdir), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/submit.sh" % (workdir))

        # Write script to delete redundant files and reduce the size of the WORKDIR before copying it back.
        out = """#!/bin/sh
# Launch this script to remove redundant files using 'screen' command:
#   screen -S GeomOpt_pose_PM7 ./clean.sh

WORKDIR=$(realpath $(dirname $0))

for molname in $(cat ${WORKDIR}/unfinished_molnames.list)
do
[ ! -e ${WORKDIR}/${molname}/RESULTS/optgeom_charges.mol2 ] && continue ;
cd ${WORKDIR}/${molname}
rm _finished _queue_run.sh _queue_submit.sh _scratch_link _sge_out.err _sge_out.out log
cd RESULTS
rm _queue_run.sh _queue_submit.sh _sge_out.err _sge_out.out history.xyz input.mol2 log opt_charges.yaml optimized.mol2
cd ${WORKDIR}
done
        """

        with open("%s/clean.sh" % (workdir), 'w') as f:
            f.writelines(out)
        run_commandline("chmod 777 %s/clean.sh" % (workdir))