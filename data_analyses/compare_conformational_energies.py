import pandas as pd

# LOAD OPLS2005 POTENTIAL ENERGIES
opls_df = pd.read_csv("%s/%s/%s_all_compounds-ligprep.renstereo_ion_tau-3000confs.OPLS2005_Energies.csv.gz" %
                      (SQMNN_ROOT_DIR, protein, protein))

# RENAME POSES
# NOTE: do not sort, maintain the order in the maegz
opls_df["pose"] = opls_df.groupby('s_m_title').cumcount() + 1
opls_df.rename(columns={"s_m_title": "structvar"}, inplace=True)
opls_df["molname"] = opls_df.apply(lambda r: r['structvar'] + "_pose%i" % r["pose"], axis=1)

# COMPUTE DELTAG FROM MINIMUM OPLS POTENTIAL ENERGY CONFORMER. TODO: test validity
opls_df["DeltaG_OPLS2005"] = opls_df.groupby("structvar")["r_mmod_Potential_Energy-OPLS-2005"].transform(lambda x: x-x.min())

# LOAD PM6/COSMO ENERGIES
pm6_df = pd.read_csv("%s/%s/%s_all_compounds-ligprep.renstereo_ion_tau-3000confs.PM6_COSMO.csv.gz" %
                     (SQMNN_ROOT_DIR, protein, protein))
pm6_df.rename(columns={"Energy:": "P6C_energy"}, inplace=True)
pm6_df["structvar"] = pm6_df["molname"].str \
        .extract("^(.*_stereo[0-9]+_ion[0-9]+_tau[0-9]+)_pose[0-9]+$") \
        .rename({0: 'structvar'}, axis=1)

# COMPUTE THE DELTAG FROM THE LOWEST ENERGY CONFORMER
pm6_df["DeltaG_PM6_COSMO"] = pm6_df.groupby("structvar")["P6C_energy"].transform(lambda x: x-x.min())

# MEASURE THE ABSOLUTE DIFFERENCE BETWEEN DELTAG PM6/COSMO aND DELTAG OPLS2005
all_energies_df = pd.merge(opls_df[["molname", "DeltaG_OPLS2005"]], pm6_df, on="molname")
all_energies_df["abs_DDeltaG"] = abs(all_energies_df["DeltaG_OPLS2005"]-all_energies_df["DeltaG_PM6_COSMO"])
all_energies_df = all_energies_df[all_energies_df["DeltaG_PM6_COSMO"]<6.0]   # keep only valid conformers
(all_energies_df["abs_DDeltaG"]).mean()
(all_energies_df["abs_DDeltaG"]).max()
(all_energies_df["abs_DDeltaG"]).min()

"""
* rename poses in opls_df
* compute DeltaG from minimum OPLS potential energy conformer.
* load PM6/COSMO energies to DataFrame and convert them to DeltaG.
* compute the average difference and stdev between PM6/COSMO DeltaG and OPLS DeltaG.
* set the maximum OPLS DeltaG for conformer retaining to be 12 kcal/mol + average difference.
"""