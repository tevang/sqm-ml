import pandas as pd
from commons.EXEC_caching import EXEC_caching_decorator
import logging

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Loading PM6/COSMO molecular free energies of all compounds in the unbound state, "
                        "and computing the minimum molecular energy for the ligand deformation energy calculation.",
                        "_min_ligandE_free.csv.gz")
def EXEC_compute_min_structvar_ligandE_free(Settings):

    return pd.read_csv("%s/%s/%s_all_compounds-ligprep.renstereo_ion_tau-3000confs.PM6_COSMO.csv.gz" %
                       (Settings.HYPER_SQM_ML_ROOT_DIR, Settings.HYPER_PROTEIN, Settings.HYPER_PROTEIN)) \
        .rename(columns={"Energy:": "P6C_energy"}) \
        .assign(molname=lambda df: df["molname"].str.lower()) \
        .assign(structvar=lambda df: df["molname"].str \
                .extract("^(.*_stereo[0-9]+_ion[0-9]+_tau[0-9]+)_pose[0-9]+$") \
                .rename({0: 'structvar'}, axis=1)["structvar"]) \
        .pipe(lambda df: df.groupby("structvar", as_index=False).apply(min) \
              .rename({"P6C_energy": "P6C_min_ligandE_free", "molname": "best_molname"}, axis=1)) \
        .assign(P6C2_min_ligandE_free=lambda df: df["P6C_min_ligandE_free"]) \
        .assign(P7C_min_ligandE_free=lambda df: df["P6C_min_ligandE_free"])
# NOTE: I copied P6C_min_ligandE_free to P6C2_min_ligandE_free and P7C_min_ligandE_free because I cannot afford
# NOTE: to run geometry optimizations 2x more times.


@EXEC_caching_decorator(lg, "Loading PM6/COSMO molecular free energies of all compounds in the unbound state, "
                        "and computing the minimum molecular energy for the ligand deformation energy calculation.",
                        "_sqm_clean_glide_mlEb.csv.gz")
def EXEC_compute_min_ligandE_bound(scores_df, Settings):

    def _merge_min_ligandE_bound(scores_df, scoring_function):

        return pd.merge(scores_df,
                        scores_df.groupby("structvar", as_index=False) \
                        .agg({scoring_function+"_ligandE_bound": min}) \
                        .rename(columns={scoring_function+"_ligandE_bound": scoring_function+"_min_ligandE_bound"}),
                        on="structvar")

    for SF in [c.replace("_Eint", "") for c in scores_df.columns if c.endswith("_Eint")]:
        scores_df = _merge_min_ligandE_bound(scores_df, SF)

    return scores_df