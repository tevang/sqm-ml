import logging

import pandas as pd

from sqmnn.commons.EXEC_caching import EXEC_caching_decorator
from sqmnn.library.energy_fluctuations import protein_energy_fluctuations_between_structvars, \
    protein_energy_fluctuations_between_basemonames

lg = logging.getLogger(__name__)

@EXEC_caching_decorator(lg, "Computing SQM energy fluctuations.", "_SQM_energy_fluctuations.csv.gz")
def EXEC_compute_all_sqm_energy_fluctuations(best_scores_df, scores_df, Settings):
    fluct_df = pd.concat([protein_energy_fluctuations_between_structvars(scores_df,
                                                                           energy_term=energy_term)
                           for energy_term in ["complexE", "proteinE_bound", "ligandE_bound"]] +
                          [protein_energy_fluctuations_between_basemonames(best_scores_df,
                                                                           energy_term=energy_term,)
                           for energy_term in ["complexE", "proteinE_bound", "ligandE_bound"]] +
                          [protein_energy_fluctuations_between_basemonames(scores_df,
                                                                           energy_term=energy_term,
                                                                           suffix="_overall_basemolname_stdev")
                           for energy_term in ["complexE", "proteinE_bound", "ligandE_bound"]]
                          )
    print(fluct_df.sort_index())
    return best_scores_df.merge(fluct_df.to_frame().T.assign(protein=best_scores_df['protein'].iloc[0]),
                                on='protein')