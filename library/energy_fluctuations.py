from sklearn.preprocessing import minmax_scale


def _normalizing_fun(s): return minmax_scale(s).std()


def protein_energy_fluctuations_between_structvars(proteinE_bound_df, energy_term="proteinE_bound",
                                                   scale=False, suffix="_mean_structvar_stdev"):

    structvar_proteinE_fluctuations_df = proteinE_bound_df.groupby(by=['basemolname', 'stereoisomer', 'ionstate', 'tautomer']) \
        .agg({col: _normalizing_fun if scale else 'std' for col in proteinE_bound_df.filter(regex="P[67]C?_" + energy_term).columns}) \
        .rename(columns={col: col + suffix for col in proteinE_bound_df.filter(regex="P[67]C?_" + energy_term).columns})

    return structvar_proteinE_fluctuations_df.reset_index(level=structvar_proteinE_fluctuations_df.index.names) \
        .filter(regex="P[67]C?_" + energy_term) \
        .mean()

def protein_energy_fluctuations_between_basemonames(proteinE_bound_df, energy_term="proteinE_bound",
                                                    scale=False, suffix="_best_basemolname_stdev"):

    return proteinE_bound_df \
        .agg({col: _normalizing_fun if scale else 'std' for col in proteinE_bound_df.filter(regex="P[67]C?_" + energy_term).columns}) \
        .rename(index={col: col + suffix for col in proteinE_bound_df.filter(regex="P[67]C?_" + energy_term).columns})


