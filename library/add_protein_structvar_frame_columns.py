def add_protein_structvar_frame_columns(df, Settings):
    return df.assign(protein = Settings.HYPER_PROTEIN,
                     structvar = lambda df: df \
                     .apply(lambda r: "%s_stereo%s_ion%s_tau%s" %
                                      (r.basemolname, r.stereoisomer, r.ionstate, r.tautomer), axis=1),
                     frame=1)