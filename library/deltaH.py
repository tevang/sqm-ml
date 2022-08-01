from lib.global_fun import save_pickle


def calc_overall_DeltaH(protein,
                        scores_df,
                        Settings,
                        scoring_function="P6C",
                        min_ligandE_column_name="min_ligandE_free",
                        DeltaH_column_name="DeltaH"):
    """
    STEP 1: Find the lowest complexE* and the associated pose for each structural variant __(aka the best docking pose)__.
    STEP 2: find the global lowest proteinE_bound in the whole dataset
    STEP 3: find the conformer of each structvar with the lowest ligandE_free (from those you created with Macromodel)
    STEP 4: calculate the DeltaH of the __best docking pose__ of each structuvar which is (1)-(2)-(3)
    STEP 5: select the best structvar of each molecule based on (4)
           # NOTE: the correct would have been to include also the entropy of each structvar.
    STEP 6: write (5) and (4) to the output file.
    e.g. complex_name = "LEM00010809_stereo1_ion1_tau1_pose7_noWAT"

    * Why we select the best pose per structvar by complexE and not by Eint.
    DeltaH = complexE -proteinE_bound -ligandE_bound + (ligandE_bound -ligandE_free) + (proteinE_bound -proteinEfree) ==>
    --> DeltaH = complexE -ligandE_free -proteinEfree , but because ligandE_free & proteinEfree are the same for every pose,
    we can safely ignore them and select the best pose just by complexE
    """

    # STEP 1
    min_complexE_df = scores_df.groupby("structvar", as_index=False)\
        .apply(lambda x: x[x[scoring_function+"_complexE"] == x[scoring_function+"_complexE"].min()]) \
        .rename(columns={"pose": scoring_function+"_best_pose"})

    # STEP 2
    min_complexE_df[scoring_function + "_min_proteinE_bound"] = scores_df[scoring_function + "_proteinE_bound"].min()

    # STEP 3: min_ligandE_free is already in the scores_df

    # STEP 4
    # The following is unnecessary, but quite elegantly programmed
    # scores_df["min_ligandE_free"] = scores_df.groupby("structvar", as_index=False)["structvar"].transform(
    #     lambda x: [min_ligandE_free_df[min_ligandE_free_df["structvar"] == s]["min_ligandE_free"].values[0]
    #                for s in x])

    min_complexE_df[scoring_function+"_"+DeltaH_column_name] = \
        min_complexE_df \
            .apply(lambda r:
                   min_complexE_df[min_complexE_df["structvar"]==r["structvar"]][scoring_function+"_complexE"].iloc[0]
                   - min_complexE_df[scoring_function + "_min_proteinE_bound"].iloc[0]
                   - scores_df[scores_df["structvar"]==
                               r["structvar"]][scoring_function+"_"+min_ligandE_column_name].iloc[0],
                   axis=1)

    # # STEP 5 (is conducted later by keep_structvar_with_extreme_col_value())
    # min_DeltaH_df = min_complexE_df.groupby("basemolname", as_index=False)\
    #     .apply(lambda g: g[g[scoring_function+"_"+DeltaH_column_name] ==
    #                        g[scoring_function+"_"+DeltaH_column_name].min()])   # groupby() is unnecessary but added for assurance

    columns = ['basemolname',
               'stereoisomer',
               'ionstate',
               'tautomer',
               'frame',
               'protein',
               'structvar',
               scoring_function + "_complexE",
               scoring_function + "_proteinE_bound",
               scoring_function + "_min_proteinE_bound",
               scoring_function + "_" + min_ligandE_column_name,
               scoring_function + '_best_pose',
               scoring_function+'_'+DeltaH_column_name]

    if not Settings.HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY.endswith("DeltaHstar") and not \
            Settings.HYPER_SELECT_BEST_STRUCTVAR_POSE_BY.endswith("DeltaHstar") and not \
            any([f.endswith('DeltaHstar') or f.endswith('_min_ligandE_bound') for f in Settings.HYPER_SQM_FEATURES]):
        columns += ['schrodinger_Sconf']

    return min_complexE_df[columns]