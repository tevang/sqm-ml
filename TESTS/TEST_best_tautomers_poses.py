import pandas as pd

best_df = pd.read_csv("PARP-1_scores_best_structvar.csv.gz")
sqm_df = pd.read_csv("PARP-1_sqm.csv.gz")

for i, r in best_df.iterrows():
    print(r[['basemolname', 'nofusion_Eint', 'stereoisomer', 'ionstate', 'tautomer', 'pose',
       'protein', 'structvar', 'is_active']].to_frame().T)
    print(sqm_df.loc[sqm_df.basemolname == r.basemolname,
                     ['basemolname', 'P6C_Eint', 'stereoisomer', 'ionstate', 'tautomer', 'pose',
                      'protein', 'structvar']])


basemolname = "zinc01618178"

df2 = df[df.groupby(by="structvar")["P6C_Eint"] \
                      .transform(min) == df["P6C_Eint"]]

df2[df2.groupby(by="basemolname")["P6C_Eint"] \
                  .transform(min) == df2["P6C_Eint"]]

HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY = "Eint"
HYPER_SELECT_BEST_STRUCTVAR_POSE_BY = "Eint"
HYPER_FUSION_METHOD = 'nofusion'
HYPER_SFs = ["P6C"]
relation = min

df[df.groupby(by="structvar")[HYPER_SELECT_BEST_STRUCTVAR_POSE_BY if HYPER_SELECT_BEST_STRUCTVAR_POSE_BY.startswith("r_i_")
        else HYPER_SFs[0] + "_" + HYPER_SELECT_BEST_STRUCTVAR_POSE_BY] \
                      .transform(relation) == df[HYPER_SELECT_BEST_STRUCTVAR_POSE_BY if HYPER_SELECT_BEST_STRUCTVAR_POSE_BY.startswith("r_i_")
        else HYPER_SFs[0] + "_" + HYPER_SELECT_BEST_STRUCTVAR_POSE_BY]] \
            .pipe(lambda df2: df2[df2.groupby(by="basemolname")[HYPER_SFs[0] + "_" + HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY] \
                  .transform(relation) == df2[HYPER_SFs[0] + "_" + HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY]]) \
            .rename(columns={HYPER_SFs[0] + "_" + HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY:
                                 "%s_%s" % (HYPER_FUSION_METHOD,
                                            HYPER_SELECT_BEST_BASEMOLNAME_SCORE_BY)})