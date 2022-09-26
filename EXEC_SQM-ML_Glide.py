import pandas as pd

from library.evaluate_model import evaluate_without_learning

best_sqm_df = pd.read_csv('18_proteins_scaled_nonuniform_all_features.SF_P6C.SBBSB_Eint.SBSPB_complexE.HC_inner.FM_nofusion.RSP_0.8.OMT_8.0.OMSPN_0.KMNP_100.KMDP_1.000000.US_False.csv.gz') \
    .dropna(subset=['basemolname', 'nofusion_Eint']) \
    .sort_values(by='nofusion_Eint') \
    .groupby(by=['protein', 'basemolname'], as_index=False) \
    .apply(lambda g: g.iloc[0])[['basemolname', 'nofusion_Eint', 'protein', 'is_active']]

best_glide_df = pd.concat([pd.read_csv('{}_glide_best_structvar.csv.gz'.format(protein)) \
                          .dropna(subset=['basemolname', 'nofusion_r_i_docking_score']) \
                          .sort_values(by='nofusion_r_i_docking_score') \
                          .groupby(by=['basemolname'], as_index=False) \
                          .apply(lambda g: g.iloc[0])[['basemolname', 'nofusion_r_i_docking_score']]
                           for protein in best_sqm_df['protein'].unique()])

commond_df = best_sqm_df.merge(best_glide_df, on='basemolname', how='left')

for protein in commond_df['protein'].unique():
    print("{} Glide:".format(protein),
          evaluate_without_learning(commond_df[commond_df['protein']==protein], "nofusion_r_i_docking_score")[3])
    print("{} SQM:".format(protein), evaluate_without_learning(commond_df[commond_df['protein']==protein],
                                                               "nofusion_Eint")[3])

