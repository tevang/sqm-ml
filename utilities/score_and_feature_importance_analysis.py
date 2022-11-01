#!/usr/bin/env python

import pandas as pd
from library.global_fun import list_files

SCORES_DIR = ""
SHAP_VALUES_DIR = ""

df_list = []
for score_csv in list_files(folder=f'repeat*_scores/', pattern='.*_features_SQM-ML_scores.csv.gz', rel_path=True):
    m = re.search('repeat([0-9]+)_scores/(.+)_features_SQM-ML_scores.csv.gz', score_csv)
    repeat, protein = m.groups()
    df_list.append(pd.read_csv(score_csv).assign(repeat=repeat, protein=protein))

mean_score_df = pd.concat(df_list).groupby(by=['protein', 'basemolname', 'structvar', 'pose']).agg('mean').reset_index()

for protein in ['GR', 'DHFR']:
    df = mean_score_df.loc[mean_score_df.protein==protein, ['structvar', 'pose', 'SQM_ML_score', 'nofusion_Eint', 'is_active']] \
        .assign(SQM_ML_score_rank=lambda df: df['SQM_ML_score'].rank(pct=True), Eint_rank=lambda df: df['nofusion_Eint'].rank(pct=True),
                rank_diff=lambda df: df['SQM_ML_score_rank']-df['Eint_rank']).sort_values(by='rank_diff')
    print("Actives overscored by SQM-ML:", df[df.is_active==1].iloc[-10:])
    print("Actives overscored by PM6/COSMO:", df[df.is_active==1].iloc[:10])
    print("Inactives underscored by SQM-ML:", df[df.is_active==0].iloc[-10:])
    print("Inactives underscored by PM6/COSMO:", df[df.is_active==0].iloc[:10])