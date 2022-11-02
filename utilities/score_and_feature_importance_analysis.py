#!/usr/bin/env python
import re

import pandas as pd
from library.global_fun import list_files

SCORES_DIR = "."
SHAP_VALUES_DIR = "/media/thomas/external_drive_4TB/thomas-GL552VW/Documents/SQM-ML/plots_SHAP_per_receptor_with_outliers"

if os.path.exists(os.path.join(SHAP_VALUES_DIR, 'average_features_SQM-ML_scores.csv.gz')):
    mean_score_df.read_csv(os.path.join(SHAP_VALUES_DIR, 'average_features_SQM-ML_scores.csv.gz'))
else:
    df_list = []
    for score_csv in list_files(folder=f'repeat*_scores/', pattern='.*_features_SQM-ML_scores.csv.gz', rel_path=True):
        m = re.search('repeat([0-9]+)_scores/(.+)_features_SQM-ML_scores.csv.gz', score_csv)
        repeat, protein = m.groups()
        df_list.append(pd.read_csv(score_csv).assign(repeat=repeat, protein=protein))

    mean_score_df = pd.concat(df_list).groupby(by=['protein', 'basemolname', 'structvar', 'pose']).agg('mean').reset_index()
    mean_score_df.to_csv(os.path.join(SHAP_VALUES_DIR, 'average_features_SQM-ML_scores.csv.gz'), index=False)


for protein in ['TP', 'GR', 'DHFR']:
    df = mean_score_df.loc[mean_score_df.protein==protein, ['structvar', 'pose', 'SQM_ML_score', 'nofusion_Eint', 'is_active']] \
        .assign(SQM_ML_score_rank=lambda df: df['SQM_ML_score'].rank(pct=True), Eint_rank=lambda df: df['nofusion_Eint'].rank(pct=True),
                rank_diff=lambda df: df['SQM_ML_score_rank']-df['Eint_rank'],
                complex_name=lambda df: df.apply(lambda r: f"{r['structvar']}_pose{r['pose']}_frm1_noWAT.pdb", axis=1)) \
        .sort_values(by='rank_diff').drop(columns=['SQM_ML_score', 'nofusion_Eint', 'pose'])
    print("Actives overscored by SQM-ML:\n", df[(df.SQM_ML_score_rank>0.5) & (df.is_active==1)].iloc[-10:])
    print("Actives overscored by PM6/COSMO:\n", df[(df.Eint_rank>0.5) & (df.is_active==1)].iloc[:10])
    print("Inactives underscored by SQM-ML:\n", df[(df.SQM_ML_score_rank<0.5) & (df.is_active==0)].iloc[-10:])
    print("Inactives underscored by PM6/COSMO:\n", df[(df.Eint_rank<0.5) & (df.is_active==0)].iloc[:10])

    # Save the actives and inactives which were badly scored by SQM-ML
    pd.concat([df[(df.SQM_ML_score_rank>0.5) & (df.is_active==1)].iloc[-10:],
               df[(df.SQM_ML_score_rank<0.5) & (df.is_active==0)].iloc[-10:]])['structvar'] \
        .to_csv(f'{protein}_mols_badly_scored_by_SQM-ML.csv', header=False, index=False)
    pd.concat([df[(df.SQM_ML_score_rank>0.5) & (df.is_active==1)].iloc[-10:],
               df[(df.SQM_ML_score_rank<0.5) & (df.is_active==0)].iloc[-10:]])['complex_name'] \
        .to_csv(f'{protein}_complexes_badly_scored_by_SQM-ML.csv', header=False, index=False)

    # Save the actives and inactives which were badly scored by PM6/COSMO
    pd.concat([df[(df.Eint_rank>0.5) & (df.is_active==1)].iloc[-10:],
               df[(df.Eint_rank<0.5) & (df.is_active==0)].iloc[-10:]])['structvar'] \
        .to_csv(f'{protein}_mols_badly_scored_by_Eint.csv', header=False, index=False)
    pd.concat([df[(df.Eint_rank>0.5) & (df.is_active==1)].iloc[-10:],
               df[(df.Eint_rank<0.5) & (df.is_active==0)].iloc[-10:]])['complex_name'] \
        .to_csv(f'{protein}_complexes_badly_scored_by_Eint.csv', header=False, index=False)
