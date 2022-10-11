import pandas as pd

prot_sets = {
    'xtest': ['A2A', 'CATL', 'DHFR', 'GBA', 'GR', 'HIV1RT', 'MK2', 'PPARG', 'SARS-HCoV', 'SIRT2', 'TPA', 'TP'],
    'crossval': ['ACHE', 'EPHB4', 'JNK2', 'MDM2', 'PARP-1']
}

for name, prot_set in prot_sets.items():
    pd.concat([pd.read_csv(f'/home2/shared_files/sqm-ml_data/plots/{p}_SHAPLEY_importances.csv') \
              .assign(csv=f'/home2/shared_files/sqm-ml_data/plots/{p}_SHAPLEY_importances.csv') for p in prot_set],
              ignore_index=True) \
        .groupby(by=['feature']) \
        .apply('mean') \
        .reset_index() \
        .sort_values(by='importance', ascending=False) \
        .to_csv(f'average_{name}_SHAP_per_feature.csv', index=False)

