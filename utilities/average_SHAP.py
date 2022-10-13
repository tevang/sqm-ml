import pandas as pd

prot_sets = {
    'crossval': ['ACHE', 'EPHB4', 'JNK2', 'MDM2', 'PARP-1'],
}

for name, prot_set in prot_sets.items():
    pd.concat([pd.read_csv(f'/home2/shared_files/sqm-ml_data/plots/{p}_SHAP_importances.csv') \
              .assign(csv=f'/home2/shared_files/sqm-ml_data/plots/{p}_SHAP_importances.csv') for p in prot_set],
              ignore_index=True) \
        .groupby(by=['feature']) \
        .apply('mean') \
        .reset_index() \
        .sort_values(by='importance', ascending=False) \
        .to_csv(f'average_{name}_SHAP_per_feature.csv', index=False)

