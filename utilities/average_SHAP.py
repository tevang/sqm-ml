import pandas as pd

xtest = ['A2A', 'CATL', 'DHFR', 'GBA', 'GR', 'HIV1RT', 'MK2', 'PPARG', 'SARS-HCoV', 'SIRT2', 'TPA', 'TP']

df = pd.concat([pd.read_csv(f'/home2/shared_files/sqm-ml_data/plots/{p}_SHAP_importances.csv') \
               .assign(csv=f'/home2/shared_files/sqm-ml_data/plots/{p}_SHAP_importances.csv') for p in xtest],
               ignore_index=True) \
    .groupby(by=['feature']) \
    .apply('mean') \
    .reset_index() \
    .sort_values(by='importance', ascending=False)
df.to_csv('average_xtest_SHAP_per_feature.csv', index=False)

