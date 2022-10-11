import pandas as pd

from library.global_fun import list_files

df = pd.concat([pd.read_csv(csv).assign(csv=csv) for csv in list_files(
    '/home2/shared_files/sqm-ml_data/plots', pattern='.*\.csv', full_path=True)], ignore_index=True) \
    .groupby(by=['feature']) \
    .apply('mean') \
    .reset_index()
df.to_csv('average_SHAP_per_feature.csv', index=False)

