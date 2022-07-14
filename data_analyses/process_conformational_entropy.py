import pandas as pd
from sqmnn.SQMNN_pipeline_settings import Settings

# LOAD ALL ACTIVITIES
from sqmnn.library.evaluate_model import evaluate_without_learning
from sqmnn.library.EXEC_join_functions import EXEC_merge_dataframes_on_columns

settings = Settings()
activity_df = pd.concat([
    pd.read_csv(settings.raw_input_file("_activities.txt", protein),
                names=["basemolname", "is_active"],
                header=0,
                delim_whitespace=True)
    for protein in ["MDM2","MARK4","ACHE","MK2","EPHB4","PPARG","PARP-1"]
])
activity_df["basemolname"] = activity_df["basemolname"].str.lower()

# LOAD CONFORMATIONAL ENTROPY
dfs = []
for rms in [1.0, 1.5, 2.0]:
    for e in [6,9,12]:
        df = pd.concat([
            pd.read_csv(settings.raw_input_file(".schrodinger_Sconf_rmscutoff%.1f_ecutoff%i.csv.gz" % (rms, e), protein),
                        names=["structvar", "Sconf_rmscutoff%.1f_ecutoff%i" % (rms, e), "scaled_Sconf_rmscutoff%.1f_ecutoff%i" % (rms, e)],
                        header=0,
                        usecols=["structvar", "Sconf_rmscutoff%.1f_ecutoff%i" % (rms, e)])
            for protein in ["MDM2","MARK4","ACHE","MK2","EPHB4","PPARG","PARP-1"]
        ])
        df["basemolname"] = df["structvar"].str \
            .extract("^(.*)_stereo[0-9]+_ion[0-9]+_tau[0-9]+$") \
            .rename({0: 'basemolname'}, axis=1)["basemolname"].str.lower()
        dfs.append( df.groupby("basemolname", as_index=False).apply(min) )

entropy_df = EXEC_merge_dataframes_on_columns("basemolname", dfs)
entropy_df.to_csv(settings.generated_file("_entropies.csv.gz", "all"))

# MERGE ENTROPY AND ACTIVITY
entropy_df = pd.merge(entropy_df, activity_df, on="basemolname")

# EVALUATE PERFORMACE AS A CLASSIFIER
entropy_columns = [c for c in entropy_df.columns if c.startswith("Sconf")]
for c in entropy_columns:
    evaluate_without_learning(entropy_df[[c, 'is_active']], column=c)

# COMPUTE CORRELATIONS
entropy_df[entropy_columns].columns
entropy_df[entropy_columns].corr().round(2).values
entropy_df[entropy_columns].corr().round(2).values.min()