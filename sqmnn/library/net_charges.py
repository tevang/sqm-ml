import pandas as pd

def EXEC_load_all_net_charges(proteins, Settings):
    dfs = []
    for protein in proteins:
        df = pd.read_csv(Settings.raw_input_file("_all_compounds-net_charges.csv.gz", protein),
                        names=["structvar", "net_charge"],
                        header=0)
        df["structvar"] = df["structvar"].str.lower()
        df["protein"] = protein
        dfs.append(df)
    return pd.concat(dfs)
