def EXEC_decompose_molname(df):
    """
    First converts all molnames to lowercase. Then it splits the molname to basemolname, stereoisomer,
    ionstate, tautomer, pose and adds them as extra columns to the input dataframe.
    :param df: dataframe containing "molname" column.
    :return:
    """
    return df.assign(molname=df["molname"].str.lower()).join(
        df["molname"].str.lower().str \
            .extract("^(.*)_stereo([0-9]+)_ion([0-9]+)_tau([0-9]+)_pose([0-9]+)$") \
            .rename({0: 'basemolname', 1: 'stereoisomer', 2: 'ionstate', 3: 'tautomer', 4: 'pose'}, axis=1) \
            .astype({'basemolname': str, 'stereoisomer': str, 'ionstate': str, 'tautomer': str, 'pose': str}))


