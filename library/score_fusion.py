def fuse_columns(df, columns, method):
    """
    Method for score fusion.

    :param df:
    :param columns:
    :param method:
    :return:
    """
    # TODO: instead of ranks, fuse Z-scores.

    if method == "nofusion":
        return df[columns]
    elif method == "minrank":
        return df[columns].min(axis=1)
    elif method == "meanrank":
        return df[columns].sum(axis=1)
    elif method == "geomean":
        return df[columns].product(axis=1).pow(1/len(columns))
    elif method == "harmmean":
        return 1/((1/df[columns]).sum(axis=1)/len(columns))