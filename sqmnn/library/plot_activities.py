def _plot_ump(x):
    """

    Parameters
    ----------
    x

    Returns
    -------

    """
    import matplotlib.pyplot as plt
    plt.scatter(x.ump1, x.ump2, s=3, alpha=.5)
    plt.show()


def _plot_ump_custom(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    markers = {1: "X", 0: "."}
    color_dict = dict({1: 'red', 0: 'grey'})
    sns.scatterplot(data=df, x='ump1', y='ump2',  hue='y', style='y',
                    markers=markers, palette= color_dict)
    plt.show()
    plt.close()


def _plot_vae_custom(df):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    markers = {1: "X", 0: "."}
    color_dict = dict({1: 'red', 0: 'grey'})
    sns.scatterplot(data=df, x='vae1', y='vae2', hue='y', style='y',
                    markers=markers, palette=color_dict)
    plt.show()
    plt.close()


# for plotting true predicted
# def confusion(collection, t, p):
#     tp = collection[(collection[t] == 1) & (collection[p] == 1)].index
#     tn = collection[(collection[t] == 0) & (collection[p] == 0)].index
#     fp = collection[(collection[t] == 0) & (collection[p] == 1)].index
#     fn = collection[(collection[t] == 1) & (collection[p] == 0)].index
#     return tp, tn, fp, fn
#
#
# y_pred = learning_model_functions[learning_model_type].predict(features_df[sel_columns])
# y_pred = pd.Series(y_pred, index=features_df['is_active'].index)
# y_true = features_df['is_active']
# ys = pd.concat([y_true, y_pred], axis=1)
# ys.columns = ['t', 'p']
#
#
# # tp, tn, fp, fn = confusion(ys, 't','p')
# def total(r):
#     if (r['t'] == 1) and (r['p'] == 1):
#         return 'tp'
#     elif (r['t'] == 0) and (r['p'] == 0):
#         return 'tn'
#     elif (r['t'] == 0) and (r['p'] == 1):
#         return 'fp'
#     elif (r['t'] == 1) and (r['p'] == 0):
#         return 'fn'
#     else:
#         return None
#
#
# res = ys.apply(total, axis=1)
#
#
# def _plot_ump(x, color):
#     import matplotlib.pyplot as plt
#     color = color.replace({'tp': 'green', 'tn': 'lightblue', 'fn': 'red', 'fp': 'orange'})
#     plt.scatter(x.ump1, x.ump2, s=8, alpha=.7, c=color)
#     plt.show()


