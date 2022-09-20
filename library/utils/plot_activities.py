import os.path
from library.features.dimensionality_reduction.UMAP import _ump_trans
import plotly.express as px
from plotly.graph_objs import *

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


def _plot_ump_custom(df, protein):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    markers = {1: "X", 0: "."}
    color_dict = dict({1: 'red', 0: 'grey'})
    sns.scatterplot(data=df, x='ump1', y='ump2',  hue='y', style='y',
                    markers=markers, palette= color_dict)
    title = f'ump_{protein}'
    print(protein)
    print(os.path.abspath('results/'))

    plt.title(title)
    plt.savefig(f'results/{title}.png')
    plt.show()
    plt.close()


def _plot_vae_custom(df, protein):
    import matplotlib.pyplot as plt
    import seaborn as sns
    fig, ax = plt.subplots()
    markers = {1: "X", 0: "."}
    color_dict = dict({1: 'red', 0: 'grey'})
    sns.scatterplot(data=df, x='vae1', y='vae2', hue='y', style='y',
                    markers=markers, palette=color_dict)
    title = f'vae_{protein}'
    plt.title(title)

    plt.savefig(f'results/{title}.png')
    plt.show()
    plt.close()


def plot_ump_with_label_3d(df, figure_title, execution_dir, label_column):
    """

    Parameters
    ----------
    df: dataframe with features, the df needs a label y
    figure_title: title of figure
    execution_dir: folder path, no slash at the end
    label_column: column in df to label the plot

    Returns
    -------
    Shows and saves figure

    """

    ump_df = _ump_trans(df.drop(label_column, axis=1), ncomp=3).join(df[label_column]).reset_index(drop=True)

    lmap = {0:'inactive', 1:'active'}
    cmap = ['red', 'whitesmoke']
    ump_df['color_column'] = ump_df[label_column].map(lmap)

    ump_df['name'] = ump_df.index.to_series()

    fig = px.scatter_3d(ump_df, x='ump1', y='ump2', z='ump3',
                        color='color_column',
                        hover_name='name',

                        color_discrete_sequence=cmap,
                        opacity=.7)
    fig.update_layout(title=f'UMAP_{figure_title}',
                      paper_bgcolor='white',
                      plot_bgcolor='white')

    fig.update_traces(marker=dict(size=3),
                      line=dict(width=2,
                                color='DarkSlateGrey'),
                      selector=dict(mode='markers'))

    title = f'UMAP_{figure_title}'

    fig.write_html(os.path.abspath(f'{execution_dir}/{title}.html'))
    fig.show()


def plot_ump_with_protein(df, figure_title, execution_dir, label_column, protein_col):
    """

    Parameters
    ----------
    df: dataframe with features, the df needs a label y
    figure_title: title of figure
    execution_dir: folder path, no slash at the end
    label_column: column in df to label the plot

    Returns
    -------
    Shows and saves figure

    """
    if not os.path.exists(execution_dir):
        os.mkdir(execution_dir)

    ump_df = _ump_trans(df.drop([label_column, protein_col],
                                axis=1), ncomp=3).join(df[[label_column, protein_col]]).reset_index(drop=True)

    lmap = {0:'inactive', 1:'active'}
    cmap = ['red', 'silver']

    ump_df['color_column'] = ump_df[label_column].map(lmap)


    fig = px.scatter_3d(ump_df, x='ump1', y='ump2', z='ump3',
                        color='color_column',
                        hover_name=protein_col,

                        color_discrete_sequence=cmap,
                        opacity=.7)
    fig.update_layout(title=f'UMAP_{figure_title}',
                      paper_bgcolor='white',
                      plot_bgcolor='white')

    fig.update_traces(marker=dict(size=3),
                      line=dict(width=2,
                                color='DarkSlateGrey'),
                      selector=dict(mode='markers'))

    title = f'UMAP_prot_{figure_title}'

    fig.write_html(os.path.abspath(f'{execution_dir}/{title}.html'))
    fig.show()
