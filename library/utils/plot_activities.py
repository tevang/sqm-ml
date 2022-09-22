import os.path

import plotly.express as px


def plot_ump_with_label_3d(ump_df, figure_title, execution_dir, label_column):
    """

    Parameters
    ----------
    df: dataframe with features, the df needs a label y
    figure_title: title of figure
    execution_dir: folder path
    label_column: column in df to label the plot

    Returns
    -------
    Shows and saves figure

    """
    if not os.path.exists(execution_dir):
        os.mkdir(execution_dir)

    fig = px.scatter_3d(ump_df.assign(color_column=lambda df: df[label_column].replace(
        {0: 'inactive', 1: 'active'})), x='ump1', y='ump2', z='ump3',
                        color='color_column',
                        hover_name='basemolname',
                        color_discrete_map={'active': 'red', 'inactive': 'rgb(102,102,102)'},
                        opacity=.7)
    fig.update_layout(title=f'UMAP_{figure_title}',
                      paper_bgcolor='white',
                      plot_bgcolor='white')
    fig.update_traces(marker=dict(size=3),
                      line=dict(width=2,
                                color='DarkSlateGrey'),
                      selector=dict(mode='markers'))

    fig.write_html(os.path.abspath(os.path.join(execution_dir, f'UMAP_{figure_title}.html')))
    fig.show()