import pandas as pd
from umap import UMAP

from library.utils.print_functions import ColorPrint


def umap_compress_fingerprint(features_df, n_neighbors, min_dist, n_components, metric, fingerprint_type):
    ColorPrint("Reducing %s dimensions with UMAP from %i to % i " % (fingerprint_type, features_df.shape[1],
                                                                     n_components), "OKBLUE")
    other_columns = features_df.columns[~features_df.columns.str.startswith(fingerprint_type)]
    compressed_df = _ump_trans(features_df.filter(regex='^%s[0-9]+$' % fingerprint_type),
                               n_neighbors, min_dist, n_components, metric)
    return features_df[other_columns].reset_index().join(compressed_df) \
        .pipe(lambda df: df.rename(columns={c: '%s_%s' % (fingerprint_type, c)
                                            for c in df.filter(regex='^ump[0-9]+$').columns}))

def _ump_trans(x, n_neighbors=50, min_dist=0.1, n_components=3, metric='correlation'):
    ColorPrint("Reducing dimensions with UMAP from %i to % i " % (x.shape[1], n_components), "OKBLUE")
    umap_reducer = UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, metric=metric)
    ump_df = pd.DataFrame(umap_reducer.fit_transform(x),
                          columns=['ump%i' % u for u in range(1, n_components+1)], index=x.index)

    return ump_df