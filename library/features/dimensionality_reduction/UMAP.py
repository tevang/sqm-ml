import pandas as pd
from umap import UMAP

def umap_compressor(X_train, X_test):
    ncomp = 2
    umap_reducer = UMAP(n_neighbors=200, min_dist=0.01, n_components=ncomp, metric='correlation')
    umap_reducer.fit(X_train)
    print('======================', X_train.shape)
    # TODO ump column names
    return pd.DataFrame(umap_reducer.transform(X_train),
                        columns=['ump%i' % u for u in range(1, ncomp+1)]), \
           pd.DataFrame(umap_reducer.transform(X_test),
                        columns=['ump%i' % u for u in range(1, ncomp+1)])


def umap_compress_fingerprint(features_df, crossval_protein, xtest_protein,
                             fingerprint_type='plec'):
    # TODO: UNTESTED!
    print("Reducing %s dimensions with UMAP" % fingerprint_type)
    train_features_df = features_df.loc[features_df["protein"].isin(crossval_protein), :]
    test_features_df = features_df.loc[features_df["protein"].isin(xtest_protein), :]
    other_columns = train_features_df.columns[~train_features_df.columns.str.startswith(fingerprint_type)]

    train_compressed_df, test_compressed_df = umap_compressor(
        train_features_df.filter(regex='^%s[0-9]+$' % fingerprint_type),
        test_features_df.filter(regex='^%s[0-9]+$' % fingerprint_type))
    print('****************\n',
          'UMAP passed\n',
          train_compressed_df.columns)
    return pd.concat([train_features_df[other_columns].reset_index().join(train_compressed_df),
                      test_features_df[other_columns].reset_index().join(test_compressed_df)]) \
        .pipe(lambda df: df.rename(columns={c: '%s_%s' % (fingerprint_type, c)
                                            for c in df.filter(regex='^ump[0-9]+$').columns}))

def _ump_trans(x, ncomp=2):
    umap_reducer = UMAP(n_neighbors=50, min_dist=0.1, n_components=ncomp, metric='correlation')
    umap_reducer.fit(x)
    ump_df = pd.DataFrame(umap_reducer.transform(x),
                          columns=['ump%i' % u for u in range(1, ncomp+1)], index=x.index)

    return ump_df