import pandas as pd

def umap_compressor(X_train, X_test):
    import umap.umap_ as UMAP

    ncomp = 2
    umap_reducer = UMAP(n_neighbors=15, min_dist=0.1, n_components=ncomp, metric='hamming')
    umap_reducer.fit(X_train)
    # TODO ump column names
    return pd.DataFrame(umap_reducer.transform(X_train),
                        columns=['ump%i' % u for u in range(1, ncomp+1)]), \
           pd.DataFrame(umap_reducer.transform(X_test),
                        columns=['ump%i' % u for u in range(1, ncomp+1)])


def umap_compress_fingerprint(features_df, crossval_protein, xtest_protein,
                             fingerprint_type='plec'):
    import umap.umap_ as UMAP

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