import pandas as pd
from sklearn.decomposition import PCA


def pca_compressor(X_train, X_test, pca_variance_explained_cutoff=0.8):
    pca = PCA(n_components=pca_variance_explained_cutoff)
    pca.fit(X_train)
    return pd.DataFrame(pca.transform(X_train), columns=['pc%i' % pc for pc in range(1, pca.n_components_+1)]), \
           pd.DataFrame(pca.transform(X_test), columns=['pc%i' % pc for pc in range(1, pca.n_components_+1)])


def pca_compress_fingerprint(features_df, crossval_protein, xtest_protein, pca_variance_explained_cutoff=0.8,
                             fingerprint_type='plec'):
    # TODO: UNTESTED!
    print("Reducing %s dimensions with PCA." % fingerprint_type)
    train_features_df = features_df.loc[features_df["protein"].isin(crossval_protein), :]
    test_features_df = features_df.loc[features_df["protein"].isin(xtest_protein), :]
    other_columns = train_features_df.columns[~train_features_df.columns.str.startswith(fingerprint_type)]
    train_compressed_df, test_compressed_df = pca_compressor(
        train_features_df.filter(regex='^%s[0-9]+$' % fingerprint_type),
        test_features_df.filter(regex='^%s[0-9]+$' % fingerprint_type),
        pca_variance_explained_cutoff=pca_variance_explained_cutoff)
    return pd.concat([train_features_df[other_columns].reset_index().join(train_compressed_df),
                      test_features_df[other_columns].reset_index().join(test_compressed_df)]) \
        .pipe(lambda df: df.rename(columns={c: '%s_%s' % (fingerprint_type, c)
                                            for c in df.filter(regex='^pc[0-9]+$').columns}))

