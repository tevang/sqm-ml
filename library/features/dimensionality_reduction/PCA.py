import pandas as pd
from sklearn.decomposition import PCA


def _pca_compressor(X, pca_variance_explained_cutoff=0.8):
    pca = PCA(n_components=pca_variance_explained_cutoff)
    return pd.DataFrame(pca.fit_transform(X), columns=['pc%i' % pc for pc in range(1, pca.n_components_+1)])


def pca_compress_fingerprint(features_df, pca_variance_explained_cutoff=0.8,
                             fingerprint_type='plec'):
    print("Reducing %s dimensions with PCA." % fingerprint_type)
    other_columns = features_df.columns[~features_df.columns.str.startswith(fingerprint_type)]
    compressed_df = _pca_compressor(
        features_df.filter(regex='^%s[0-9]+$' % fingerprint_type),
        pca_variance_explained_cutoff=pca_variance_explained_cutoff)
    return features_df[other_columns].reset_index().join(compressed_df) \
        .pipe(lambda df: df.rename(columns={c: '%s_%s' % (fingerprint_type, c)
                                            for c in df.filter(regex='^pc[0-9]+$').columns}))

