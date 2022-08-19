from sklearn.preprocessing import minmax_scale


def minmax_scale_crossval_xtest(crossval_molname_fingerprint_dict, xtest_molname_fingerprint_dict={}):
    """
    Scale feature vector value from 0 to 1 (crossval + xtest). This method takes the feature vectors as dictionaries
    and OPTIONALLY scales the values using the min, max and stdev of each feature in both crossval and xtest set,
    and returns the two scaled sets individually. If the xtest dict is not given, then it will scale based on the crossval.

    :param crossval_molname_fingerprint_dict:
    :param xtest_molname_fingerprint_dict:
    :return:
    """
    scaled_crossval_molname_fingerprint_dict, scaled_xtest_molname_fingerprint_dict = {}, {}
    crossval_molnames = list(crossval_molname_fingerprint_dict.keys())
    xtest_molnames = list(xtest_molname_fingerprint_dict.keys())
    all_fingerprints = [crossval_molname_fingerprint_dict[m] for m in crossval_molnames] + \
                       [xtest_molname_fingerprint_dict[m] for m in xtest_molnames]
    scaled_all_fingerprints = list(minmax_scale(all_fingerprints))
    scaled_crossval_molname_fingerprint_dict = {m: fp for m, fp in zip(crossval_molnames, scaled_all_fingerprints[
                                                                                          :len(crossval_molnames)])}
    scaled_xtest_molname_fingerprint_dict = {m: fp for m, fp in
                                             zip(xtest_molnames, scaled_all_fingerprints[len(xtest_molnames):])}
    if xtest_molname_fingerprint_dict:
        return scaled_crossval_molname_fingerprint_dict, scaled_xtest_molname_fingerprint_dict
    else:
        return scaled_crossval_molname_fingerprint_dict