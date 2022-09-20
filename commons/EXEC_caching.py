import os

import pandas as pd


def EXEC_caching_decorator(lg, log_text, cache_csv_suffix, **outter_kwargs):
    def wrap(f):
        def wrapped_f(*args, **kwargs):
            lg.info(log_text)
            if 'Settings' not in kwargs:
                raise(Exception("'Settings' must be the keyword argument of every EXEC box!"))

            Settings = kwargs['Settings']

            if 'force_computation' in outter_kwargs.keys():
                fc = outter_kwargs['force_computation']
            else:
                fc = Settings.HYPER_FORCE_COMPUTATION

            if 'full_csv_name' in outter_kwargs.keys():

                if outter_kwargs['full_csv_name']:
                    csv_file_basename = cache_csv_suffix
                    if 'append_signature' in outter_kwargs.keys() and outter_kwargs['append_signature']:
                        csv_file_basename = cache_csv_suffix + Settings.create_feature_csv_name()
                    if 'prepend_protein' in outter_kwargs.keys() and outter_kwargs['prepend_protein']:
                        csv_file_basename = Settings.HYPER_PROTEIN + csv_file_basename
                    if 'prepend_all_proteins' in outter_kwargs.keys() and outter_kwargs['prepend_all_proteins']:
                        # csv_file_basename = "_".join(Settings.ALL_PROTEINS) + csv_file_basename
                        csv_file_basename = "%i_proteins" % len(Settings.ALL_PROTEINS) + csv_file_basename
                    csv_file = Settings.HYPER_SQM_ML_ROOT_DIR + "/" + Settings.HYPER_EXECUTION_DIR_NAME + "/" + csv_file_basename
                else:
                    csv_file = Settings.generated_file(cache_csv_suffix)
            else:
                csv_file = Settings.generated_file(cache_csv_suffix)

            if os.path.exists(csv_file) and (not fc):
                lg.warning("Reading the resulting dataframe from " + csv_file)
                return pd.read_csv(csv_file)
            else:
                res = f(*args, **kwargs)
                res.to_csv(csv_file, index=False, sep=",", float_format='%g')
                return res

        return wrapped_f

    return wrap
