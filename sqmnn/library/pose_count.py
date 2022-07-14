import pandas as pd


def get_total_pose_num(sqm_df, glide_df):
    COLS = ['basemolname', 'stereoisomer', 'ionstate', 'tautomer']
    structvar_posenum_df = sqm_df[COLS].drop_duplicates()

    def _get_total_pose_num_from_df(r):
        r_df = pd.DataFrame(data=[r])
        return r_df[COLS].to_records(index=False)[0], pd.merge(glide_df, r_df, on=COLS,
                 indicator=True).shape[0]

    structvar_totalposenum_dict = {tuple(k):v for k,v in structvar_posenum_df.apply(_get_total_pose_num_from_df, axis=1)}

    def _get_total_pose_num_from_dict(r):
        return structvar_totalposenum_dict[tuple(r[COLS])]

    return _get_total_pose_num_from_dict


def get_scored_pose_num(sqm_df):
    COLS = ['basemolname', 'stereoisomer', 'ionstate', 'tautomer']
    structvar_scoredposenum_dict = sqm_df.groupby(COLS) \
        .apply(lambda g: g.shape[0]).to_dict()

    def _get_scored_pose_num_from_dict(r):
        return structvar_scoredposenum_dict[tuple(r[COLS])]

    return _get_scored_pose_num_from_dict