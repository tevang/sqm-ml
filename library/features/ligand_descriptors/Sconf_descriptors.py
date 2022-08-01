import pandas as pd
from lib.global_fun import get_structvar

def calc_Sconf_descriptors(energies_csv):
    return pd.read_csv(energies_csv) \
        .assign(structvar=lambda df: df['molname'].apply(get_structvar).str.lower()) \
        .groupby('structvar', as_index=False) \
        .apply(lambda g:
               g.assign(DeltaG=g['Energy:']-g['Energy:'].min(),
                        DeltaG_0to1=lambda gg: gg[(0 <= gg['DeltaG']) & (gg['DeltaG'] < 1)].shape[0],
                        DeltaG_1to2=lambda gg: gg[(1 <= gg['DeltaG']) & (gg['DeltaG'] < 2)].shape[0],
                        DeltaG_2to3=lambda gg: gg[(2 <= gg['DeltaG']) & (gg['DeltaG'] < 3)].shape[0],
                        DeltaG_3to4=lambda gg: gg[(3 <= gg['DeltaG']) & (gg['DeltaG'] < 4)].shape[0],
                        DeltaG_4to5=lambda gg: gg[(4 <= gg['DeltaG']) & (gg['DeltaG'] < 5)].shape[0],
                        DeltaG_5to6=lambda gg: gg[(5 <= gg['DeltaG']) & (gg['DeltaG'] < 6)].shape[0]
                        )).drop(columns=['molname', 'Energy:', 'DeltaG']).drop_duplicates()