import pandas as pd
from collections import Counter

def get_bond_types_count_as_vector(mol):
    
    bond_dict = \
        {'bondType_THREECENTER': [0], 'bondType_UNSPECIFIED': [0], 'bondType_OTHER': [0], 'bondType_HYDROGEN': [0],
         'bondType_DATIVER': [0], 'bondType_QUADRUPLE': [0], 'bondType_TWOANDAHALF': [0], 'bondType_DATIVEL': [0],
         'bondType_HEXTUPLE': [0], 'bondType_ONEANDAHALF': [0], 'bondType_DATIVE': [0], 'bondType_ZERO': [0],
         'bondType_FOURANDAHALF': [0], 'bondType_AROMATIC': [0], 'bondType_QUINTUPLE': [0], 'bondType_TRIPLE': [0],
         'bondType_DATIVEONE': [0], 'bondType_IONIC': [0], 'bondType_FIVEANDAHALF': [0], 'bondType_DOUBLE': [0],
         'bondType_SINGLE': [0], 'bondType_THREEANDAHALF': [0]}

    return pd.DataFrame(data={**{'structvar': [mol.GetProp('_Name')]}, **bond_dict,
                              **{'bondType_'+str(k): [v] for k, v in
                              Counter([b.GetBondTypeAsDouble() for b in mol.GetBonds()]).items()}})