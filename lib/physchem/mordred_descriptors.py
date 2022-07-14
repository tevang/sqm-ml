from mordred import descriptors

mordred_dfunc = {   # MORDRED descripor functions
    'nRot': descriptors.RotatableBond.RotatableBondsCount,
    'nAtom': descriptors.AtomCount.AtomCount("Atom"),
    'nHeavyAtom': descriptors.AtomCount.AtomCount("HeavyAtom"),
    'nSpiro': descriptors.AtomCount.AtomCount("Spiro"),
    'nBridgehead': descriptors.AtomCount.AtomCount("Bridgehead"),
    'nHetero': descriptors.AtomCount.AtomCount("Hetero"),
    'SlogP_VSA1' : descriptors.MoeType.SlogP_VSA(k=1),
    'SlogP_VSA2' : descriptors.MoeType.SlogP_VSA(k=2),
    'SlogP_VSA3' : descriptors.MoeType.SlogP_VSA(k=3),
    'SlogP_VSA4' : descriptors.MoeType.SlogP_VSA(k=4),
    'SlogP_VSA5' : descriptors.MoeType.SlogP_VSA(k=5),
    'SlogP_VSA6' : descriptors.MoeType.SlogP_VSA(k=6),
    'SlogP_VSA7' : descriptors.MoeType.SlogP_VSA(k=7),
    'SlogP_VSA8' : descriptors.MoeType.SlogP_VSA(k=8),
    'SlogP_VSA9' : descriptors.MoeType.SlogP_VSA(k=9),
    'SlogP_VSA10' : descriptors.MoeType.SlogP_VSA(k=10),
    'SlogP_VSA11' : descriptors.MoeType.SlogP_VSA(k=11),
    'SLogP': descriptors.SLogP.SLogP,
    'MW': descriptors.Weight.Weight(True, False),
    'AMW': descriptors.Weight.Weight(True, True),
}