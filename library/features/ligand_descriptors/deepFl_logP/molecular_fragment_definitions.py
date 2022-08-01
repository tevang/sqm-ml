from rdkit import Chem


def fr_Trithiolane(mol):
    patt = Chem.MolFromSmarts('[#16]-[#6]-1-[#6]-[#16]-[#16]-[#16]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Dithiolane(mol):
    patt = Chem.MolFromSmarts('[#16]-[#6]-1-[#6]-[#16]-[#16]-[#16]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_DiSulf(mol):
    patt = Chem.MolFromSmarts('[#16](-[#16]-[*])-[*]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Chlorobenzene(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[Cl]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_dichlorobenzene12(mol):
    patt = Chem.MolFromSmarts('c1ccc(c(c1)-[Cl])-[Cl]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_dichlorobenzene14(mol):
    patt = Chem.MolFromSmarts('c1cc(ccc1-[Cl])-[Cl]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_C4(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#6]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_C8(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]-[#6]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_propanoic_SMART(mol):
    patt = Chem.MolFromSmarts('[#8]-[#6]-[#6]-[#6](=[#8])-[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_propanoic(mol):
    patt = Chem.MolFromSmarts('[#8]-[#6]-[#6]-[#6](=[#8])-[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_urea(mol):
    patt = Chem.MolFromSmarts('[#6](-[#7])(=[#8])-[#7]-[#6]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_so2(mol):
    patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])-[#8-]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_quinoline(mol):
    patt = Chem.MolFromSmarts('c1ccc2c(c1)cccn2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_cyclpropnitro(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#7]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Thiadiazole(mol):
    patt = Chem.MolFromSmarts('c1nncs1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_dioxolane(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#8]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Pyrazinedihydro(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#7]=[#6]-[#6]=[#7]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_pyridine(mol):
    patt = Chem.MolFromSmarts('c1ccncc1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_betalactone(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-1=[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_nitrosomorpholine(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#6]-[#7]-1-[#7]=[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_dichlorodioxane(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6](-[#6](-[#8]-1)-[Cl])-[Cl]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_dioxane(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#6]-[#8]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_metTHF(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6]-1-[#6]-[#6]-[#6]-[#8]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_4pyranone(mol):
    patt = Chem.MolFromSmarts('[#6]-1=[#6]-[#8]-[#6]=[#6]-[#6]-1=[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_THF(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]-[#8]-[#6]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Dimethoxyethane(mol):
    patt = Chem.MolFromSmarts('[#6]-[#8]-[#6]-[#6]-[#8]-[#6]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Dihydropyran(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]=[#6]-[#8]-[#6]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Tetrahydropyran(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]-[#8]-[#6]-[#6]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Isoxazole(mol):
    patt = Chem.MolFromSmarts('c1conc1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Thiazole(mol):
    patt = Chem.MolFromSmarts('c1cscn1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Pyridazine(mol):
    patt = Chem.MolFromSmarts('c1ccnnc1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Pyrimidine(mol):
    patt = Chem.MolFromSmarts('c1cncnc1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Pyrazine(mol):
    patt = Chem.MolFromSmarts('c1cnccn1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Dimethylisoxazole(mol):
    patt = Chem.MolFromSmarts('[#6]-c1cc(no1)-[#6]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Oxetane(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Arginine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](-[#6](=[#8])-[#7])-[#7])-[#6]-[#7]=[#6](-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Proline(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6](-[#7]-[#6]-1)-[#6](=[#8])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Tryptophane(mol):
    patt = Chem.MolFromSmarts('c1ccc2c(c1)c(cn2)-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Alanine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Lysine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6]-[#6]-[#7])-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Phenylalanine(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Tyrosine(mol):
    patt = Chem.MolFromSmarts('c1cc(ccc1-[#6]-[#6](-[#6](=[#8])-[#7])-[#7])-[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Methionine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#16]-[#6]-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Leucine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6](-[#6])-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Isoleucine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6]-[#6](-[#6])-[#6](-[#6](=[#8])-[#8])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Valine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6](-[#6])-[#6](-[#6](=[#8])-[#8])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Glutamate(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](=[#8])-[#8])-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Glutamine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](=[#8])-[#7])-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Aspartate(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#8])-[#7])-[#6](=[#8])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Glycine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Histidine(mol):
    patt = Chem.MolFromSmarts('c1c(ncn1)-[#6]-[#6](-[#6](=[#8])-[#8])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Serine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Threonine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6](-[#6](-[#6](=[#8])-[#8])-[#7])-[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Asparagine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#6](=[#8])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Cysteine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#16]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Chromene(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]=[#6]-c1ccccc1-[#8]-2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Chromene_2(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#6]-c1ccccc1-[#8]-2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Chromane(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]-c1ccccc1-[#8]-[#6]-2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Chromanone(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#8]-c1ccccc1-[#6]-2=[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Chromone_2(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#8]-c1ccccc1-[#6]-2=[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Furan_2(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]=[#6]-[#6]-[#8]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Oxazoline(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]=[#7]-1')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Nitrobenzene(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[#7+](=[#8])-[#8-]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Thiophene_N(mol):
    patt = Chem.MolFromSmarts('[#7]-2-c1ccccc1-[#16]-[#6]-2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Quinolonium(mol):
    patt = Chem.MolFromSmarts('[#6]-c3cc[n+]c4ccccc34')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Benzimidazole(mol):
    patt = Chem.MolFromSmarts('c1ccc2c(c1)ncn2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Chlorzoxazone(mol):
    patt = Chem.MolFromSmarts('c1cc-2c(cc1-[Cl])-[#7]-[#6](=[#8])-[#8]-2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Naphthalene(mol):
    patt = Chem.MolFromSmarts('c2ccc1ccccc1c2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Hbond(mol):
    patt = Chem.MolFromSmarts('[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_quatNwC(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#7+](-[#6])(-[#6])-[#6]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Ammoniopropyl(mol):
    patt = Chem.MolFromSmarts('[#6]-[#7+](-[#6])-[#6]-[#6]-[#6]-[#7+]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Aniline(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_ACA(mol):
    patt = Chem.MolFromSmarts('[*]-[#6]-[*]-[#6]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_ANH(mol):
    patt = Chem.MolFromSmarts('[*]-[#7]-[#1]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_AN(mol):
    patt = Chem.MolFromSmarts('[*]-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_COdbO(mol):
    patt = Chem.MolFromSmarts('[#6]=[#6]=[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_CdbNdbO(mol):
    patt = Chem.MolFromSmarts('[#6]=[#7+]=[#8-]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_CdbN(mol):
    patt = Chem.MolFromSmarts('[#6]=[#7+]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_CN(mol):
    patt = Chem.MolFromSmarts('[#6]-[#7+]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_NSO(mol):
    patt = Chem.MolFromSmarts('[#7]=[#16]=[#8]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_NN(mol):
    patt = Chem.MolFromSmarts('[#7](=[#7+]=[*])-[*]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Etsub(mol):
    patt = Chem.MolFromSmarts('c1cc[n+](cc1)-[*]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_ringNO(mol):
    patt = Chem.MolFromSmarts('[n+]1(c(oc2ccccc12)-[*])-[*]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Etplus(mol):
    patt = Chem.MolFromSmarts('[n+]3(c1cc(ccc1c2ccc(cc2c3)-[#7])-[#7])-[*]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_benzoicsulfonic(mol):
    patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])(-[*])-[*]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Etroot(mol):
    patt = Chem.MolFromSmarts(
        '[#6]-[#7+](-[#6])(-[#6]-[#6]-[#6]-[n+]1ccccc1)-[#6]-[#6]-[#6]-[#7+](-[#6])(-[#6])-[#6]-[#6]-[#6]-[n+]2ccccc2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Benzoxazolium(mol):
    patt = Chem.MolFromSmarts('[n+]1coc2ccccc12')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Benzoxazol(mol):
    patt = Chem.MolFromSmarts('[n]1coc2ccccc12')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_Dapi(mol):
    patt = Chem.MolFromSmarts('c1cc(ccc1-[*])-[#6](=[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_HemiBabim(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[#6](=[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def fr_HemiBabim_2(mol):
    patt = Chem.MolFromSmarts('c1nc2c(n1)cccc2')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


# counting fragments

def ct_Trithiolane(mol):
    patt = Chem.MolFromSmarts('[#16]-[#6]-1-[#6]-[#16]-[#16]-[#16]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Dithiolane(mol):
    patt = Chem.MolFromSmarts('[#16]-[#6]-1-[#6]-[#16]-[#16]-[#16]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_DiSulf(mol):
    patt = Chem.MolFromSmarts('[#16](-[#16]-[*])-[*]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Chlorobenzene(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[Cl]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_dichlorobenzene12(mol):
    patt = Chem.MolFromSmarts('c1ccc(c(c1)-[Cl])-[Cl]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_dichlorobenzene14(mol):
    patt = Chem.MolFromSmarts('c1cc(ccc1-[Cl])-[Cl]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_propanoic(mol):
    patt = Chem.MolFromSmarts('[#8]-[#6]-[#6]-[#6](=[#8])-[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_urea(mol):
    patt = Chem.MolFromSmarts('[#6](-[#7])(=[#8])-[#7]-[#6]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


# new ends here

def ct_so2(mol):
    patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])-[#8-]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_quinoline(mol):
    patt = Chem.MolFromSmarts('c1ccc2c(c1)cccn2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_cyclpropnitro(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#7]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Thiadiazole(mol):
    patt = Chem.MolFromSmarts('c1nncs1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_dioxolane(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#8]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Pyrazinedihydro(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#7]=[#6]-[#6]=[#7]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_pyridine(mol):
    patt = Chem.MolFromSmarts('c1ccncc1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_betalactone(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-1=[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_nitrosomorpholine(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#6]-[#7]-1-[#7]=[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_dichlorodioxane(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6](-[#6](-[#8]-1)-[Cl])-[Cl]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_dioxane(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-[#6]-[#8]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_metTHF(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6]-1-[#6]-[#6]-[#6]-[#8]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_4pyranone(mol):
    patt = Chem.MolFromSmarts('[#6]-1=[#6]-[#8]-[#6]=[#6]-[#6]-1=[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_THF(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]-[#8]-[#6]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Dimethoxyethane(mol):
    patt = Chem.MolFromSmarts('[#6]-[#8]-[#6]-[#6]-[#8]-[#6]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Dihydropyran(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]=[#6]-[#8]-[#6]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Tetrahydropyran(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6]-[#8]-[#6]-[#6]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Isoxazole(mol):
    patt = Chem.MolFromSmarts('c1conc1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Thiazole(mol):
    patt = Chem.MolFromSmarts('c1cscn1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Pyridazine(mol):
    patt = Chem.MolFromSmarts('c1ccnnc1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Pyrimidine(mol):
    patt = Chem.MolFromSmarts('c1cncnc1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Pyrazine(mol):
    patt = Chem.MolFromSmarts('c1cnccn1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Dimethylisoxazole(mol):
    patt = Chem.MolFromSmarts('[#6]-c1cc(no1)-[#6]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Oxetane(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Arginine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](-[#6](=[#8])-[#7])-[#7])-[#6]-[#7]=[#6](-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Proline(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#6](-[#7]-[#6]-1)-[#6](=[#8])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Tryptophane(mol):
    patt = Chem.MolFromSmarts('c1ccc2c(c1)c(cn2)-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Alanine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Lysine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6]-[#6]-[#7])-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Phenylalanine(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Tyrosine(mol):
    patt = Chem.MolFromSmarts('c1cc(ccc1-[#6]-[#6](-[#6](=[#8])-[#7])-[#7])-[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Methionine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#16]-[#6]-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Leucine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6](-[#6])-[#6]-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Isoleucine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6]-[#6](-[#6])-[#6](-[#6](=[#8])-[#8])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Valine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6](-[#6])-[#6](-[#6](=[#8])-[#8])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Glutamate(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](=[#8])-[#8])-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Glutamine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6]-[#6](=[#8])-[#7])-[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Aspartate(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#8])-[#7])-[#6](=[#8])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Glycine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](=[#8])-[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Histidine(mol):
    patt = Chem.MolFromSmarts('c1c(ncn1)-[#6]-[#6](-[#6](=[#8])-[#8])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Serine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Threonine(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6](-[#6](-[#6](=[#8])-[#8])-[#7])-[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Asparagine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#6](=[#8])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Cysteine(mol):
    patt = Chem.MolFromSmarts('[#6](-[#6](-[#6](=[#8])-[#7])-[#7])-[#16]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Chromene(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]=[#6]-c1ccccc1-[#8]-2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Chromene_2(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#6]-c1ccccc1-[#8]-2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Chromane(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]-c1ccccc1-[#8]-[#6]-2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Chromanone(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#8]-c1ccccc1-[#6]-2=[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Chromone_2(mol):
    patt = Chem.MolFromSmarts('[#6]-2-[#6]-[#8]-c1ccccc1-[#6]-2=[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Furan_2(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]=[#6]-[#6]-[#8]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Oxazoline(mol):
    patt = Chem.MolFromSmarts('[#6]-1-[#6]-[#8]-[#6]=[#7]-1')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Nitrobenzene(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[#7+](=[#8])-[#8-]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Thiophene_N(mol):
    patt = Chem.MolFromSmarts('[#7]-2-c1ccccc1-[#16]-[#6]-2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Quinolonium(mol):
    patt = Chem.MolFromSmarts('[#6]-c3cc[n+]c4ccccc34')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Benzimidazole(mol):
    patt = Chem.MolFromSmarts('c1ccc2c(c1)ncn2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Chlorzoxazone(mol):
    patt = Chem.MolFromSmarts('c1cc-2c(cc1-[Cl])-[#7]-[#6](=[#8])-[#8]-2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Naphthalene(mol):
    patt = Chem.MolFromSmarts('c2ccc1ccccc1c2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Hbond(mol):
    patt = Chem.MolFromSmarts('[O,N;!H0]-*~*-*=[$([C,N;R0]=O)]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_quatNwC(mol):
    patt = Chem.MolFromSmarts('[#6]-[#6]-[#6]-[#7+](-[#6])(-[#6])-[#6]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Ammoniopropyl(mol):
    patt = Chem.MolFromSmarts('[#6]-[#7+](-[#6])-[#6]-[#6]-[#6]-[#7+]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Aniline(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_ACA(mol):
    patt = Chem.MolFromSmarts('[*]-[#6]-[*]-[#6]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_ANH(mol):
    patt = Chem.MolFromSmarts('[*]-[#7]-[#1]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_AN(mol):
    patt = Chem.MolFromSmarts('[*]-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_COdbO(mol):
    patt = Chem.MolFromSmarts('[#6]=[#6]=[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_CdbNdbO(mol):
    patt = Chem.MolFromSmarts('[#6]=[#7+]=[#8-]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_CdbN(mol):
    patt = Chem.MolFromSmarts('[#6]=[#7+]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_CN(mol):
    patt = Chem.MolFromSmarts('[#6]-[#7+]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_NSO(mol):
    patt = Chem.MolFromSmarts('[#7]=[#16]=[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_NN(mol):
    patt = Chem.MolFromSmarts('[#7](=[#7+]=[*])-[*]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Etsub(mol):
    patt = Chem.MolFromSmarts('c1cc[n+](cc1)-[*]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_ringNO(mol):
    patt = Chem.MolFromSmarts('[n+]1(c(oc2ccccc12)-[*])-[*]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Etplus(mol):
    patt = Chem.MolFromSmarts('[n+]3(c1cc(ccc1c2ccc(cc2c3)-[#7])-[#7])-[*]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_benzoicsulfonic(mol):
    patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])(-[*])-[*]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Etroot(mol):
    patt = Chem.MolFromSmarts(
        '[#6]-[#7+](-[#6])(-[#6]-[#6]-[#6]-[n+]1ccccc1)-[#6]-[#6]-[#6]-[#7+](-[#6])(-[#6])-[#6]-[#6]-[#6]-[n+]2ccccc2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Benzoxazolium(mol):
    patt = Chem.MolFromSmarts('[n+]1coc2ccccc12')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Benzoxazol(mol):
    patt = Chem.MolFromSmarts('[n]1coc2ccccc12')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_Dapi(mol):
    patt = Chem.MolFromSmarts('c1cc(ccc1-[*])-[#6](=[#7])-[#7]')
    x = mol.HasSubstructMatch(patt)
    x = int(x == True)
    return x


def ct_HemiBabim(mol):
    patt = Chem.MolFromSmarts('c1ccc(cc1)-[#6](=[#7])-[#7]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_HemiBabim_2(mol):
    patt = Chem.MolFromSmarts('c1nc2c(n1)cccc2')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_phosphorous(mol):
    patt = Chem.MolFromSmarts('[P]-[#8]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_phosphorous_negative(mol):
    patt = Chem.MolFromSmarts('[P]-[#8-]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_so2neutral(mol):
    patt = Chem.MolFromSmarts('[#16](=[#8])(=[#8])')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x


def ct_fluorobenzene(mol):
    patt = Chem.MolFromSmarts('c1cc(c(c(c1-[F])-[F])-[F])-[F]')
    x = mol.GetSubstructMatches(patt)
    x = len(x)
    return x