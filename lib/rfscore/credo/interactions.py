from openbabel import OBAtomAtomIter, OBAtomBondIter, OBElementTable
from rfscore.config import config, contact_types, logger

ET = OBElementTable()

# CREATE CONSTANT FOR THE HYDROGEN VDW RADIUS
VDW_HYDROGEN = ET.GetVdwRad(1)

# INDICES OF THE ATOM TYPE ARRAY
DONOR = 0
ACCEPTOR = 1
AROMATIC = 2
WEAK_ACC = 3
WEAK_DON = 4
HYDROPHOBE = 5
METAL = 6
POS_IONISABLE = 7
NEG_IONISABLE = 8
XBOND_DONOR = 9
XBOND_ACC = 10
CBNYL_O = 11
CBNYL_C = 12

def sum_vdw_radii(hetatm, atom):
    '''
    Returns the sum of the Van der Waals radii of the two atoms.
    '''
    return ET.GetVdwRad(hetatm.GetAtomicNum()) + ET.GetVdwRad(atom.GetAtomicNum())

def hydrogen_nbrs(atom):
    '''
    Returns an iterator over all hydrogens of an atom. Required to calculate hydrogen
    bonds.
    '''
    for nbr in OBAtomAtomIter(atom):
        if nbr.IsHydrogen():
            yield nbr

def single_bond_nbr(atom):
    '''
    '''
    for bond in OBAtomBondIter(atom):
        if bond.IsSingle():
            return bond.GetNbrAtom(atom)

def is_hbond(hetatm, hetatm_types, atom, atom_types, distance):
    '''
    '''
    def _is_hbond(donor, acceptor):
        '''
        '''
        # GET ALL HYDROGENS ATTACHED TO THE DONOR
        for hydrogen in hydrogen_nbrs(donor):
            if acceptor.GetDistance(hydrogen) <= VDW_HYDROGEN + ET.GetVdwRad(acceptor.GetAtomicNum()) + config['VDW comp factor']:

                # CALCULATE THE ANGLE BETWEEN D-H--->A
                if hydrogen.GetAngle(donor, acceptor) <= contact_types['hbond']['angle']:
                    return True

        return False

    IS_HBOND = False

    # HETATM IS ACCEPTOR / PROTEIN ATOM IS DONOR
    if hetatm_types.get("hbond acceptor") and atom_types[DONOR]:
        IS_HBOND = _is_hbond(atom,hetatm)

    # LIGAND ATOM IS DONOR
    elif hetatm_types.get("hbond donor") and atom_types[ACCEPTOR]:
        IS_HBOND = _is_hbond(hetatm,atom)

    return IS_HBOND

def is_weak_hbond(hetatm, hetatm_types, atom, atom_types, distance):
    '''
    '''
    def _is_weak_hbond(donor, acceptor):
        '''
        '''
        for hydrogen in hydrogen_nbrs(donor):

            # CHECK FOR BOND-LIKE CHARACTER
            if acceptor.GetDistance(hydrogen) <= VDW_HYDROGEN + ET.GetVdwRad(acceptor.GetAtomicNum()) + config['VDW comp factor']:

                # CALCULATE THE ANGLE BETWEEN D-H--->A
                if hydrogen.GetAngle(donor, acceptor) <= contact_types['weak hbond']['angle']:
                    return True

        return False

    IS_WEAK_HBOND = False

    # HETATM IS ACCEPTOR / PROTEIN ATOM IS WEAK DONOR
    if hetatm_types.get("hbond acceptor") and atom_types[WEAK_DON]:
        IS_WEAK_HBOND = _is_weak_hbond(atom,hetatm)

    # HETATM IS WEAK DONOR / PROTEIN IS ACCEPTOR
    elif hetatm_types.get("weak hbond donor") and atom_types[ACCEPTOR]:
        IS_WEAK_HBOND = _is_weak_hbond(hetatm,atom)

    return IS_WEAK_HBOND

def is_xbond(hetatm, hetatm_types, atom, atom_types, distance):
    '''
    Halogens can form electrostatic interactions with Lewis-bases (nucleophiles)
    in a head-on orientation.
    '''
    if distance <= sum_vdw_radii(hetatm,atom) + config['VDW comp factor']:

        # ONLY HETATM CAN BE XBOND DONOR
        if hetatm_types.get('xbond donor') and atom_types[XBOND_DONOR]:

            # ATOM ATTACHED TO HALOGEN - IMPORTANT FOR THE IDENTIFICATION OF THE HEAD-ON ORIENTATION
            nbr = single_bond_nbr(hetatm)

            # ANGLE BETWEEN NEIGHBOURING ATOM, HALOGEN AND ACCEPTOR
            theta = nbr.GetAngle(hetatm,atom)

            # ANGLE FOR THE HEAD-ON ORIENTATION
            if theta >= contact_types['xbond']['angle theta 1']:

                # DEBUG HALOGEN BOND INFO / RARE CONTACT TYPE
                logger.info("Halogen bond found.")

                return True

def is_ionic(hetatm, hetatm_types, atom, atom_types, distance):
    '''
    '''
    IS_IONIC = False

    if distance <= contact_types['ionic']['distance']:
        if hetatm_types.get("neg ionisable") and atom_types[POS_IONISABLE]: IS_IONIC = True
        elif hetatm_types.get("pos ionisable") and atom_types[NEG_IONISABLE]: IS_IONIC = True

    return IS_IONIC

def is_metal_complex(hetatm, hetatm_types, atom, atom_types, distance):
    '''
    '''
    IS_METAL_COMPLEX = False

    if distance <= contact_types['metal']['distance']:
        if hetatm.MatchesSMARTS("[Ca,Fe,Mg,Mn,Ni,Zn]") and atom_types[ACCEPTOR]:
            IS_METAL_COMPLEX = True

        # CASES WHERE METAL IONS CAN BE FOUND IN THE BINDING SITE
        elif atom.MatchesSMARTS("[Ca,Fe,Mg,Mn,Ni,Zn]") and hetatm_types.get('acceptor'):
            IS_METAL_COMPLEX = True

    return IS_METAL_COMPLEX

def is_aromatic(hetatm, hetatm_types, atom, atom_types, distance):
    '''
    '''
    if distance <= contact_types['aromatic']['distance']:
        if hetatm.IsAromatic() and atom.IsAromatic():
            return True

def is_hydrophobic(hetatm, hetatm_types, atom, atom_types, distance):
    '''
    '''
    if distance <= contact_types['hydrophobic']['distance']:
        if hetatm_types.get('hydrophobe') and atom_types[HYDROPHOBE]:
            return True

def is_carbonyl(hetatm, hetatm_types, atom, atom_types, distance):
    '''
    '''
    IS_CARBONYL = False

    if distance <= contact_types['carbonyl']['distance']:
        if hetatm_types.get('carbonyl carbon') and atom_types[CBNYL_O]: IS_CARBONYL = True
        elif hetatm_types.get('carbonyl oxygen') and atom_types[CBNYL_C]: IS_CARBONYL = True

    return IS_CARBONYL