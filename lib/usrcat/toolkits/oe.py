"""
The usrcat function in this module is implemented with the help of OpenEye's
OEChem toolkit.
"""

from numpy import append, array, zeros, vstack
from openeye.oechem import OEGraphMol, OEMol, OEMatchAtom, OESuppressHydrogens

from usrcat import PHARMACOPHORES, NUM_MOMENTS
from usrcat.geometry import usr_moments, usr_moments_with_existing

def generate_moments(molecule, hydrogens=False):
    """
    Returns a 2D array of USRCAT moments for all conformers of the input molecule.
    """
    # an OEMol object is required to iterate through conformers
    if isinstance(molecule, OEGraphMol): molecule = OEMol(molecule)

    # ignore hydrogens
    if not hydrogens: OESuppressHydrogens(molecule)

    # initial zeroed array to use with vstack - will be discarded eventually
    moments = zeros(NUM_MOMENTS, dtype=float)

    # create an atom subset for each pharmacophore definition
    subsets = [molecule.GetAtoms(OEMatchAtom(smarts)) for smarts in PHARMACOPHORES]

    # iterate through conformers and generate USRCAT moments for each
    for conformer in molecule.GetConfs():

        # get all coordinates
        coords = conformer.GetCoords()

        # generate the four reference points and USR moments for all atoms
        (ctd,cst,fct,ftf), om = usr_moments(array(list(coords.values())))

        # generate the USR moments for the feature specific coordinates
        for subset in subsets:

            # only keep the atomic coordinates of the subset
            fcoords = array([coords.get(atom.GetIdx()) for atom in subset])

            # initial zeroed out USRCAT feature moments
            fm = zeros(12)

            # only attempt to generate moments if there are enough atoms available!
            if len(fcoords):
                fm = usr_moments_with_existing(fcoords, ctd, cst, fct, ftf)

            # append feature moments to the existing ones
            om = append(om, fm)

        # add conformer USRCAT moments to array for this molecule
        moments = vstack((moments, om))

    # do not include first row: all zeros!
    return moments[1:]