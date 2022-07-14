import os
import sys
import warnings

from numpy import append, array, zeros, vstack

# Import ConsScorTK libraries
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )  # import the top level package directory
from lib.usrcat import PHARMACOPHORES, NUM_MOMENTS
from lib.usrcat.geometry import usr_moments, usr_moments_with_existing
#import pybel
from lib.modlib import pybel


def generate_moments(molecule, hydrogens=False):
    """
    Returns a 2D array of USRCAT moments for all conformers of the input molecule.

    :param molecule: a pybel Molecule object with a single conformer.
    :param hydrogens: if True, then the coordinates of hydrogen atoms will be
                      included in the moment generation.
    """
    # how to suppress hydrogens?
    if not hydrogens:
        molecule.removeh()

    # initial zeroed array to use with vstack - will be discarded eventually
    moments = zeros(NUM_MOMENTS, dtype=float)

    # initialize SMARTS patterns as Mol objects
    patterns = [pybel.Smarts(smarts) for smarts in PHARMACOPHORES]

    # create an atom idx subset for each pharmacophore definition
    subsets = []
    for pattern in patterns:

        # get a list of atom identifiers that match the pattern (if any)
        #Pybel returns atom indices as tuples, get the first one
        matches =  pattern.findall(molecule)

        # append this list of atoms for this pattern to the subsets
        if matches:
            subsets.extend(list(zip(*matches)))
        else:
            subsets.append([])

    #Only deals with ONE conformer
    # get the coordinates of all atoms
    if molecule.OBMol.NumConformers() > 1:
        warnings.warn("Multiple conformers found,"
                      " All but the first will be ignored.")
    coords = {}
    for atom in molecule:
        coords[atom.idx] = atom.coords

    # generate the four reference points and USR moments for all atoms
    (ctd,cst,fct,ftf), om = usr_moments(array(list(coords.values())))

    # generate the USR moments for the feature specific coordinates
    for subset in subsets:

        # only keep the atomic coordinates of the subset
        fcoords = array([coords.get(atomidx) for atomidx in subset])

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
