"""
This module contains the usrcat function that uses the RDKit Python bindings.
"""
import os
import sys

import oddt
from numpy import append, array, zeros, vstack, dot, argsort, linalg, sqrt, arctan2, mean
from oddt.shape import *
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

from lib import PLUMED_functions

oddt.toolkit = oddt.toolkits.rdk  # force ODDT to use RDKit

# Import ConsScorTK libraries
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )  # import the top level package directory
from lib.usrcat import PHARMACOPHORES
from lib.usrcat.geometry import usr_moments, usr_moments_with_existing


def cartesian2spherical(xyz):
    """
        FUNCTION to convert a 1x3 array "xyz" containing the cartesian to spherical coordinates. 
    """
    spherical = zeros(3)
    xy = xyz[0]**2 + xyz[1]**2
    spherical[0] = sqrt(xy + xyz[2]**2)
    # print("DEBUG: xy=", xy, "xyz=", xyz.tolist())
    spherical[1] = arctan2(sqrt(xy), xyz[2]) # for elevation angle defined from Z-axis down
    #spherical[:,2] = arctan2(xyz[:,2], sqrt(xy)) # for elevation angle defined from XY-plane up
    spherical[2] = arctan2(xyz[1], xyz[0])
    return spherical


def calc_principal_axes(coords, ctd):
    # center with geometric center
    coords = coords - ctd
    
    # compute principal axis matrix
    inertia = dot(coords.transpose(), coords)
    e_values, e_vectors = linalg.eig(inertia)
    # warning eigen values are not necessary ordered!
    #--------------------------------------------------------------------------
    # order eigen values (and eigen vectors)
    #
    # axis1 is the principal axis with the biggest eigen value (eval1)
    # axis2 is the principal axis with the second biggest eigen value (eval2)
    # axis3 is the principal axis with the smallest eigen value (eval3)
    #--------------------------------------------------------------------------
    order = argsort(e_values)
    eval3, eval2, eval1 = e_values[order]
    axis3, axis2, axis1 = e_vectors[:, order].transpose()
    # Inertia axis are now ordered !
    sphaxis3 = cartesian2spherical(axis3)
    sphaxis2 = cartesian2spherical(axis2)
    sphaxis1 = cartesian2spherical(axis1)
    
    return (sphaxis3, sphaxis2, sphaxis1), (eval3, eval2, eval1)


def calculate_shape_param(coords, masses):
    
    """
    Calculates the gyration tensor of a structure.  
    Returns a tuple containing shape parameters:
    
      (gy_tensor, (a,b,c), rg, A)
      gy_tensor -   the gyration tensors (3x3 array) in spherical coordinates
      (a,b,c) - the dimensions of the smallest ellipsoid into which the input molecule can fit 
      rg 	  - radius of gyration of the structure
      A     - anisotropy value
    """
  
    # COM of the input coordinates
    # print("DEBUG: masses=", masses.tolist())
    # print("DEBUG: coords=", coords.tolist())
    # replicate mass values (Nx1) 3 times to match the coords array dimensions (Nx3)
    com = array(array([masses, masses, masses]).transpose() * coords).sum(axis=0) / masses.sum()
    cx, cy, cz = com
  
    n_atoms = 0
    tensor_xx, tensor_xy, tensor_xz = 0, 0, 0
    tensor_yx, tensor_yy, tensor_yz = 0, 0, 0
    tensor_zx, tensor_zy, tensor_zz = 0, 0, 0
  
    for coord in coords:
        ax, ay, az = coord
        tensor_xx += (ax-cx)*(ax-cx)
        tensor_yx += (ax-cx)*(ay-cy)
        tensor_xz += (ax-cx)*(az-cz)
        tensor_yy += (ay-cy)*(ay-cy)
        tensor_yz += (ay-cy)*(az-cz)
        tensor_zz += (az-cz)*(az-cz)
        n_atoms += 1
  
    gy_tensor =  array([[tensor_xx, tensor_yx, tensor_xz], [tensor_yx, tensor_yy, tensor_yz], [tensor_xz, tensor_yz, tensor_zz]])
    gy_tensor_spherical =  array([cartesian2spherical(array([tensor_xx, tensor_yx, tensor_xz])),
                                 cartesian2spherical(array([tensor_yx, tensor_yy, tensor_yz])),
                                 cartesian2spherical(array([tensor_xz, tensor_yz, tensor_zz]))])
    gy_tensor = (1.0/n_atoms) * gy_tensor
    
    D,V = linalg.eig(gy_tensor)
    [a, b, c] = sorted(sqrt(5 * D))
    rg = sqrt(sum(D))
    
    l = mean([D[0],D[1],D[2]])
    A = (((D[0] - l)**2 + (D[1] - l)**2 + (D[2] - l)**2) / l**2) * 1/6
    S = (((D[0] - l) * (D[1] - l) * (D[2] - l))/ l**3)
    
    print("%s" % '#Dimensions(a,b,c) #Rg #Anisotropy')
    print("%.2f" % round(2*a,2), round(2*b,2), round(2*c,2) , round(rg,2) , round(A,2))  # I doubled axes here!
  
    return (gy_tensor_spherical, a,b,c,rg,A)
    

def get_shape_params_vector(coords, masses):
    """
        FUNCTION to place all numbers return by calculate_shape_param into a vector.
    """
    
    gy_tensor_spherical, a,b,c,rg,A = calculate_shape_param(coords, masses)
    return array(gy_tensor_spherical.reshape([1,9]).tolist()[0] + [a,b,c,rg,A])


def generate_moments(molecule,
                     hydrogens=False,
                     moment_number=4,
                     onlyshape=False,
                     molname="",
                     SMILES="",
                     ensemble_mode=1,
                     plumed_colvars=False,
                     electro_shape=True):
    """
    Returns a 2D array of USRCAT moments for all conformers of the input molecule.

    ARGS:
    molecule: a rdkit Mol object that is expected to have conformers.
    hydrogens: if True, then the coordinates of hydrogen atoms will be
                      included in the moment generation.
    RETURNS:
    (molname, SMILES, [moments]):   where moments is a feature vector containing the moments of the current molecule. If the molecule object
                                    contains multiple conformers and ensemble_mode >=1, then the average feature vector is returned. If
                                    ensemble_mode=0 then the feature vector of the last conformer (last frame) is returned.
    """
    print("DEBUG: generate the moments of %i conformers of molecule %s" % (len([c for c in molecule.GetConformers()]), molname))
    sys.stdout.write(molname + " ")
    sys.stdout.flush()
    
    # how to suppress hydrogens?
    if not hydrogens: Chem.RemoveHs(molecule)

    # initial zeroed array to use with vstack - will be discarded eventually
    # if moment_number == 4:
        # print("DEBUG: NUM_MOMENTS=", NUM_MOMENTS)
    # total number of moments // +1 to include the all atom set
    if onlyshape:
        NUM_MOMENTS = moment_number * 8 + 9; # + 14 ; #+ 11
    else:
        # print("DEBUG: PHARMACOPHORES=", PHARMACOPHORES)
        NUM_MOMENTS = moment_number * 8 * (len(PHARMACOPHORES) + 1) + 9 * (len(PHARMACOPHORES) + 1); # + 14 * (len(PHARMACOPHORES) + 1) ; #+ 11
    if electro_shape:
        NUM_MOMENTS += 15
    moments = zeros(NUM_MOMENTS, dtype=float)
    # print("DEBUG: moments=", moments)

    # initialize SMARTS patterns as Mol objects
    patterns = [Chem.MolFromSmarts(smarts) for smarts in PHARMACOPHORES]

    # create an atom idx subset for each pharmacophore definition
    subsets = []
    for pattern in patterns:

        # get a list of atom identifiers that match the pattern (if any)
        matches = molecule.GetSubstructMatches(pattern)

        # append this list of atoms for this pattern to the subsets
        if matches: subsets.extend(list(zip(*matches)))
        else: subsets.append([])
    
    # find the scaffold of the molecule
    core = MurckoScaffold.GetScaffoldForMol(molecule)
    
    # iterate through conformers and generate USRCAT moments for each
    molecule_conformers = [c for c in molecule.GetConformers()]
    core_conformers = [c for c in core.GetConformers()]
    if ensemble_mode == 0:
        molecule_conformers = [molecule_conformers[-1]]   # keep only the last frame
        core_conformers = [core_conformers[-1]]
    for conformer, coreconf in zip(molecule_conformers, core_conformers):

        # get the coordinates of all atoms
        coords = {}
        masses = {}
        for atom in molecule.GetAtoms():
            point = conformer.GetAtomPosition(atom.GetIdx())
            masses[atom.GetIdx()] = atom.GetMass()
            coords[atom.GetIdx()] = (point.x, point.y, point.z)
        coords_array = array(list(coords.values()))
        masses_array = array(list(masses.values()))
        
        # get the coordinates of scaffold atoms
        corecoords = {}
        coremasses = {}
        for coreatom in core.GetAtoms():
            corepoint = coreconf.GetAtomPosition(coreatom.GetIdx())
            coremasses[coreatom.GetIdx()] = coreatom.GetMass()
            corecoords[coreatom.GetIdx()] = (corepoint.x, corepoint.y, corepoint.z)
        corecoords = array(list(corecoords.values()))
        coremasses = array(list(coremasses.values()))
        core_ctd = corecoords.mean(axis=0)
        # # replicate mass values (Nx1) 3 times to match the coords array dimensions (Nx3)
        # core_com = (array([coremasses, coremasses, coremasses]).transpose() * corecoords).sum(axis=0) / coremasses.sum()
        core_com = None
        
        # generate the four reference points and USR moments for all atoms
        # ORIGINAL line: (ctd,cst,fct,ftf,ftpp1,ctpp1,ftnp1,ctnp1,ftpp2,ctpp2,ftnp2,ctnp2), om = usr_moments(array(coords.values()), moment_number=moment_number)
        point_args, om = usr_moments(coords_array, masses_array, moment_number=moment_number, core_ctd=core_ctd, core_com=core_com)   # point_args can contain an arbitrary number of points
        # print("DEBUG: point 0 om=", om.shape)
        
        # append principal axes and their eigenvalues to the existing moments
        princ = array([])
        if len(coords_array) > 3:    # you need at least 4 atoms to calculate principal axes
            center = coords_array.mean(axis=0)
            principal_axes, eigenvalues = calc_principal_axes(coords_array, center)
            # print("DEBUG: coords_array=", coords_array.tolist())
            # print("DEBUG: principal_axes=", principal_axes)
            for axis,eigenval in zip(principal_axes, eigenvalues):
                princ = append(princ, axis)
                # princ = append(princ, eigenval)
            
            # ## Append Gyration tensor and shape parameters
            # princ = append(princ, get_shape_params_vector(coords_array, masses_array))
        else:   # if not enough atoms append zeros to the moments list
            princ = append(princ, zeros(9))
            # princ = append(princ, zeros(14))    # 14 zeros for missing Gyration tensor and shape parameters
        
        if not onlyshape:
            # generate the USR moments for the feature specific coordinates (Pharmacophore atoms)
            for subset in subsets:
    
                # only keep the atomic coordinates of the subset
                fcoords = array([coords.get(atomidx) for atomidx in subset])
                fmasses = array([masses.get(atomidx) for atomidx in subset])
    
                # initial zeroed out USRCAT feature moments
                if moment_number == 4:
                    fm = zeros(16*2)    # double the size because we calculate the same moments from the 2D projection of the coordinates
                else:
                    fm = zeros(12*2)
    
                # only attempt to generate moments if there are enough atoms available!
                if len(fcoords):
                    fm = usr_moments_with_existing(fcoords, *point_args, moment_number=moment_number)
    
                # append feature moments to the existing ones
                om = append(om, fm)
                # print("DEBUG: fm=", fm)
                # print("DEBUG: point 1 om=", om.shape)
                
                # append principal axes and their eigenvalues to the existing moments
                if len(fcoords) > 3:    # you need at least 4 atoms to calculate principal axes
                    center = fcoords.mean(axis=0)
                    principal_axes, eigenvalues = calc_principal_axes(fcoords, center)
                    # print("DEBUG: fcoords=", fcoords.tolist())
                    # print("DEBUG: principal_axes=", principal_axes)
                    for axis,eigenval in zip(principal_axes, eigenvalues):
                        princ = append(princ, axis)
                        # princ = append(princ, eigenval)
                        
                    # ## Append Gyration tensor and shape parameters
                    # princ = append(princ, get_shape_params_vector(fcoords, fmasses))
                else:   # if not enough atoms append zeros to the moments list
                    princ = append(princ, zeros(9))
                    # princ = append(princ, zeros(14))    # 14 zeros for missing Gyration tensor and shape parameters
        
        om = append(om, princ)
        # print("DEBUG: point 2 om=", om.shape)
        
        if plumed_colvars:
            os.chdir("plumed_files")
            # PLUMED atom numbering starts from 1 not from 0!
            atom_groups = [[a.GetIdx()+1 for a in molecule.GetAtoms()]] + [[s+1 for s in subset] for subset in subsets]
            shape_vector = PLUMED_functions.get_shape_vector(molname+".pdb", atom_groups, onlyshape=True)
            # print("DEBUG: shape_vector=", shape_vector.tolist())
            os.chdir("../")
            om = append(om, shape_vector)
        
        if electro_shape:   # calculate and append the 15 electroshape numbers for this conformer
            mol_copy = Chem.Mol(molecule)
            mol_copy.RemoveAllConformers()
            mol_copy.AddConformer(conformer, assignId=True)
            oddt_molecule = oddt.toolkit.Molecule(mol_copy)
            electroshape_vector = electroshape(oddt_molecule)
            om = append(om, electroshape_vector)
        
        # add conformer USRCAT moments to array for this molecule
        # print("DEBUG: moments=", moments.shape)
        # print("DEBUG: om=", om.shape)
        moments = vstack((moments, om))
        
    # do not include first row: all zeros!
    if ensemble_mode >= 1:
        feat_vec = [mean(moments[1:], axis=0)]    # return the average moment array in case multiple conformers were supplied
    elif ensemble_mode == 0:
        feat_vec = moments[1:]  # otherwise return the moments of the last frame
    
    return (molname, SMILES, feat_vec)


# if __name__ == '__main__':
#     from rdkit.Chem import AllChem
#
#     dist_matrix = Chem.MolFromSmiles('Cc1ccc(cc1Nc2nccc(n2)c3cccnc3)NC(=O)c4ccc(cc4)C[NH+]5CC[NH+](CC5)C')
#     AllChem.EmbedMultipleConfs(dist_matrix)
#
#     moments = generate_moments(dist_matrix)
#     print(moments)
