import os
import sys

# Import ConsScorTK libraries
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )  # import the top level package directory
from openbabel import openbabel as ob
#import pybel
from lib.modlib import pybel
from lib.rfscore.config import logger


def get_molecule(path, get_all_mols=False):
    """
    Reads a molecular structure with Open Babel.
    ARGS:
    path:   filepath
            can be any structure file type supported by OpenBabel.
    get_all_conf:   boolean
            If False and the file has multiple conformers or multiple different molecules, you will get only the last one. If true, then you will get
            a list of all molecule objects.
    RETURNS:
    molecules:  list of PyBel.OBMol molecule objects
                one for each molecule/conformation in the file.
    """
    if os.path.exists(path):

        # DETERMINE FILE FORMAT
        filename, filetype = os.path.splitext(path)
        
        try:
            molecules = list(pybel.readfile(filetype[1:], path))
        except StopIteration:
            logger.error("cannot read molecule {}: StopIteration.".format(path))
            sys.exit(1)
        
        if get_all_mols:
           return molecules
        else:
            return molecules[-1]

    else:
        logger.error('cannot load molecule: the file {} does not exist.'.format(path))
        sys.exit(1)

def get_atom_types(molecule, config):
    """
    Creates a dictionary containing all the identified atom types for all atoms
    in this molecule. Only feasible really for small molecules and not for protein
    structures.
    """
    # PREPARE DICTIONARY DATA STRUCTURE FOR MOLECULE ATOM TYPES
    atom_types = dict((atom.idx,{}) for atom in molecule.atoms)

    # ITERATE THROUGH ALL SMARTS PATTERNS OF ALL CREDO ATOM TYPES
    for atom_type in config['atom types']:
        for smarts in list(config['atom types'][atom_type].values()):

            # MATCH THAT IS GOING TO BE SET FOR MATCHING ATOMS
            pattern = ob.OBSmartsPattern()
            pattern.Init(str(smarts))

            pattern.Match(molecule.OBMol)

            for match in pattern.GetUMapList():
                for idx in match:
                    atom_types[idx][atom_type] = 1

    return atom_types

def extract_ligand(obmol, res_name='INH'):
    """
    This function is used to extract the ligands which are always named 'INH'
    from CSAR structures.
    """
    ligand = ob.OBMol()
    
    for residue in ob.OBResidueIter(obmol):
        if residue.GetName() == res_name:
        
            mapping = {}
        
            # insert the ligand atoms into the new molecule
            for i,atom in enumerate(ob.OBResidueAtomIter(residue),1):                
                ligand.InsertAtom(atom)
                mapping[atom.GetIdx()] = i
            
            # re-create the bonds
            for atom in ob.OBResidueAtomIter(residue):
                for bond in ob.OBAtomBondIter(atom):
                    bgn_idx = mapping[bond.GetBeginAtomIdx()]
                    end_idx = mapping[bond.GetEndAtomIdx()]
                    
                    ligand.AddBond(bgn_idx, end_idx, bond.GetBondOrder(), bond.GetFlags())
            
            # remove the ligand from the original structure
            for atom_idx in sorted(list(mapping.keys()), reverse=True):
                atom = obmol.GetAtom(atom_idx)
                obmol.DeleteAtom(atom)
                
            obmol.DeleteResidue(residue)
            
            break
        
    return pybel.Molecule(ligand)   
