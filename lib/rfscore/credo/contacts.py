import os
import sys
from itertools import groupby, islice, product
from operator import itemgetter

import numpy
import oddt
from oddt.fingerprints import PLEC
from rdkit import Chem
from scipy.spatial import KDTree

from lib.global_fun import tree

oddt.toolkit = oddt.toolkits.rdk # force ODDT to use RDKit

# Import ConsScorTK libraries
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )  # import the top level package directory
from lib.rfscore.config import config, logger, res_atom_types
from lib.rfscore import ob
from lib.rfscore.ob import get_molecule

# Import ConsScorTK libraries
CREDO_DIR = os.path.dirname(os.path.realpath(__file__))
CONSSCORTK_LIB_DIR = CREDO_DIR[:-13] + "lib"
sys.path.append(CONSSCORTK_LIB_DIR)
from openbabel.openbabel import GetAtomicNum, GetSymbol, \
    GetCovalentRad, GetVdwRad  # MOVE THIS ON TOP ONCE YOU SOLVE THE IMPORTERROR
from openbabel import pybel

def get_contacts(protein, ligand, cutoff):
    """
    Returns a list of inter-atomic contacts in a protein-ligand complex, grouped
    by ligand atoms.
    """
    # CREATE KDTREES FOR BOTH THE PROTEIN AND THE LIGAND
    kdprot = KDTree(numpy.array([atom.coords for atom in protein.atoms]), leafsize=10)
    kdlig = KDTree(numpy.array([atom.coords for atom in ligand.atoms]), leafsize=10)

    # CREATE A SCIPY SPARSE DISTANCE MATRIX
    sdm = kdlig.sparse_distance_matrix(kdprot, cutoff, p=2.0)

    # CREATE A CONTACT LIST OF TUPLES IN THE FORM ((HETATM IDX, ATOM IDX), DISTANCE)
    # AND SORT BY HETATM IDX
    contacts = sorted(iter(sdm.items()), key=itemgetter(0))

    # GROUP CONTACTS BY LIGAND ATOMS
    for hidx, contactiter in groupby(contacts, key=lambda x:x[0][0]):
        hetatm = ligand.OBMol.GetAtom(int(hidx+1))

        atoms = []

        # IGNORE HYDROGENS
        if hetatm.IsHydrogen(): continue

        for ((hidx,aidx), distance) in contactiter:

            # GET THE PROTEIN ATOM
            atom = protein.OBMol.GetAtom(int(aidx+1))

            # IGNORE THIS ONE AS WELL IF HYDROGEN
            if atom.IsHydrogen(): continue

            atoms.append((atom, distance))

        yield hetatm, atoms

def get_distance_bins(cutoff, binsize):
    """
    """
    # start at zero to get properly distributed distance bins
    bins = numpy.arange(0, cutoff + binsize, binsize)

    # but remove the first bin that would only be hit if a distance is < 0
    bins = bins[1:]

    # DEBUG DISTANCE BINS
    logger.debug("The distance bins in Angstrom are {0}.".format(bins))

    return bins

def sum_descriptor_bins(descriptor, bins):
    """
    """
    # MAKE BIN ZERO THE SUM OF THE OTHER BINS
    for colidx in islice(range(descriptor.size), 0, descriptor.size, bins.size + 1):
        descriptor[colidx] += (descriptor[colidx+1:colidx+1+bins.size]).sum()

    return descriptor

def element_descriptor(protein, ligand, binsize=0.0):
    """
    Calculates a descriptor based on the combination of elements of the interacting
    atoms. This descriptor was used in the original RF-Score paper.

    Parameters
    ----------
    protein: str
        Path to the PDB structure of the protein.
    ligand: str
        Path to the structure of the ligand, must be readable by Open Babel.
    binsize: float
        Size of the distance bins in Angstrom that will be used to bin the contacts.
        The total range will be from 1.0 to <cutoff> + <binsize> in <binsize> steps.

    Returns
    -------
    descriptor: numpy.ndarray
        Array containing the sum of the founc interactions per element pair (and
        distance bin if specified).
    label: list
        List of labels that can be used as column headers.
    """
    # SUPPRESS OPENBABEL WARNINGS
    pybel.ob.obErrorLog.StopLogging()

    # CONVERT ELEMENT SYMBOLS TO ATOMIC NUMBERS
    atomicnums = (GetAtomicNum(str(element)) for element in config['elements'])

    # CREATE A NUMERICAL ID TO ELEMENT COMBINATION MAPPING
    # IMPORTANT TO MAP THE DESCRIPTOR VECTOR BACK TO THE LABELS
    element_pairs = product(sorted(atomicnums),repeat=2)
    element_pairs = dict((p,i) for i,p in enumerate(element_pairs))

    # ALSO CREATE A COLUMN LABEL FOR THIS DESCRIPTOR
    sorted_pairs = list(zip(*sorted(list(element_pairs.items()), key=itemgetter(1))))[0]

    numcols = len(element_pairs)

    # GENERATE THE DISTANCE BINS
    if binsize:

        # get the distance bins for the given cutoff and bin size
        bins = get_distance_bins(config['cutoff'], binsize)

        # NUMBER OF TOTAL COLUMNS IN DESCRIPTOR
        numcols *= (bins.size + 1)

        # CREATE A COLUMN FOR EACH ELEMENT PAIR AND DISTANCE BIN
        labels = []
        for x,y in sorted_pairs:
            for i in range(len(bins) + 1):
                label = "{0}.{1}-B{2}".format(GetSymbol(x), GetSymbol(y), i)
                labels.append(label)

    # LABEL WITHOUT BINS
    else:
        labels = ['.'.join((GetSymbol(x),GetSymbol(y))) for x,y in sorted_pairs]

    # DESCRIPTOR THAT WILL CONTAIN THE SUM OF ALL ELEMENT-ELEMENT INTERACTIONS
    descriptor = numpy.zeros(numcols, dtype=int)

    # GET THE CONTACTS
    contacts = get_contacts(protein, ligand, config['cutoff'])

    # ITERATE THROUGH CONTACT PAIRS AND DETERMINE SIFT
    for hetatm, hetatm_contacts in contacts:
        hetatm_num = hetatm.GetAtomicNum()

    # ITERATE THROUGH ALL THE CONTACTS THE HETATM HAS
        for atom, distance in hetatm_contacts:
            residue = atom.GetResidue()

            if residue.GetAtomID(atom).strip() in ['FE','FE2']:
                atom_num == 26
            else:
                atom_num = atom.GetAtomicNum()

            # IGNORE WATER RESIDUES
            if residue.GetName() == 'HOH': continue

            # IGNORE ZN,FE ETC.
            try: index = element_pairs[(atom_num, hetatm_num)]
            except KeyError: continue

            # BIN INTERACTIONS
            if binsize:

                # GET THE BIN THIS CONTACT BELONGS IN
                # DIGITIZE TAKES AN ARRAY-LIKE AS INPUT
                bin_id = numpy.digitize([distance,], bins)[0]
                descriptor[1 + index + index*bins.size + bin_id] += 1

            else:

                # ELEMENTS ARE SORTED NUMERICALLY
                descriptor[index] += 1

    if binsize: sum_descriptor_bins(descriptor, bins)

    return descriptor, labels

def sybyl_atom_type_descriptor(protein, ligand, binsize=0.0):
    """
    Calculates a descriptor based on the combination of elements of the interacting
    atoms. This descriptor was used in the original RF-Score paper.

    Parameters
    ----------
    protein: str
        Path to the PDB structure of the protein.
    ligand: str
        Path to the structure of the ligand, must be readable by Open Babel.
    binsize: float
        Size of the distance bins in Angstrom that will be used to bin the contacts.
        The total range will be from 1.0 to <cutoff> + <binsize> in <binsize> steps.

    Returns
    -------
    descriptor: numpy.ndarray
        Array containing the sum of the founc interactions per element pair (and
        distance bin if specified).
    label: list
        List of labels that can be used as column headers.
    """
    # SUPPRESS OPENBABEL WARNINGS
    pybel.ob.obErrorLog.StopLogging()

    # CREATE A NUMERICAL ID TO SYBYL ATOM TYPE PAIR COMBINATION MAPPING
    # IMPORTANT TO MAP THE DESCRIPTOR VECTOR BACK TO THE LABELS
    sybyl_pairs = product(sorted(config["sybyl atom types"]),repeat=2)
    sybyl_pairs = dict((p,i) for i,p in enumerate(sybyl_pairs))

    # ALSO CREATE A COLUMN LABEL FOR THIS DESCRIPTOR
    sorted_pairs = list(zip(*sorted(list(sybyl_pairs.items()), key=itemgetter(1))))[0]

    numcols = len(sybyl_pairs)

    # GENERATE THE DISTANCE BINS
    if binsize:

        # get the distance bins for the given cutoff and bin size
        bins = get_distance_bins(config['cutoff'], binsize)

        # NUMBER OF TOTAL COLUMNS IN DESCRIPTOR
        numcols *= (bins.size + 1)

        # CREATE A COLUMN FOR EACH ELEMENT PAIR AND DISTANCE BIN
        labels = []
        for x,y in sorted_pairs:
            for i in range(len(bins) + 1):
                label = "{0}.{1}-B{2}".format(x, y, i)
                labels.append(label)

    # LABEL WITHOUT BINS
    else:
        labels = ['.'.join((x, y)) for x,y in sorted_pairs]

    # DESCRIPTOR THAT WILL CONTAIN THE SUM OF ALL ELEMENT-ELEMENT INTERACTIONS
    descriptor = numpy.zeros(numcols, dtype=int)

    # GET THE CONTACTS
    contacts = get_contacts(protein, ligand, config['cutoff'])

    # ITERATE THROUGH CONTACT PAIRS AND DETERMINE SIFT
    for hetatm, hetatm_contacts in contacts:
        hetatm_type = hetatm.GetType()

        # ITERATE THROUGH ALL THE CONTACTS THE HETATM HAS
        for atom, distance in hetatm_contacts:
            residue = atom.GetResidue()

            if residue.GetAtomID(atom).strip() in ['FE','FE2']:
                atom_num == 26
                atom_type = 'Fe'
            else:
                atom_num = atom.GetAtomicNum()
                atom_type = atom.GetType()

            # IGNORE WATER RESIDUES
            if residue.GetName() == 'HOH': continue

            # IGNORE ZN,FE ETC.
            try: index = sybyl_pairs[(atom_type, hetatm_type)]
            except KeyError: continue

            # BIN INTERACTIONS
            if binsize:

                # GET THE BIN THIS CONTACT BELONGS IN
                # DIGITIZE TAKES AN ARRAY-LIKE AS INPUT
                bin_id = numpy.digitize([distance,], bins)[0]
                descriptor[1 + index + index*bins.size + bin_id] += 1

            else:

                # ELEMENTS ARE SORTED NUMERICALLY
                descriptor[index] += 1

    if binsize: sum_descriptor_bins(descriptor, bins)

    return descriptor, labels


def remove_selected_interactions(descriptor, labels, residue_sift_dict, residue_atomname_sift_dict, interactions2remove):
    
    indices2remove = []
    for interaction in interactions2remove:
        indices2remove.append(labels.index(interaction))
    
    new_residue_sift_dict = {}
    new_residue_atomname_sift_dict = tree()
    new_labels = [i for j, i in enumerate(labels) if j not in indices2remove]
    
    for RESIDUE in list(residue_sift_dict.keys()):
        new_residue_sift_dict[RESIDUE] = numpy.delete(residue_sift_dict[RESIDUE], indices2remove)
        for atom_name in list(residue_atomname_sift_dict[RESIDUE].keys()):
            new_residue_atomname_sift_dict[RESIDUE][atom_name] = numpy.delete(residue_atomname_sift_dict[RESIDUE][atom_name], indices2remove)
        
        
    return descriptor, new_labels, new_residue_sift_dict, new_residue_atomname_sift_dict


def get_binary_sift(descriptor, residue_sift_dict, residue_atomname_sift_dict):
    
    binary_descriptor = numpy.array(descriptor>0, dtype=int)
    binary_residue_sift_dict = {}
    binary_residue_atomname_sift_dict = tree()
    
    for RESIDUE in list(residue_sift_dict.keys()):
        binary_residue_sift_dict[RESIDUE] = numpy.array(residue_sift_dict[RESIDUE]>0, dtype=int)
        for atom_name in list(residue_atomname_sift_dict[RESIDUE].keys()):
            binary_residue_atomname_sift_dict[RESIDUE][atom_name] = numpy.array(residue_atomname_sift_dict[RESIDUE][atom_name]>0, dtype=int)
        
    return binary_descriptor, binary_residue_sift_dict, binary_residue_atomname_sift_dict


def sift_descriptor(proteins, ligands, binsize=0.0, skip_empty=True, binary=True, ensemble_mode=0, molname=None):
    """
    Modified Function by Thomas Evangelidis.
    Calculates a descriptor of the protein-ligand complex as the sum of the structural
    interaction fingerprints (SIFTs) of all interacting atoms.

    Parameters
    ----------
    protein: OpenBabel molecule object
        the protein
    ligand_path: OpenBabel molecule object
        the ligand
    binsize: float
        Size of the distance bins in Angstrom that will be used to bin the contacts.
        The total range will be from 1.0 to <cutoff> + <binsize> in <binsize> steps.
    skip_empty: boolean
        remove from 'residue_sift_dict' residues that have no interaction of any type and from 'residue_atomname_sift_dict' atoms
        that have no interaction of any type. (Thomas' addition)
        
    Returns
    -------
    descriptor: numpy.ndarray
        The shape of the descriptor array will be 1D equal to the number of contact types, or 2D (number of bins x number of contact types)
        if a binsize was given.
    
    labels: a 72-element list with the name of the interaction type at each bin of the SiFt array of each residue. For more details see the Supporting Info
            of "Ballester2010 - A machine learning approach to predicting protein-ligand binding affinity with applications to molecular docking (RFscore)".
            labels= ['covalent-B0', 'covalent-B1', 'covalent-B2', 'covalent-B3', 'covalent-B4', 'covalent-B5',
            'vdw_clash-B0', 'vdw_clash-B1', 'vdw_clash-B2', 'vdw_clash-B3', 'vdw_clash-B4', 'vdw_clash-B5',
            'vdw-B0', 'vdw-B1', 'vdw-B2', 'vdw-B3', 'vdw-B4', 'vdw-B5', 'proximal-B0', 'proximal-B1', 'proximal-B2',
            'proximal-B3', 'proximal-B4', 'proximal-B5', 'hbond-B0', 'hbond-B1', 'hbond-B2', 'hbond-B3', 'hbond-B4',
            'hbond-B5', 'weak_hbond-B0', 'weak_hbond-B1', 'weak_hbond-B2', 'weak_hbond-B3', 'weak_hbond-B4', 'weak_hbond-B5',
            'xbond-B0', 'xbond-B1', 'xbond-B2', 'xbond-B3', 'xbond-B4', 'xbond-B5', 'ionic-B0', 'ionic-B1', 'ionic-B2',
            'ionic-B3', 'ionic-B4', 'ionic-B5', 'metal_complex-B0', 'metal_complex-B1', 'metal_complex-B2',
            'metal_complex-B3', 'metal_complex-B4', 'metal_complex-B5', 'aromatic-B0', 'aromatic-B1', 'aromatic-B2',
            'aromatic-B3', 'aromatic-B4', 'aromatic-B5', 'hydrophobic-B0', 'hydrophobic-B1', 'hydrophobic-B2',
            'hydrophobic-B3', 'hydrophobic-B4', 'hydrophobic-B5', 'carbonyl-B0', 'carbonyl-B1', 'carbonyl-B2',
            'carbonyl-B3', 'carbonyl-B4', 'carbonyl-B5']
    residue_sift_dict:  dict with the per residue SIFT: residue->72-element SiFt array (Thomas' addition)
    residue_atomname_sift_dict: multidict with the per atom SIFT: residue->atom name->72-element SiFt array (Thomas' addition)
    """
    
    if molname:
        print("Calculating SiFt of molecule", molname)
    
    if type(proteins) == str:
        proteins = get_molecule(proteins, get_all_mols=True)
    if type(ligands) == str:
        ligands = get_molecule(ligands, get_all_mols=True)
    
    # SUPPRESS OPENBABEL WARNINGS
    pybel.ob.obErrorLog.StopLogging()

    # ELEMENT TABLE TO DETERMINE VDW AND COVALENT BONDS
    from lib.rfscore.credo import interactions  # MOVE THIS ON TOP ONCE YOU SOLVE THE OBElementTable ImportError

    # CREDO DESCRIPTOR LABELS
    interaction_types = ['covalent','vdw_clash','vdw','proximal','hbond','weak_hbond',
                         'xbond','ionic','metal_complex','aromatic','hydrophobic',
                         'carbonyl']

    numcols = len(interaction_types)

    # GENERATE THE DISTANCE BINS
    if binsize:

        # get the distance bins for the given cutoff and bin size
        bins = get_distance_bins(config['cutoff'], binsize)

        offset = bins.size + 1

        # DEBUG DISTANCE BINS
        logger.debug("The distance bins in Angstrom are {0}.".format(bins))

        # NUMBER OF TOTAL COLUMNS IN DESCRIPTOR
        numcols *= (bins.size + 1)

        labels = []
        # CREATE A COLUMN FOR EACH ELEMENT PAIR AND DISTANCE BIN
        for interaction_type in interaction_types:
            for i in range(len(bins) + 1):
                label = "{0}-B{1}".format(interaction_type, i)
                labels.append(label)

    # LABEL WITHOUT BINS
    else: labels = interaction_types
    
    
    ensemble_descriptor_list = []
    ensemble_residue_siftList_dict = {}
    ensemble_residue_atomname_siftList_mdict = tree()
    
    frameNum = 0
    if ensemble_mode == 0:  # keep only the last frames
        proteins = [ proteins[-1] ]
        ligands = [ ligands[-1] ]
    for protein, ligand in zip(proteins, ligands):  # iterate over all input molecule pairs (usually conformers)
        frameNum += 1
        # print("Calculating SiFt of frame", frameNum)
        # DESCRIPTOR THAT WILL CONTAIN THE SUM OF ALL ELEMENT-ELEMENT INTERACTIONS
        descriptor = numpy.zeros(numcols, dtype=int)
    
        # GET THE ATOM TYPES FOR THE LIGAND
        # CALCULATED ON THE FLY
        lig_atom_types = ob.get_atom_types(ligand, config)
    
        contacts = get_contacts(protein, ligand, config['cutoff'])
        
        # ITERATE THROUGH CONTACT PAIRS AND DETERMINE SIFT
        residue_sift_dict = {}
        residue_atomname_sift_dict = tree()
        for hetatm, hetatm_contacts in contacts:
            
            # GET THE ATOM TYPES FOR THE HETATM
            hetatm_types = lig_atom_types[hetatm.GetIdx()]
    
            # GET ATOM RADII FOR THE LIGAND ATOM
            hetatm_cov = GetCovalentRad(hetatm.GetAtomicNum())
            hetatm_vdw = GetVdwRad(hetatm.GetAtomicNum())
    
            # ITERATE THROUGH ALL THE CONTACTS THE HETATM HAS
            for atom, distance in hetatm_contacts:
    
                # INITIALIZE STRUCTURAL INTERACTION FINGERPRINT
                sift = numpy.zeros(descriptor.size, dtype=int)
    
                residue = atom.GetResidue()
                res_name = residue.GetName()[:3]
    
                # IGNORE WATER RESIDUES
                if res_name in ['HOH', 'WAT']: continue
                
                # SAVE THE FINGERPRINT OF THIS RESIDUE INTO A DICTIONARY
                RESIDUE = residue.GetName()+str(residue.GetNum())
                atom_name = residue.GetAtomID(atom).replace(" ", "")
                if not RESIDUE in list(residue_sift_dict.keys()):
                    residue_sift_dict[RESIDUE] = numpy.zeros(descriptor.size, dtype=int)
    
                # GET ATOM TYPES FOR THE PROTEIN ATOM
                try:
                    atom_types = res_atom_types[res_name][residue.GetAtomID(atom).strip()]
                except KeyError:
                    logger.warn("Cannot find atom types for {} {}."
                                .format(res_name, residue.GetAtomID(atom).strip()))
                    continue
    
                sum_cov = hetatm_cov + GetCovalentRad(atom.GetAtomicNum())
                sum_vdw = hetatm_vdw + GetVdwRad(atom.GetAtomicNum())
    
                # BIN INTERACTIONS
                if binsize:
    
                    # GET THE BIN THIS CONTACT BELONGS IN
                    # DIGITIZE TAKES AN ARRAY-LIKE AS INPUT
                    bin_id = numpy.digitize([distance,], bins)[0] + 1
    
                else:
                    offset = 1
                    bin_id = 0
    
                # COVALENT BOND - SHOULD NOT OCCUR IN PDBBIND
                if distance <= sum_cov: sift[0 * offset + bin_id] = 1
    
                # VAN DER WAALS CLASH
                elif distance <= sum_vdw: sift[1 * offset + bin_id] = 1
    
                # VAN DER WAALS CONTACT
                elif distance <= sum_vdw + 0.5: sift[2 * offset + bin_id] = 1
    
                # PROXIMAL
                else: sift[3 * offset + bin_id] = 1
    
                if interactions.is_hbond(hetatm,hetatm_types,atom,atom_types,distance): sift[4 * offset + bin_id] = 1
                if interactions.is_weak_hbond(hetatm,hetatm_types,atom,atom_types,distance): sift[5 * offset + bin_id] = 1
                if interactions.is_xbond(hetatm,hetatm_types,atom,atom_types,distance): sift[6 * offset + bin_id] = 1
                if interactions.is_ionic(hetatm,hetatm_types,atom,atom_types,distance): sift[7 * offset + bin_id] = 1
                if interactions.is_metal_complex(hetatm,hetatm_types,atom,atom_types,distance): sift[8 * offset + bin_id] = 1
                if interactions.is_aromatic(hetatm,hetatm_types,atom,atom_types,distance): sift[9 * offset + bin_id] = 1
                if interactions.is_hydrophobic(hetatm,hetatm_types,atom,atom_types,distance): sift[10 * offset + bin_id] = 1
                if interactions.is_carbonyl(hetatm,hetatm_types,atom,atom_types,distance): sift[11 * offset + bin_id] = 1
                descriptor += sift
                # print("RESIDUE=", RESIDUE[:3], "atom name", residue.GetAtomID(atom), "atom_types=", atom_types)
                # print("DEBUG: adding sift=", sift)
                residue_sift_dict[RESIDUE] += sift
                residue_atomname_sift_dict[RESIDUE][atom_name] = sift
    
        if binsize: sum_descriptor_bins(descriptor, bins)
        if skip_empty == True:
            for RESIDUE in list(residue_sift_dict.keys()):
                if residue_sift_dict[RESIDUE].sum() == 0:
                    del residue_sift_dict[RESIDUE]
                
                for atom_name in residue_atomname_sift_dict[RESIDUE]:
                    if residue_atomname_sift_dict[RESIDUE][atom_name].sum() == 0:
                        del residue_atomname_sift_dict[RESIDUE][atom_name]
        
        
        # print("residue_sift_dict=")
        # for k,v in residue_sift_dict.items():
        #     print(k, v.tolist())
        # print(numpy.sum(residue_sift_dict.values(), axis=0).tolist())
        # print("descriptor=")
        # print(descriptor.tolist())
        # print("DEBUG: residue_atomname_sift_dict=", residue_atomname_sift_dict)
        
        new_descriptor, new_labels, new_residue_sift_dict, new_residue_atomname_sift_dict = descriptor, labels, residue_sift_dict, residue_atomname_sift_dict
        
        # OPTIONAL: remove some interaction types
        # print("labels=", labels)
        # new_descriptor, new_labels, new_residue_sift_dict, new_residue_atomname_sift_dict = remove_selected_interactions(descriptor,
        #         labels, residue_sift_dict, residue_atomname_sift_dict,
        #         interactions2remove=['proximal-B0', 'proximal-B1', 'proximal-B2', 'proximal-B3', 'proximal-B4', 'proximal-B5'])
        # new_descriptor, new_labels, new_residue_sift_dict, new_residue_atomname_sift_dict = descriptor, labels, residue_sift_dict, residue_atomname_sift_dict
        
        # OPTIONAL: get binary sifts
        if binary:
            new_labels = labels
            new_descriptor, new_residue_sift_dict, new_residue_atomname_sift_dict = get_binary_sift(descriptor, residue_sift_dict, residue_atomname_sift_dict)
    
        # Update the ensemble_* list and dictionaries
        ensemble_descriptor_list.append(new_descriptor)
        for r in list(new_residue_sift_dict.keys()):
            if not r in list(ensemble_residue_siftList_dict.keys()):
                ensemble_residue_siftList_dict[r] = [ new_residue_sift_dict[r] ]
            else:
                ensemble_residue_siftList_dict[r].append( new_residue_sift_dict[r] )
    
    ensemble_descriptor = numpy.mean(ensemble_descriptor_list, axis=0)
    ensemble_labels = new_labels
    # Append arrays of zeros at those residues which had no contact with the ligand in some of the frames
    for r in list(ensemble_residue_siftList_dict.keys()):
        while len(ensemble_residue_siftList_dict[r]) < frameNum:
            ensemble_residue_siftList_dict[r].append( numpy.zeros(numcols, dtype=int) )
    
    if ensemble_mode == 0:  # keep the sift of the last frame onl
        for r in list(ensemble_residue_siftList_dict.keys()):
            ensemble_residue_siftList_dict[r] = ensemble_residue_siftList_dict[r][-1]
    elif ensemble_mode == 1:  # calculate the mean of the sift
        for r in list(ensemble_residue_siftList_dict.keys()):
            ensemble_residue_siftList_dict[r] = numpy.mean(ensemble_residue_siftList_dict[r], axis=0)
    elif ensemble_mode == 2:  # average and round the bits to 0 and 1
        for r in list(ensemble_residue_siftList_dict.keys()):
            ensemble_residue_siftList_dict[r] = numpy.round( numpy.mean(ensemble_residue_siftList_dict[r], axis=0) )
    elif ensemble_mode == 3:  # calculate the mean and stdev of the sift
        for r in list(ensemble_residue_siftList_dict.keys()):
            ensemble_residue_siftList_dict[r] = numpy.array([e for duplet in zip(numpy.mean(ensemble_residue_siftList_dict[r], axis=0), numpy.std(ensemble_residue_siftList_dict[r], axis=0)) for e in duplet])
    # TODO: the ensemble_residue_atomname_sift_mdict !! (not necessary at this point                     
    ensemble_residue_sift_dict = ensemble_residue_siftList_dict # because now the values are arrays, not Lists of arrays
        
    return ensemble_descriptor, ensemble_labels, ensemble_residue_sift_dict, new_residue_atomname_sift_dict


def calc_PLEC_multiconf(ligands_file, proteins_file, depth_ligand=2, depth_protein=4, distance_cutoff=4.5,
                        size=16384, count_bits=True, sparse=False, ignore_hoh=True, ensemble_mode=0):
    """
        protein:    OpenBabel mol
        ligands:    OpenBabel mol
        RETURNS:
        ensemble_PLEC:  the PLEC fingerprint(of the whole trajectory, determined by the ensemble_mode.
    """
    
    protMol = list(oddt.toolkit.readfile('pdb', proteins_file))[0]  ; # pdb format does not support list of molecules; isntead get the conformers through protein.Mol.GetConformers()
    protConfs = protMol.Mol.GetConformers()
    ligands = list(oddt.toolkit.readfile('mol2', ligands_file))
    
    frameNum = 0
    PLEC_list = []
    if ensemble_mode == 0:  # keep only the last frame
        print("Calculating PLEC of last frame.")
        ligand = ligands[-1]    # take last ligand conformer
        protConf = protConfs[-1]    # take last pocket conformer
        protmol_copy = Chem.Mol(protMol.Mol)
        protmol_copy.RemoveAllConformers()
        protmol_copy.AddConformer(protConf, assignId=True)
        oddt_protein = oddt.toolkit.Molecule(protmol_copy)
        oddt_ligand = oddt.toolkit.Molecule(ligand)
        plec_vec = PLEC(oddt_ligand, oddt_protein, depth_ligand=depth_ligand, depth_protein=depth_protein, distance_cutoff=distance_cutoff,
               size=size, count_bits=count_bits, sparse=sparse, ignore_hoh=ignore_hoh)
        PLEC_list.append(plec_vec)
    for protConf, ligand in zip(protConfs, ligands):  # iterate over all input molecule pairs (usually conformers)
        frameNum += 1
        print("Calculating PLEC of frame", frameNum)
        protmol_copy = Chem.Mol(protMol.Mol)
        protmol_copy.RemoveAllConformers()
        protmol_copy.AddConformer(protConf, assignId=True)
        oddt_protein = oddt.toolkit.Molecule(protmol_copy)
        oddt_ligand = oddt.toolkit.Molecule(ligand)
        
        plec_vec = PLEC(oddt_ligand, oddt_protein, depth_ligand=depth_ligand, depth_protein=depth_protein, distance_cutoff=distance_cutoff,
               size=size, count_bits=count_bits, sparse=sparse, ignore_hoh=ignore_hoh)
        PLEC_list.append(plec_vec)
    
    if ensemble_mode == 0:  # keep the sift of the last frame only
        ensemble_PLEC = PLEC_list[0]
    elif ensemble_mode == 1:  # calculate the ensemble average PLEC
        ensemble_PLEC = numpy.mean(PLEC_list, axis=0)
    elif ensemble_mode == 2:  # average and round the bits to 0 and 1 (NOT FUNCTIONAL YET!)
        ensemble_PLEC = numpy.round( numpy.mean(PLEC_list, axis=0) )
    elif ensemble_mode == 3:  # calculate the mean and stdev of the sift
        average_PLEC = numpy.mean(PLEC_list, axis=0)
        stdev_PLEC =  numpy.std(PLEC_list, axis=0)
        ensemble_PLEC = numpy.append(average_PLEC, stdev_PLEC)
    
    return ensemble_PLEC

