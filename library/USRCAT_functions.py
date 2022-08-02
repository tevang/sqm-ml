import copy
import os
import re
import sys
from itertools import combinations
from operator import itemgetter

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.spatial import distance
from scoop import shared
from sklearn.preprocessing import minmax_scale

from library.molfile.ligfile_parser import write_mols2files, load_structure_file
from .ConsScoreTK_Statistics import TopDown_Concordance2, Kendalls_tau, Kendalls_W, Create_Curve
from .utils.print_functions import ColorPrint
from .featvec.similarity_matrix import calc_Fingeprint_sim_list
from .global_fun import which, run_commandline, concatenate_files, tree

sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )  # import the top level package directory
CONSSCORTK_LIB_DIR = os.path.dirname(os.path.realpath(__file__))
CONSSCORTK_BIN_DIR = CONSSCORTK_LIB_DIR[:-3] + "general_tools"
# from pybel import Outputfile, readfile
# for 2D pharmacophore fingerprints
from rdkit.Chem import ChemicalFeatures, Conformer

fdefName = CONSSCORTK_LIB_DIR + '/BaseFeatures_DIP2_NoMicrospecies.fdef'
featFactory = ChemicalFeatures.BuildFeatureFactory(fdefName)
from rdkit.Chem.Pharm2D.SigFactory import SigFactory
sigFactory = SigFactory(featFactory,minPointCount=2,maxPointCount=3)
sigFactory.SetBins([(0,2),(2,5),(5,8)])
sigFactory.Init()
sigFactory.GetSigSize()
from rdkit.Chem.Pharm2D import Gobbi_Pharm2D, Generate

from scipy.stats import zscore, rankdata, pearsonr
import pandas as pd
from .usrcat.toolkits.rd import generate_moments
import numpy as np

def find_scaffolds(molname_SMILES_conformersMol_mdict, actives_list):
    
    scaffold_set = set()    # set of active ligand scaffolds
    molname2scaffold_dict = {}
    for molname in list(molname_SMILES_conformersMol_mdict.keys()):
        if not molname in actives_list:
            continue
        for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
            mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_SMILES = Chem.MolToSmiles(core, isomericSmiles=True, canonical=True, allBondsExplicit=True)
            scaffold_set.add(scaffold_SMILES)
            molname2scaffold_dict[molname] = scaffold_SMILES
    
    return molname2scaffold_dict


def scale_moments(moments_list, group=False):
    """
        FUNCTION to rescale a list of moments by using the overal min and max values (useful from ML training). 
    """
    
    if group == False:
        moments_list = np.array(moments_list, dtype=float)
        Xmax = np.max(moments_list)
        Xmin = np.min(moments_list)
        scaled_moments_list = (moments_list - Xmin) / float(Xmax - Xmin)
    elif group == True:
        moments_list = np.array(moments_list, dtype=float)
        Xmax1 = np.max(moments_list[:, 0::12])
        Xmin1 = np.min(moments_list[:, 0::12])
        Xmax2 = np.max(moments_list[:, 1::12])
        Xmin2 = np.min(moments_list[:, 1::12])
        Xmax3 = np.max(moments_list[:, 2::12])
        Xmin3 = np.min(moments_list[:, 2::12])
        Xmax4 = np.max(moments_list[:, 3::12])
        Xmin4 = np.min(moments_list[:, 3::12])
        Xmax5 = np.max(moments_list[:, 4::12])
        Xmin5 = np.min(moments_list[:, 4::12])
        Xmax6 = np.max(moments_list[:, 5::12])
        Xmin6 = np.min(moments_list[:, 5::12])
        Xmax7 = np.max(moments_list[:, 6::12])
        Xmin7 = np.min(moments_list[:, 6::12])
        Xmax8 = np.max(moments_list[:, 7::12])
        Xmin8 = np.min(moments_list[:, 7::12])
        Xmax9 = np.max(moments_list[:, 8::12])
        Xmin9 = np.min(moments_list[:, 8::12])
        Xmax10 = np.max(moments_list[:, 9::12])
        Xmin10 = np.min(moments_list[:, 9::12])
        Xmax11 = np.max(moments_list[:, 10::12])
        Xmin11 = np.min(moments_list[:, 10::12])
        Xmax12 = np.max(moments_list[:, 11::12])
        Xmin12 = np.min(moments_list[:, 11::12])
        scaled_moments_list = copy.deepcopy(moments_list)
        scaled_moments_list[:, 0::12] = (moments_list[:, 0::12] - Xmin1) / float(Xmax1 - Xmin1)
        scaled_moments_list[:, 1::12] = (moments_list[:, 1::12] - Xmin2) / float(Xmax2 - Xmin2)
        scaled_moments_list[:, 2::12] = (moments_list[:, 2::12] - Xmin3) / float(Xmax3 - Xmin3)
        scaled_moments_list[:, 3::12] = (moments_list[:, 3::12] - Xmin4) / float(Xmax4 - Xmin4)
        scaled_moments_list[:, 4::12] = (moments_list[:, 4::12] - Xmin5) / float(Xmax5 - Xmin5)
        scaled_moments_list[:, 5::12] = (moments_list[:, 5::12] - Xmin6) / float(Xmax6 - Xmin6)
        scaled_moments_list[:, 6::12] = (moments_list[:, 6::12] - Xmin7) / float(Xmax7 - Xmin7)
        scaled_moments_list[:, 7::12] = (moments_list[:, 7::12] - Xmin8) / float(Xmax8 - Xmin8)
        scaled_moments_list[:, 8::12] = (moments_list[:, 8::12] - Xmin9) / float(Xmax9 - Xmin9)
        scaled_moments_list[:, 9::12] = (moments_list[:, 9::12] - Xmin10) / float(Xmax10 - Xmin10)
        scaled_moments_list[:, 10::12] = (moments_list[:, 10::12] - Xmin11) / float(Xmax11 - Xmin11)
        scaled_moments_list[:, 11::12] = (moments_list[:, 11::12] - Xmin12) / float(Xmax12 - Xmin12)    
    
    return scaled_moments_list
    

def exec_obspectrophores(sdf):
    """
        FUNCTION to create Spectrophores from an sdf file.
    """
    obspectrophore_exe = which("obspectrophore")
    
    ColorPrint("\nCalculating Spectrophores from file " + sdf, "BOLDBLUE")
    run_commandline(obspectrophore_exe+" -i " + sdf + " -s All -a 10",
                                            logname=sdf.replace(".sdf", "") + "_obspectrophore.log")

def get_spectrophores(LIGAND_STRUCTURE_FILE):
    """
        FUNCTION to calculate the spectrophore vector for each molecule with OpenBabel.
        
        ARGS:
        LIGAND_STRUCTURE_FILE:   a .mol2 or .sdf multi-mol, multi-conf ligand file.
    """
    from scoop import futures

    if os.path.exists("obspectrophore.log"):
        ColorPrint("\nLoading Spectrophores from file obspectrophore.log.", "BOLDBLUE")
    else:
        # Split the input file to individual sdf files, each one containing conformations of one molecule
        molnames_list = write_mols2files(LIGAND_STRUCTURE_FILE, get_molnames=True)
        sdf_args = [m+".sdf" for m in molnames_list]
        
        # Parallel execution
        print("Calculating Spectrophores in parallel...")
        results = list(futures.map(exec_obspectrophores, sdf_args))   # list of if the form: [(molname, SMILES, moments), ...]
        
        # concatenate individual obspectrophore.log files
        filename_list = [m+"_obspectrophore.log" for m in molnames_list]
        concatenate_files(filename_list, 'obspectrophore.log', clean=True)
    
    # process the output file
    molname_spectrophoreList_dict = {}
    molname_order = []
    spectrophore_list = []
    with open('obspectrophore.log', 'r') as f:
        for line in f:
            words = line.split()
            if len(words) < 20: # skip header lines
                continue
            molname = re.sub("_iso[0-9]+", "", words[0]).lower()  # remove the _iso suffix (THIS MAY BE DANGEROUS IF YOU HAVE MAY ISOMERS OF THE SAME MOLECULE IN THIS FILE)
            molname_order.append(molname)
            spectrophore = [float(s) for s in words[1:]]
            spectrophore_list.append(spectrophore)
    
    spectrophore_list = minmax_scale(spectrophore_list).tolist()    # standarize the spectrophore values (otherwise the RMSEc will be huge!)
    for molname, spectrophore in zip(molname_order, spectrophore_list):
        try:
            molname_spectrophoreList_dict[molname].append(spectrophore)
        except KeyError:
            molname_spectrophoreList_dict[molname] = [ spectrophore ]
    
    # Calculate mean values (TODO: distribution moments)
    molname_spectrophoreVec_dict = {}
    for molname in list(molname_spectrophoreList_dict.keys()):
         moment1 = np.mean(molname_spectrophoreList_dict[molname], axis=0)
         # moment2 = np.std(molname_spectrophoreList_dict[molname], axis=0)
         molname_spectrophoreVec_dict[molname] = moment1
    
    return molname_spectrophoreVec_dict


def expand_momoments(*args):
    """
        FUNCTION to expand the moments vector by appending an arbitrary number of extra vectors. The 1st arguments must always be the
        moments dictionary.
    """
    # print("DEBUG: args[1]=", args[1]))
    molname_SMILES_moments_mdict = args[0]
    for molname in list(molname_SMILES_moments_mdict.keys()):
        for SMILES in list(molname_SMILES_moments_mdict[molname].keys()):
            for d in args[1:]:  # for every external vector dictionary
                # print("DEBUG: molname_SMILES_moments_mdict[molname][SMILES]=", molname_SMILES_moments_mdict[molname][SMILES]))
                # print("DEBUG: d[molname]=", d[molname].tolist()))
                moments = molname_SMILES_moments_mdict[molname][SMILES][0]
                # print("DEBUG: moments=", moments.tolist())
                # molname_SMILES_moments_mdict[molname][SMILES] = [ np.vstack( (moments, d[molname]) ) ]
                molname_SMILES_moments_mdict[molname][SMILES] = [ np.append(moments, d[molname]) ]
    
    return molname_SMILES_moments_mdict


def calculate_moments(molname_SMILES_conformersMol_mdict,
                      ensemble_mode=1,
                      moment_number=4,
                      onlyshape=False,
                      plumed_colvars=False,
                      use_spectrophores=True,
                      conf_num=1,
                      LIGAND_STRUCTURE_FILE=None):
    from .usrcat.toolkits.rd import generate_moments    # MOVE ON TOP ONCE YOU SOLVE THE ImportError of OpenBabel
    from scoop import futures

    # Now Generate Moments and save them in a common array for all the conformers of the same compound
    ColorPrint("Generating Moments of all compounds... ", "BOLDBLUE")
    if conf_num > 1:    # initialize the parallel conformer generator
        try:
            shared.setConst(MOLNAME_SMILES_CONFORMERSMOL_MULTIDICT=molname_SMILES_conformersMol_mdict)
        except TypeError:
            pass  # shoud always be initialized
        conf = Conformer()  # this is the ConsScorTK Conformer class
    molname_SMILES_moments_mdict = tree()  # molname->SMILES->array of moments (Nx60, where N is the number of conformers)
    molecule_list = []
    hydrogens_list = []
    moment_number_list = []
    molname_list = []
    SMILES_list = []
    onlyshape_list = []
    ensemble_mode_list = []
    plumed_colvars_list = []
    for molname in list(molname_SMILES_conformersMol_mdict.keys()):
        for SMILES in list(molname_SMILES_conformersMol_mdict[molname].keys()):
            # Serial execution
            # if conf_num > 1:
            #     conf.gen_singlemol_conf_parallel(molname, SMILES, N=conf_num)
            # molname_SMILES_moments_mdict[molname][SMILES] = generate_moments(molname_SMILES_conformersMol_mdict[molname][SMILES],
            #                                                                      moment_number=moment_number, onlyshape=onlyshape, molname=molname,
            #                                                                      ensemble_mode=ensemble_mode, plumed_colvars=False)    # an Nx60 array where N is the number of conformers
            molecule_list.append(molname_SMILES_conformersMol_mdict[molname][SMILES])
            hydrogens_list.append(False)
            moment_number_list.append(moment_number)
            molname_list.append(molname)
            SMILES_list.append(SMILES)
            onlyshape_list.append(onlyshape)
            ensemble_mode_list.append(ensemble_mode)
            plumed_colvars_list.append(False)

    # Parallel execution
    if conf_num > 1:    # molname_SMILES_conformersMol_mdict will be automatically updated with new conformers
        unecessary = list(futures.map(conf.gen_singlemol_conf_parallel, molname_list, SMILES_list, [conf_num]*len(SMILES_list)))
    results = list(futures.map(generate_moments, molecule_list, hydrogens_list, moment_number_list, onlyshape_list, molname_list,
                    SMILES_list, ensemble_mode_list,
                    plumed_colvars_list))  # list of if the form: [(molname, SMILES, moments), ...]
    for triplet in results:
        molname = triplet[0]
        SMILES = triplet[1]
        moments = triplet[2]
        molname_SMILES_moments_mdict[molname][SMILES] = moments

    if use_spectrophores:
        # Expand the moments by appending the Spectrophore vectors
        molname_spectrophoreVec_dict = get_spectrophores(LIGAND_STRUCTURE_FILE)
        molname_SMILES_moments_mdict = expand_momoments(molname_SMILES_moments_mdict,
                                                            molname_spectrophoreVec_dict)

    # # Serial execution for debugging
    # for a,b,c,d,e,f,g,h in zip(molecule_list, hydrogens_list, moment_number_list, onlyshape_list, molname_list, SMILES_list, ensemble_mode_list, plumed_colvars_list):
    #     molname, SMILES, moments = generate_moments(a,b,c,d,e,f,g)
    #     molname_SMILES_moments_mdict[molname][SMILES] = moments

    print("")  # just to change line
    return molname_SMILES_moments_mdict


def load_query_structures_and_calculate_moments(LIGAND_STRUCTURE_FILE,
                                                return_conformers=False,
                                                ensemble_mode=1,
                                                moment_number=4,
                                                onlyshape=False,
                                                plumed_colvars=False,
                                                keep_iso=True,
                                                use_spectrophores=True,
                                                get_SMILES=False):
    """
        keep_iso:   keep the "iso_[0-9]" suffix in the molname. Use with caution because if False only one of the isomers will be saved
    """
    ## Load compound structure file
    molname_SMILES_conformersMol_mdict = load_structure_file(LIGAND_STRUCTURE_FILE, keep_structvar=keep_iso, get_SMILES=get_SMILES)
    
    if plumed_colvars:
        if not os.path.exists("plumed_files"): os.mkdir("plumed_files")
        os.chdir("plumed_files")
        run_commandline(CONSSCORTK_BIN_DIR+"/../utilities/split_file.py -in "+LIGAND_STRUCTURE_FILE+" -otype pdb")
        
        os.chdir("../")

    # Now Generate Moments and save them in a common array for all the conformers of the same compound
    molname_SMILES_moments_mdict = calculate_moments(molname_SMILES_conformersMol_mdict,
                                                          ensemble_mode=ensemble_mode,
                                                          moment_number=moment_number,
                                                          onlyshape=onlyshape,
                                                          plumed_colvars=plumed_colvars,
                                                          use_spectrophores=use_spectrophores)

    if return_conformers:
        return molname_SMILES_moments_mdict, molname_SMILES_conformersMol_mdict
    else:
        return molname_SMILES_moments_mdict


# helper function to convert fingerprints from RDKit to numpy arrays


def fp_generator_2Dpp(mol, molname):
    """
        ATTENTION: This function presumes that each molname corresponds to ONLY ONE SMILES string!
        
        RETURNS:
        molname_key:        the molname key for molname_fingerprint_dict dictionary.
        feat_vec_2Dpp:      the final 2Dpp feature vector.
    """
    sys.stdout.write(molname + " ")
    sys.stdout.flush()

    fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory)
    molname_key = re.sub("_iso[0-9]+", "", molname)
    feat_vec_2Dpp = np.array([int(b) for b in fp.ToBitString()])    # convert the 2D pharmacophore fp to bit array
    
    return molname_key, feat_vec_2Dpp


def fp_generator_3Dpp(mol, molname, featvec_average_mode):
    """
        ATTENTION: This function presumes that each molname corresponds to ONLY ONE SMILES string!
        
        RETURNS:
        molname_key:        the molname key for molname_fingerprint_dict dictionary.
        feat_vec_3Dpp:      the final 3Dpp feature vector.
    """
    sys.stdout.write(molname + " ")
    sys.stdout.flush()
    
    if mol.GetNumConformers() == 1:
        fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory, dMat=Chem.Get3DDistanceMatrix(mol))
        molname_key = re.sub("_iso[0-9]+", "", molname)
        feat_vec_3Dpp = np.array([int(b) for b in fp.ToBitString()])    # convert the 3D pharmacophore fp to bit array
    elif mol.GetNumConformers() > 1:
        fingerprint_list = []   # list of the fingerprint(arrays of all the conformers
        confIds = [c.GetId() for c in mol.GetConformers()]  # get the Conformer IDs
        
        if featvec_average_mode == 0:    # keep only the last conformer (=last frame)
            Id = confIds[-1]
            fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory, dMat=Chem.Get3DDistanceMatrix(mol, confId=Id))
            molname_key = re.sub("_iso[0-9]+", "", molname)
            feat_vec_3Dpp = np.array([int(b) for b in fp.ToBitString()])
        elif featvec_average_mode in [1, 2, 3]:
            for Id in confIds: 
                fp = Generate.Gen2DFingerprint(mol, Gobbi_Pharm2D.factory, dMat=Chem.Get3DDistanceMatrix(mol, confId=Id))
                fingerprint_list.append( np.array([int(b) for b in fp.ToBitString()]) )   # convert the 3D pharmacophore fp to bit array
            if featvec_average_mode == 1:
                molname_key = re.sub("_iso[0-9]+", "", molname)
                # average the bits
                feat_vec_3Dpp = np.mean(fingerprint_list, axis=0)
            elif featvec_average_mode == 2:
                molname_key = re.sub("_iso[0-9]+", "", molname)
                # average and round the bits to become 0 or 1
                feat_vec_3Dpp = np.round( np.mean(fingerprint_list, axis=0) )
            elif featvec_average_mode == 3:  # calculate mean and stdev from ensebmle
                molname_key = re.sub("_iso[0-9]+", "", molname)
                feat_vec_3Dpp = np.array([e for duplet in zip(np.mean(fingerprint_list, axis=0), np.std(fingerprint_list, axis=0)) for e in duplet])
    
    return molname_key, feat_vec_3Dpp


class Mol2Vec():

    def __init__(self, model="mol2vec1000_1"):
        from gensim.models import word2vec
        # Load a pre-trained Mol2vec model which was trained on 20 million compounds downloaded from ZINC
        if os.environ.get('MOL2VEC_MODELS_DIR') is not None:
            dim, radius = model.replace("mol2vec", "").split('_')
            model = "model.dim%s.radius%s.pkl" % (dim, radius)
            print("Loading pre-trained Mol2vec model from file", os.environ.get('MOL2VEC_MODELS_DIR') +'/%s' % model)
            self.model = word2vec.Word2Vec.load(os.environ.get('MOL2VEC_MODELS_DIR') +'/%s' % model)
        else:   # load the default file that comes with the distribution
            print("Loading pre-trained Mol2vec model from file", CONSSCORTK_LIB_DIR + '/../data/%s' % model)
            self.model = word2vec.Word2Vec.load(CONSSCORTK_LIB_DIR + '/../data/%s' % model)

    def calc_mol2vec(self, molname_SMILES_conformersMol_mdict, get_dict=False):
        """
        ATTENTION: this function assumes that all SMILES keys in molname_SMILES_conformersMol_mdict are 'SMI'!
        """
        ColorPrint("Calculating mol2vec vectors for all compounds...", "OKGREEN")

        # Generate "molecular sentences" that are then used to featurize the molecules
        # (i.e. vectors of identifiers are extracted from Mol2vec model and summed up)
        mol2vec_list = []
        molname_mol_list = []
        for molname in list(molname_SMILES_conformersMol_mdict.keys()):
            SMILES = list(molname_SMILES_conformersMol_mdict[molname].keys())[0]  # use the first SMILES by default
            # SMILES = 'SMI'
            mol = molname_SMILES_conformersMol_mdict[molname][SMILES]
            molname_mol_list.append((str(molname), mol))
        df = pd.DataFrame.from_records(molname_mol_list, columns=["molname", "ROMol"])
        df['sentence'] = df.apply(lambda x: MolSentence(mol2alt_sentence(x['ROMol'], 1)), axis=1)
        df['mol2vec'] = [DfVec(x) for x in sentences2vec(df['sentence'], self.model, unseen='UNK')]
        mol2vec_list = [x.vec for x in df['mol2vec']]
        molnames_list = [m for m in df['molname']]

        molname_mol2vec_dict = {}
        for m,v in zip(molnames_list, mol2vec_list):
            molname_mol2vec_dict[m] = v

        if get_dict:
            return molname_mol2vec_dict
        else:
            return mol2vec_list

def calc_USR_sim(m1, m2):
    """
    Method to calculate the USRCAT distance. m1 should always be a 1x60 array. m2 can be an 1x60 array or an Nx60 array, where N is the
    number of conformers of the target compound or the moments of a set of screening compounds. In the latter case all N comparisons will
    be performed and the maximum similarity will be returned.
    """
    N = m1.shape[0] # number of moments
    
    if m2.shape == (N,):
        USRCAT_sim = 1.0/(1.0 + np.sum(np.absolute(m1-m2))/float(N) )
    else:
        mr1 = np.array( [m1.tolist()] * m2.shape[0])   # repeat m1 array N times (number of conformers of target compound)
        USRCAT_sim = np.max( 1.0/(1.0 + np.sum(np.absolute(mr1-m2), axis=1)/float(N)) )  # return the maximum similarity score
    
    return USRCAT_sim


def calc_USRCAT_sim(m1, m2, individual=[1,1,1,1,1]):
    """
    Method to calculate the USRCAT distance. m1 should always be a 1x60 array. m2 can be an 1x60 array or an Nx60 array, where N is the
    number of conformers of the target compound or the moments of a set of screening compounds. In the latter case all N comparisons will
    be performed and the maximum similarity will be returned.
    :param m1:
    :param m2:
    :param individual:
    :return:
    """
    ow = individual[0]  # shape
    hw = individual[1]  # hydrophobic atoms
    rw = individual[2]  # aromatic
    aw = individual[3]  # acceptors
    dw = individual[4]  # donors
    N = m1.shape[0]/5 # number of moments per feature
    if m1.shape[0]%5 == 0:
        sf = np.array([ow]*N + [hw]*N + [rw]*N + [aw]*N + [dw]*N)	# scaling factors array
    else:   # if this is an expanded moments vector
        sf = np.array([1]*m1.shape[0])
    
    if m2.shape == (N*5,):
        USRCAT_sim = 1.0/(1.0 + np.sum(sf * np.absolute(m1-m2))/float(N) )
    else:
        mr1 = np.array( [m1.tolist()] * m2.shape[0])   # repeat m1 array N times (number of conformers of target compound)
        USRCAT_sim = np.max( 1.0/(1.0 + np.sum(sf * np.absolute(mr1-m2), axis=1)/float(N)) )  # return the maximum similarity score
    
    return USRCAT_sim


def calc_USRCATsift_sim(m1, m2):
    """
        FUNCTION to calculate the distance between two 'moment_sift' hybrid fingerprints. For the moments part (first 60 numbers) the
        USRCAT distance is calculated, while for the SiFt part the Tanimoto distance is calculated (assuming that the SiFt is in the binary form).
        m1 should always be a 1D array. m2 can be an 1D array or an N-dimensional array.
    """
    
    if len(m2.shape) == 1:
        USRCAT_sim = 1.0/(1.0 + np.sum(np.absolute(m1[:60]-m2[:60]))/12.0 )
        sift_sim = 1.0 - distance.jaccard(m1[60:], m2[60:])
        USRCATsift_sim = np.mean([USRCAT_sim, sift_sim])
    else:
        USRCATsift_sim_list = []
        moments1 = m1[:60]
        sift1 = m1[60:]
        for row in range(m2.shape[0]):
            moments2 = m2[row][:60]
            sift2 = m2[row][60:]
            USRCAT_sim = 1.0/(1.0 + np.sum(np.absolute(moments1-moments2))/12.0 )
            sift_sim = 1.0 - distance.jaccard(sift1, sift2)
            sim = np.mean([USRCAT_sim, sift_sim])
            USRCATsift_sim_list.append(sim)
        USRCATsift_sim = np.max(USRCATsift_sim_list)    # return the maximum similarity score
    
    return USRCATsift_sim


def normalize_USRCAT_moments(data, mean_array=np.array([]), stdev_array=np.array([])):
    """
        FUNCTION to normalize each moment of an array of USRCAT moments. If provided, mean_array and stdev_array values will be used for
        normalization, otherwise they will be calculated for every moments independently.
        ARGS:
        data:   list or array with the moments
    """
    norm_data = np.array(data)
    colNum = norm_data.shape[1]  # the number of columns of the input data, presuming that they are 1D arrays or 1D lists
    if mean_array.shape == (0,) and stdev_array.shape == (0,):
        mean_array = np.zeros([colNum])
        stdev_array = np.zeros([colNum])
        for col in range(colNum):
            mean_array[col] = np.mean(norm_data[:, col])
            stdev_array[col] = np.std(norm_data[:, col])
            if stdev_array[col] == 0.0:
                stdev_array[col] = 1.0  # to avoid inf value in normalization
                # mean_array[col] = 0.0
            
        norm_data = norm_data - mean_array
        norm_data = norm_data / stdev_array
        norm_data = np.nan_to_num(norm_data)
        return norm_data, mean_array, stdev_array
    
    elif mean_array.shape != (0,) and stdev_array.shape != (0,):
        norm_data = norm_data - mean_array
        norm_data = norm_data / stdev_array
        return norm_data


def sort_inactives_by_active_resemblance(molname_SMILES_moments_mdict, trainmol_activity_dict):
    active_moments_list = []
    inactive_moments_list = []
    active_names_list = []
    inactive_names_list = []
    for molname in list(trainmol_activity_dict.keys()):
        if trainmol_activity_dict[molname] == 0:
            inactive_names_list.append(molname)
            smi = [k for k in list(molname_SMILES_moments_mdict[molname].keys()) if k!=0][0]
            inactive_moments_list.append(molname_SMILES_moments_mdict[molname][smi][0].tolist())
        elif trainmol_activity_dict[molname] == 1:
            active_names_list.append(molname)
            smi = [k for k in list(molname_SMILES_moments_mdict[molname].keys()) if k!=0][0]
            active_moments_list.append(molname_SMILES_moments_mdict[molname][smi][0].tolist())
    
    inactiveMinSimTuple_list = [] # [(imolname, imoment, minimum similarity to all actives), ...]
    for iname, imoment in zip(inactive_names_list, inactive_moments_list):
        sim_list = []
        for aname,amoment in zip(active_names_list, active_moments_list):
            sim_list.append(calc_USRCAT_sim(imoment, amoment))
        inactiveMinSimTuple_list.append( (iname , imoment, np.min(sim_list)) )
    
    inactiveMinSimTuple_list.sort(key=itemgetter(2), reverse=True)
    return inactiveMinSimTuple_list


def calc_USRCAT_sim_list(molname_SMILES_moments_mdict, sorted_ligand_experimentalE_dict, query_molname, USRCAT_weights=[1,1,1,1,1],
                         is_aveof=False, query_molfile=None, return_molnames=False, moment_number=4, onlyshape=False):
    """
        FUNCTION to calculate the USRC distance from the lowest energy conformation (usually crystal conformation) of a query ligand, of all
        isomers and conformers of each target compound.
        RETURN:
        reordered_USRCATsim_list:   list of USRCAT similarities (same order as reordered_experimentalE_list but without the query ligand)
        reordered_experimentalE_list:   list of Exp DeltaG (same order as reordered_USRCATsim_list but without the query ligand)
    """
    
    if query_molfile:
        qmol=Chem.MolFromMol2File(query_molfile)
        query_molname = qmol.GetProp('_Name')
        qmoment = generate_moments(qmol, moment_number=moment_number, onlyshape=onlyshape, ensemble_mode=1)
    else:
        if not query_molname in list(molname_SMILES_moments_mdict.keys()):    # if this ligand is not in the sdf file
            return False
        query_SMILES_list = list(molname_SMILES_moments_mdict[query_molname].keys())
        qmoment = molname_SMILES_moments_mdict[query_molname][query_SMILES_list[0]][0]  # keep the moment of the lowest energy conformer
        
    molname_USRCATsim_dict = {} # target molname-> USRCAT distance from query ligand
    for molname in list(molname_SMILES_moments_mdict.keys()):
        for SMILES in list(molname_SMILES_moments_mdict[molname].keys()):
            USRCATsim_list = []
            for target_moments in molname_SMILES_moments_mdict[molname][SMILES]:
                USRCATsim_list.append(calc_USRCAT_sim(qmoment, target_moments, USRCAT_weights))
            molname_USRCATsim_dict[molname] = np.max(USRCATsim_list)    # keep the max similarity with all the isomers of molname
    
    molname_list = []   # contains the common molnames between molname_USRCATsim_dict and sorted_ligand_experimentalE_dict
    for k in list(sorted_ligand_experimentalE_dict.keys()):
        if k in list(molname_USRCATsim_dict.keys()):
            molname_list.append(k)
        # ADD A WARNING HERE ! # else: 
        #     print("WARNING..."))
    if is_aveof == True:    # Deactivate the following for the MinRank Method
        if query_molname in molname_list:
            molname_list.remove(query_molname)
    if is_aveof == False:   # only from 'minrank' and 'consscortk' scoring schemes
        molname_USRCATsim_dict[query_molname] = -1.0 # set the similarity of query ligand to -1 to avoid bias in the ranking
    reordered_USRCATsim_list = []
    reordered_experimentalE_list = []
    for molname in molname_list:
        reordered_USRCATsim_list.append(molname_USRCATsim_dict[molname])
        reordered_experimentalE_list.append(sorted_ligand_experimentalE_dict[molname])
    
    if return_molnames == True:
        return reordered_USRCATsim_list, reordered_experimentalE_list, molname_list
    else:
        # return 2 lists in the same order, the USRCAT similarities and the Exp DeltaG
        return reordered_USRCATsim_list, reordered_experimentalE_list


def measure_subset_diversity(similarity_matrix, subset_actives_list, actives_list, between=False):
    """
    similarity_matrix, molnames_crossval, molnames_xtest
    actives_list:   must be the list with the molnames in the same order as they are in similarity_matrix. subset_actives_list mus be
                    a subset of actives_list!
    between:    if True, then the average similarity between the non-common members of subset_actives_list and actives_list is returned.
                if False, then the average similarity between subset_actives_list is returned.
    """
    if subset_actives_list != actives_list:
        if not between:
            subset_indices_list  = [actives_list.index(n) for n in subset_actives_list]
        else:
            subset_indices_list  = [actives_list.index(n) for n in subset_actives_list]
            nonsubset_indices_list  = [actives_list.index(n) for n in actives_list if not n in subset_actives_list]
    else:
        subset_indices_list = list(range(len(actives_list)))
    
    fp_sim_list = []
    if not between:
        for c in combinations(subset_indices_list, 2):
            fp_sim_list.append(similarity_matrix[c[0], c[1]])
    else:
        for si in subset_indices_list:
            for nsi in nonsubset_indices_list:
                fp_sim_list.append(similarity_matrix[si, nsi])
    
    return np.mean(fp_sim_list), np.std(fp_sim_list)


def calc_Objective_Function(OBJECTIVE_FUNCTION_LIST, USRCATsim_array, ExpDG_array, ENERGY_THRESHOLD,
                            reordered_molname_USRCATsim_dict=None, sorted_ligand_experimentalE_dict=None,
                            molname_list=[], actives_list=[], molname2scaffold_dict={}):
    
    coefficient_tuple = tuple()
    AU_ROC, AU_CROC, croc_AU_ROC, croc_BEDROC, EF1, EF5, EF10 = -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0
    Curves = None
    for OBJECTIVE_FUNCTION in OBJECTIVE_FUNCTION_LIST:
        if ("Pearson" == OBJECTIVE_FUNCTION):
            R, R_Pvalue = pearsonr(list(reordered_molname_USRCATsim_dict.values()), list(sorted_ligand_experimentalE_dict.values()))
            coefficient_tuple += (R,)
        elif("TopDown_Concordance" == OBJECTIVE_FUNCTION):
            C, C_Pvalue = TopDown_Concordance2(reordered_molname_USRCATsim_dict, sorted_ligand_experimentalE_dict)
            coefficient_tuple += (C,)
        elif("Kendalls_tau" == OBJECTIVE_FUNCTION):
            tau, tau_Pvalue = Kendalls_tau(reordered_molname_USRCATsim_dict, sorted_ligand_experimentalE_dict)
            coefficient_tuple += (tau,)
        elif("Kendalls_W" == OBJECTIVE_FUNCTION):
            W, W_P_value = Kendalls_W(reordered_molname_USRCATsim_dict, sorted_ligand_experimentalE_dict)
            coefficient_tuple += (W,)
        elif ("ROC" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            AU_ROC = Curves.ROC_curve()
            coefficient_tuple += (AU_ROC,)
        elif ("CROC" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            AU_CROC = Curves.CROC_curve()
            coefficient_tuple += (AU_CROC,)
        elif ("BEDROC" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            croc_BEDROC = Curves.BEDROC_curve()
            coefficient_tuple += (croc_BEDROC,)
        elif ("EF1" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            EF1 = Curves.Enrichment_Factor(1)[0]    # save only the normalized EF, not the N_sel
            coefficient_tuple += (EF1,)
        elif ("EF5" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            EF5 = Curves.Enrichment_Factor(5)[0] # save only the normalized EF, not the N_sel
            coefficient_tuple += (EF5,)
        elif ("EF10" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            EF10 = Curves.Enrichment_Factor(10)[0] # save only the normalized EF, not the N_sel
            coefficient_tuple += (EF10,)
        elif ("scaffold_EF1" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            scaffold_EF1 = Curves.scaffold_Enrichment_Factor(1)[0]    # save only the normalized EF, not the N_sel
            coefficient_tuple += (scaffold_EF1,)
        elif ("scaffold_EF5" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            scaffold_EF5 = Curves.scaffold_Enrichment_Factor(5)[0] # save only the normalized EF, not the N_sel
            coefficient_tuple += (scaffold_EF5,)
        elif ("scaffold_EF10" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            scaffold_EF10 = Curves.scaffold_Enrichment_Factor(10)[0] # save only the normalized EF, not the N_sel
            coefficient_tuple += (scaffold_EF10,)
        elif ("croc_ROC" == OBJECTIVE_FUNCTION):
            if Curves == None:
                Curves = Create_Curve(ExpDG_array, USRCATsim_array, ENERGY_THRESHOLD, actives_list=actives_list, molname_list=molname_list, molname2scaffold_dict=molname2scaffold_dict)
            croc_AU_CROC = Curves.croc_ROC_curve()
            coefficient_tuple += (croc_AU_CROC,)
    
    return coefficient_tuple


def calc_USRCAT_statistics(molname_SMILES_moments_mdict, USRCAT_weights, query_ligands_list, OBJECTIVE_FUNCTION_LIST,
                            sorted_ligand_experimentalE_dict, ENERGY_THRESHOLD, return_scores=False, write_csv_matrix=False):
    """
        FUNCTION to calculate statistics like AUC-ROC by running multiple times USRCAT, each time using a different active compound as a query
        ligand, and at the end averaging the values of these statistics
        ARGUMENTS:
        query_ligand_list:  list of molnames to be used as query ligands. If an empty list then all the actives in the activity files
                            will be used as queries.
    """
    
    coefficient_tuples_list = []    # list of coefficient tuples to be averaged at the end
    for query_molname in query_ligands_list:
        results = calc_USRCAT_sim_list(molname_SMILES_moments_mdict, sorted_ligand_experimentalE_dict, query_molname, USRCAT_weights, is_aveof=True)
        if results == False:
            continue
        USRCATsim_array = np.array(results[0])
        ExpDG_array = np.array(results[1])
        coefficient_tuple = calc_Objective_Function(OBJECTIVE_FUNCTION_LIST, USRCATsim_array, ExpDG_array, ENERGY_THRESHOLD)
        coefficient_tuples_list.append(coefficient_tuple)
    
    # calculate the average coefficient values and their stdev
    coefficient_tuples_array = np.array(coefficient_tuples_list)
    mean_array = np.mean(coefficient_tuples_array, axis=0)
    std_array = np.std(coefficient_tuples_array, axis=0)
    aveStd_coefficient_list = []
    for m,s in zip(mean_array, std_array):
        aveStd_coefficient_list.extend([round(m, 8), round(s, 8)])
    aveStd_coefficient_tuple = tuple(aveStd_coefficient_list)
    #    print("coefficient_tuple is ", coefficient_tuple))
    return aveStd_coefficient_tuple   # ATTENTION: YOU MUST RETURN THE FITNESS VALUE AS A TUPLE TO BE READABLE BY DEAP !!! FOR 1 QUANTITY USE A TRAILING ",", E.G. "return C,"


def calc_BordaCount(USRCATsim_matrix):
    colNum = USRCATsim_matrix.shape[1]
    rowNum = USRCATsim_matrix.shape[0]
    borda_coeff_array = np.zeros(colNum)
    borda_coeff_array.fill(rowNum**5)   # exponent 5 adds more weight to the top ranked predictors
    for k in reversed(list(range(1, rowNum))):
        extra_row = np.zeros(colNum)
        extra_row.fill(k**5)
        borda_coeff_array = np.vstack([borda_coeff_array, extra_row]) # stack new row of Borda coefficients
    USRCATsim_matrix.sort(axis=0)   # sort each column in ascending order
    USRCATsim_matrix = np.flipud(USRCATsim_matrix) # flip rows so than columns are sorted in descending order (highest to lowest value)
    USRCATsim_array = np.sum(USRCATsim_matrix * borda_coeff_array, axis=0)   # x-1 to make the top scored compounds have the lowest score
                                                                                     # for correct statistics calculation
    return USRCATsim_array


def report_Fingerprint(query_molname, featvec_type, fpsim_list, ExpDG_list, molname_list, append=False):
    """ Function to write a detailed report with the Consensus Scores assigned to the best isomer per ligand """
    
    if not os.path.exists("reports"):
        os.mkdir("reports")
    
    fname = "reports/"+featvec_type+"_Report.dat"
    title = "QUERY_MOL\t\tSCORING_FUNCTION\t\tLIGAND\t\tBEST_ISOMER\t\tBEST_POSE\t\tENERGY/SCORE\t\tZ-SCORE\t\tEXPERIMENTAL_ENERGY\t\tZ-SCORE\n"
    
    if append == False and os.path.exists(fname):   # remove previous files to prevent appending
        os.remove(fname)
    if not os.path.exists("reports/"):
        os.mkdir("reports/")
    f = open(fname, 'a')
    f.write(title)
    for ligand,fpsim,ExpDG in zip(molname_list, fpsim_list, ExpDG_list):
        f.write(query_molname+"\t\t"+featvec_type+"\t\t"+str(ligand)+"\t\tN/A\t\t"\
                "N/A\t\t"+str(fpsim)+"\t\tN/A\t\t"\
                +str(ExpDG)+"\t\tN/A\n")
    f.flush()
    f.close()


def get_threshold_value_index(sim_list, certain_active_threshold=0.5):
    """
        FUNCTION to find the index of the value in a list that is closer and >= than the specified certain_active_threshold. E.g. if
        certain_active_threshold=0.5 and sim_list = [0.12, 0.45, 0.52, 0.67], then the function will return 2 which correspond to value 0.52.
    """
    sim_array = np.array(sim_list)
    N = sim_array.shape[0]
    diff_array = sim_array - np.array([certain_active_threshold]*N)
    diff_array.sort()
    for i in range(N):
        if diff_array[i] >= 0.0:
            print(i)
            break


def calc_USRCAT_min_statistics(molname_SMILES_conformersMol_mdict,
                               molname_SMILES_moments_mdict,
                               USRCAT_weights,
                               query_molnames_list,
                               OBJECTIVE_FUNCTION_LIST,
                               sorted_ligand_experimentalE_dict,
                               ENERGY_THRESHOLD,
                               return_scores=False,
                               write_csv_matrix=False,
                               is_minrank=False,
                               actives_list=[],
                               molname2scaffold_dict=[]):
    """
        FUNCTION to calculate statistics like AUC-ROC by running multiple times USRCAT, each time using a different active compound as a query
        ligand, and at the end averaging the values of these statistics
        ARGUMENTS:
        query_ligand_list:  list of molnames to be used as query ligands. If an empty list then all the actives in the activity files
                            will be used as queries.
    """
    
    original_molname_list = list(sorted_ligand_experimentalE_dict.keys())
    rowNum = len(query_molnames_list) # the rows of the USRCATsim_matrix
    colNum = len(original_molname_list) # the columns of the USRCATsim_matrix
    if molname_SMILES_conformersMol_mdict != {}:    # twice rows for the Fingerprint(similarities
        rowNum = 5*rowNum
    USRCATsim_matrix = np.zeros([rowNum, colNum])
    row = 0
    for featvec_type in ["USRCAT", "Morgan3"]:
        if os.path.exists("reports/"+featvec_type+"_Report.dat"):   # remove previous files to prevent appending
            os.remove("reports/"+featvec_type+"_Report.dat")
    scoreThreshold_list = []    # this list stores for every row of USRCATsim_matrix the Zscore or rank that corresponds to the
                                # certain_active_threshold (usually 0.5)
    certain_actives_indices_set = set()
    certain_active_threshold = 0.4  # every mol that has Morgan3 sim about this threshold will have its maximum Zscore value its whole column in USRCATsim_matrix
    for query_molname in query_molnames_list:
        USRCATsim_list, ExpDG_list, molname_list = calc_USRCAT_sim_list(molname_SMILES_moments_mdict, sorted_ligand_experimentalE_dict, query_molname, USRCAT_weights, return_molnames=True)
        # [certain_actives_indices_set.add(i) for i,sim in enumerate(USRCATsim_list) if sim>=certain_active_threshold]
        if is_minrank == False: # Borda count Method (superior using -1*Z-scores rather than -1*ranks)
            USRCATsim_matrix[row] = np.array(zscore(USRCATsim_list)) # highest Zscores to highest similarities
        else:   ## Min Rank Method (superior using -1*ranks rather than Z-scores)
            USRCATsim_matrix[row] = -1*np.array(rankdata(USRCATsim_list)) # highest ranks go to the highest similarities!!!
        row += 1
        report_Fingerprint(query_molname, "USRCAT", USRCATsim_list, ExpDG_list, molname_list, append=True)
        if molname_SMILES_conformersMol_mdict != {}:
            FPsim_list, ExpDG_list, molname_list = calc_Fingeprint_sim_list(molname_SMILES_conformersMol_mdict, sorted_ligand_experimentalE_dict, query_molname, featvec_type="Morgan3", return_molnames=True)
            [certain_actives_indices_set.add(i) for i,sim in enumerate(FPsim_list) if sim>=certain_active_threshold]
            if is_minrank == False: # Borda count Method (superior using -1*Z-scores rather than -1*ranks)
                USRCATsim_matrix[row] = np.array(zscore(FPsim_list)) # highest Zscores to highest similarities
            else:   ## Min Rank Method (superior using -1*ranks rather than Z-scores)
                USRCATsim_matrix[row] = -1*np.array(rankdata(FPsim_list)) # highest ranks go to Morgan3 highest similarities!!!
            row += 1
            report_Fingerprint(query_molname, "Morgan3", FPsim_list, ExpDG_list, molname_list, append=True)
        if molname_SMILES_conformersMol_mdict != {}:
            [certain_actives_indices_set.add(i) for i,sim in enumerate(FPsim_list) if sim>=0.65]
            FPsim_list, ExpDG_list, molname_list = calc_Fingeprint_sim_list(molname_SMILES_conformersMol_mdict, sorted_ligand_experimentalE_dict, query_molname, featvec_type="RDK5", return_molnames=True)
            if is_minrank == False: # Borda count Method (superior using -1*Z-scores rather than -1*ranks)
                USRCATsim_matrix[row] = np.array(zscore(FPsim_list)) # highest Zscores to highest similarities
            else:   ## Min Rank Method (superior using -1*ranks rather than Z-scores)
                USRCATsim_matrix[row] = -1*np.array(rankdata(FPsim_list)) # highest ranks go to the highest similarities!!!
            row += 1
            report_Fingerprint(query_molname, "RDK5", FPsim_list, ExpDG_list, molname_list, append=True)
        if molname_SMILES_conformersMol_mdict != {}:
            [certain_actives_indices_set.add(i) for i,sim in enumerate(FPsim_list) if sim>=0.8]
            FPsim_list, ExpDG_list, molname_list = calc_Fingeprint_sim_list(molname_SMILES_conformersMol_mdict, sorted_ligand_experimentalE_dict, query_molname, featvec_type="MACCS", return_molnames=True)
            if is_minrank == False: # Borda count Method (superior using -1*Z-scores rather than -1*ranks)
                USRCATsim_matrix[row] = np.array(zscore(FPsim_list)) # highest Zscores to highest similarities
            else:   ## Min Rank Method (superior using -1*ranks rather than Z-scores)
                USRCATsim_matrix[row] = -1*np.array(rankdata(FPsim_list)) # highest ranks go to the highest similarities!!!
            row += 1
            report_Fingerprint(query_molname, "MACCS", FPsim_list, ExpDG_list, molname_list, append=True)
        if molname_SMILES_conformersMol_mdict != {}:
            FPsim_list, ExpDG_list, molname_list = calc_Fingeprint_sim_list(molname_SMILES_conformersMol_mdict, sorted_ligand_experimentalE_dict, query_molname, featvec_type="AP", return_molnames=True)
            [certain_actives_indices_set.add(i) for i,sim in enumerate(FPsim_list) if sim>=0.5]
            if is_minrank == False: # Borda count Method (superior using -1*Z-scores rather than -1*ranks)
                USRCATsim_matrix[row] = np.array(zscore(FPsim_list)) # highest Zscores to highest similarities
            else:   ## Min Rank Method (superior using -1*ranks rather than Z-scores)
                USRCATsim_matrix[row] = -1*np.array(rankdata(FPsim_list)) # highest ranks go to the highest similarities!!!
            row += 1
            report_Fingerprint(query_molname, "AP", FPsim_list, ExpDG_list, molname_list, append=True)
    
    if is_minrank == False:    # Borda count Method (superior using -1*Z-scores rather than -1*ranks)
        borda_coeff_array = np.zeros(colNum)
        borda_coeff_array.fill(rowNum**5)   # exponent 5 adds more weight to the top ranked predictors
        for k in reversed(list(range(1, rowNum))):
            extra_row = np.zeros(colNum)
            extra_row.fill(k**5)
            borda_coeff_array = np.vstack([borda_coeff_array, extra_row]) # stack new row of Borda coefficients
        USRCATsim_matrix.sort(axis=0)   # sort each column in ascending order
        USRCATsim_matrix = np.flipud(USRCATsim_matrix) # flip rows so than columns are sorted in descending order (highest to lowest value)
        for col in certain_actives_indices_set:
            maxsim = USRCATsim_matrix[0, col]   # maximum similarity value of this columns that exceeds the certain_active_threshold
            USRCATsim_matrix[:, col] = maxsim   # set all rows of this column to the maximum similarity value in the column
        USRCATsim_array = np.sum(-1*USRCATsim_matrix * borda_coeff_array, axis=0)   # x-1 to make the top scored compounds have the lowest score
                                                                                     # for correct statistics calculation
    else:    ## Min Rank Method (superior using -1*ranks rather than Z-scores)
        USRCATsim_array = np.min(USRCATsim_matrix, axis=0)  # keep only the minimum Z-score (normalized similarity) value per compound
    
    ExpDG_array = np.array(ExpDG_list)  # don't touch this !
    # print("DEBUG: USRCATsim_array=", USRCATsim_array.tolist()))
    # print("DEBUG: ExpDG_array=", ExpDG_array.tolist()))
    coefficient_tuple = calc_Objective_Function(OBJECTIVE_FUNCTION_LIST, USRCATsim_array, ExpDG_array, ENERGY_THRESHOLD, molname_list=molname_list, actives_list=actives_list, molname2scaffold_dict=molname2scaffold_dict)
    
    return coefficient_tuple   # ATTENTION: YOU MUST RETURN THE FITNESS VALUE AS A TUPLE TO BE READABLE BY DEAP !!! FOR 1 QUANTITY USE A TRAILING ",", E.G. "return C,"


def calc_USRCAT_statistics_singleqmol(molname_SMILES_moments_mdict, USRCAT_weights, query_molnames_list, OBJECTIVE_FUNCTION_LIST,
                            sorted_ligand_experimentalE_dict, ENERGY_THRESHOLD, return_scores=False, write_csv_matrix=False):
    
    query_molname = query_molnames_list[0]
    results = calc_USRCAT_sim_list(molname_SMILES_moments_mdict, sorted_ligand_experimentalE_dict, query_molname, USRCAT_weights)
    # if results == False:
    #     continue
    # USRCATsim_array = -1*np.array(rankdata(results[0]))  # highest ranks to highest similarities
    USRCATsim_array = -1*np.array(zscore(results[0])) # highest Zscores to highest similarities (superior for Borda count!!!)
    experimentalE_array = np.array(results[1])  # convert the list of experimental DeltaG to array
    
    ## Borda count Method (superior using -1*Z-scores rather than -1*ranks)
    colNum = USRCATsim_array.shape[0]
    rowNum = 1
    
    return USRCATsim_array, experimentalE_array


def calc_USRCAT_statistics_qclusters(molname_SMILES_moments_mdict, USRCAT_weights, query_ligands_list, OBJECTIVE_FUNCTION_LIST,
                            sorted_ligand_experimentalE_dict, ENERGY_THRESHOLD, return_scores=False, write_csv_matrix=False):
    """
        FUNCTION to calculate statistics like AUC-ROC by running multiple times USRCAT, each time using a different active compound as a query
        ligand, and at the end averaging the values of these statistics
        ARGUMENTS:
        query_ligand_list:  list of molnames to be used as query ligands. If an empty list then all the actives in the activity files
                            will be used as queries.
    """
    
    original_molname_list = list(sorted_ligand_experimentalE_dict.keys())
    rowNum = len(query_ligands_list) # the rows of the USRCATsim_matrix
    colNum = len(original_molname_list) # the columns of the USRCATsim_matrix
    USRCATsim_matrix = np.zeros([rowNum, colNum])
    row = 0
    for query_molname in query_ligands_list:
        USRCATsim_list, ExpDG_list, molname_list = calc_USRCAT_sim_list(molname_SMILES_moments_mdict, sorted_ligand_experimentalE_dict, query_molname, USRCAT_weights, return_molnames=True)
        USRCATsim_matrix[row] = np.array(USRCATsim_list) # populate the rows
        row += 1
    USRCATsim_array = np.min(USRCATsim_matrix, axis=0)
    ExpDG_array = np.array(ExpDG_list)
    
    return USRCATsim_array, ExpDG_array
