from multiprocessing import cpu_count
from lib.molfile.ligfile_parser import *
from lib.utils.print_functions import ColorPrint

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, SDWriter
    from rdkit.Chem.Lipinski import RotatableBondSmarts, NumRotatableBonds
except ImportError:
    print("WARNING: rdkit module could not be found!")


def write_centroids_to_sdf(centroids_sdf,
                           valid_confs_sdf,
                           RDKitMol,
                           clustersizes_dict,
                           centroidID_energy_dict,
                           clusterAssignments_dict,
                           kekulize):
    print("DEBUG: centroidID_energy_dict=", centroidID_energy_dict)
    print_dict_contents(centroidID_energy_dict)
    print("DEBUG: clustersizes_dict=", clustersizes_dict)
    print_dict_contents(clustersizes_dict)
    with open(centroids_sdf, 'a') as cf:
        writer = SDWriter(cf)
        writer.SetKekulize(kekulize)
        writer.SetProps(['molecular energy', "cluster size"])   # write only these 2 properties
        for confID, energy in centroidID_energy_dict.items():
            RDKitMol.SetProp("cluster size", str(clustersizes_dict[confID]))
            # RDKitMol.SetProp("molecular energy", str(energy))
            # for member_confID in clusterAssignments_dict[confID]:
            writer.write(RDKitMol, confId=confID)
            writer.flush()
        writer.close()
    if valid_confs_sdf:
        # Write all valid conformers below the energy thresholds
        with open(valid_confs_sdf, 'a') as cf:
            writer = SDWriter(cf)
            writer.SetKekulize(kekulize)
            writer.SetProps(['molecular energy', "cluster size"])   # write only these 2 properties
            for confID, energy in centroidID_energy_dict.items():
                RDKitMol.SetProp("cluster size", str(clustersizes_dict[confID]))
                # RDKitMol.SetProp("molecular energy", str(energy))
                for member_confID in clusterAssignments_dict[confID]:
                    writer.write(RDKitMol, confId=member_confID)
                    writer.flush()
            writer.close()

def calc_ligEntropy(mol,
                    numConfs,
                    seed=2019,
                    CPUs=cpu_count(),
                    descriptor="confS",
                    SCHRODINGER=False,
                    RMSD_CUTOFF=1.0,
                    DELTAG_CUTOFF=1000000,
                    MAX_CONFNUM=1000000,
                    centroids_sdf=None,
                    valid_confs_sdf=None,
                    kekulize=False):
    """
    Calculates the descriptor nConf20 described in the paper:

    "Beyond Rotatable Bond Counts: Capturing 3D Conformational Flexibility in a Single Descriptor",
    Jerome G. P. Wicker and Richard I. Cooper.  J. Chem. Inf. Model. 2016, 56, 2347-2352

    :param mol:
    :param numConfs:
    :param seed:    the final number changes a lot wrt the seed. Only if you increase numConfs >=500 you narrow down the differences.
    :return confS:  in units kcal/mol
    """
    confID_energy_dict = {}  # stores the conformer IDs that succeed in minimization along with their energy
    if SCHRODINGER:
        # STEP 1: load the conformers generated with Macromodel that contain their energy as a property
        molecule = mol
        for prop in mol.GetProp('conf_r_mmod_Potential_Energy-OPLS-2005').split('|'):
            confID, energy = prop.split(':')
            confID_energy_dict[int(confID)] = float(energy)
    else:
        # STEP 1: generate conformers and minimize them (not applicable to SCHRODINGER==True)
        if type(mol) == str:  # if the input is a SMILES string
            mol = Chem.MolFromSmiles(mol)
        molecule = Chem.AddHs(mol)
        conformerIntegers = []
        conformerIDs = AllChem.EmbedMultipleConfs(molecule, numConfs, pruneRmsThresh=0.5, numThreads=CPUs, randomSeed=seed)
        if len(conformerIDs) == 0:
            ColorPrint("FAIL: no conformations could be generated for molecule %s!" % molecule.GetProp("_Name"), "OKRED")
            return False
        attempts = 1
        while len(conformerIntegers) == 0 and attempts <= 3:
            if attempts > 1:
                print("This is attempt No %i  to minimize conformers of this molecule." % attempts)
            optimized_and_energies = AllChem.MMFFOptimizeMoleculeConfs(molecule,
                                                                       maxIters=2000,
                                                                       numThreads=CPUs,
                                                                       nonBondedThresh=100.0)
            for confID in conformerIDs:
                not_optimized, energy = optimized_and_energies[confID]
                if not_optimized == 0:  # the minimization for that conformer converged.
                    confID_energy_dict[confID] = energy
                    conformerIntegers.append(confID)
            attempts += 1

        if len(conformerIntegers) == 0:
            ColorPrint("FAIL: no minimized conformations of molecule %s were produced!" % molecule.GetProp("_Name"), "OKRED")
            return False

    # STEP 2: find the lowest energy conformer and sort conformers by energy
    conf_energy_list = [(c, e) for c, e in confID_energy_dict.items()]
    conf_energy_list.sort(key=itemgetter(1))
    confID_energy_dict = OrderedDict(conf_energy_list) # the keys now are ordered by their energy
    lowestenergy = min(confID_energy_dict.values())
    for k, v in (list(confID_energy_dict.items())):
        if v == lowestenergy:
            lowestEnergyConformerID = k
    if DELTAG_CUTOFF < 1000000:
        # Delete conformers with DeltaG from the lowest Energy conformer above the given cutoff
        copy_confID_energy_dict = confID_energy_dict.copy()
        invalid_confnum = 0
        max_energy_threshold = lowestenergy + DELTAG_CUTOFF
        for confID, energy in copy_confID_energy_dict.items():
            if energy > max_energy_threshold:
                del confID_energy_dict[confID]
                invalid_confnum += 1
        ColorPrint("\tThe lowest energy conformer had molecular energy %f. Deleted %i conformers with energy above %f."
                   " The number of conformers was reduced from %i to %i." %
                   (lowestenergy, invalid_confnum, max_energy_threshold, len(copy_confID_energy_dict),
                    len(confID_energy_dict)), "OKBLUE")
    if MAX_CONFNUM < 1000000:
        copy_confID_energy_dict = confID_energy_dict.copy()
        for i, confID in enumerate(copy_confID_energy_dict.keys()):
            if i >= MAX_CONFNUM:
                del confID_energy_dict[confID]
        ColorPrint("Retained only the lowest energy %i conformers. Their number was reduced from %i to %i." %
                   (MAX_CONFNUM, len(copy_confID_energy_dict), len(confID_energy_dict)), "OKBLUE")

    centroidID_energy_dict = OrderedDict()  # the confIDs of the centroids and their potential energy
    centroidID_energy_dict[lowestEnergyConformerID] = lowestenergy    # save the lowest energy conformer first

    # STEP 3: cluster conformers into microstates. The centroid of each mictostate is the conformer with the lowest energy.
    molecule = AllChem.RemoveHs(molecule, sanitize=False)   # remove Hs before alignment & RMSD calculation
    # ATTENTION: we are comparing conformers of the same molecules, therefore make sure you match exactly the same atoms!
    # ATTENTION: (no need for molecule.GetSubstructMatches; if the molecule is symmetrical you will get spurious results).
    maps = [[(a.GetIdx(), a.GetIdx()) for a in molecule.GetAtoms()]]
    # First iterate over all minimized conformers in energy order to find cluster centroids.
    for confID in list(confID_energy_dict.keys()):
        okayToAdd = True
        for centroidID in centroidID_energy_dict.keys():
            # Get the lowest possible RMSD between the finalconformerID (probe) and conformerID (ref)
            try:
                RMS = AllChem.GetBestRMS(molecule, molecule, centroidID, confID, maps)
            except ValueError:
                ColorPrint("WARNING: Bad Conformer Id %i" % confID, "WARNING")
                continue
            if RMS < RMSD_CUTOFF:   # if a similar conformer with lower energy was found, then skip the current conformer
                okayToAdd = False
                break
        if okayToAdd:
            centroidID_energy_dict[confID] = confID_energy_dict[confID]
    # Then iterate over all minimized conformers in energy order to assign them to clusters based on their distance
    # with each centroid.
    clusterAssignments_dict = {c:[c] for c in centroidID_energy_dict.keys()}   # confID of centroid -> confIDs of its members.
    clustersizes_dict = {c:1 for c in centroidID_energy_dict.keys()}    # confID of centroid -> cluster population. This dict is sorted by
                                                                        # the energy of the centroid of each cluster
    for confID in list(confID_energy_dict.keys()):
        if confID in centroidID_energy_dict.keys():
            continue    # skip centroids, we already know their clusterID
        RMS_centroidID_list = []
        for centroidID in centroidID_energy_dict.keys():
            RMS = AllChem.GetBestRMS(molecule, molecule, centroidID, confID, maps)
            if RMS < RMSD_CUTOFF:
                RMS_centroidID_list.append( (RMS, centroidID) )
        if len(RMS_centroidID_list) > 0:
            RMS_centroidID_list.sort(key=itemgetter(0))
            closest_centroidID = RMS_centroidID_list[0][1]
            clustersizes_dict[closest_centroidID] += 1  # increase the population of the cluster with the closest centroid
            clusterAssignments_dict[closest_centroidID].append(confID)
    ColorPrint("\t%i microstates were found with populations: %s" % (len(clustersizes_dict),
        ", ".join(map(str, sorted(list(clustersizes_dict.values()), reverse=True))) ), "OKBLUE")

    if centroids_sdf or valid_confs_sdf:
        write_centroids_to_sdf(centroids_sdf,
                               valid_confs_sdf,
                               molecule,
                               clustersizes_dict,
                               centroidID_energy_dict,
                               clusterAssignments_dict,
                               kekulize)

    if "_confS" in descriptor:
        # STEP 4: calculate the conformational Entropy
        # NOTE: for the ligand conformational entropy what we calculate is the S of the free state. However, this is
        # NOTE: an approximate value, therefore we assume the the bound state has S=0. Also kB is too small and reduces
        # NOTE: the Sfree to zero, hence we omit it along with the Temperature multiplier and thus we calculate the
        # NOTE: confS = -T*DeltaS = Sum(p(i) ( ln(p(i)) )
        # kB = 1.380649 * 10e-23 J/K = 3.2998303059 * 10e-27 kcal/K
        # kB = 3.2998303059 * 10e-27
        # But because we want kcal/(K*mol) in 298 Kelvin, we set
        KT = 0.593

        if "_Boltzmann" in descriptor:
            # AVENUE 1: calculate cluster probabilities by Boltzmann re-weighting. Sometimes is good for affinity ranking.
            m = re.search(".*_Boltzmann_KT([0-9]+\.[0-9]+)$", descriptor)
            if m:   # use the KT value specified in the descriptor's name
                KT = float(m.group(1))
            clusterprobs_dict = defaultdict(float)
            min_energy = np.min(list(confID_energy_dict.values()))
            for centroidID in clusterAssignments_dict.keys():
                for confID in clusterAssignments_dict[centroidID]:
                    DDG = confID_energy_dict[confID] - min_energy   # Delta Delta G (relative free Energy)
                    clusterprobs_dict[centroidID] += np.exp(-DDG/KT)  # Boltzmann factor
            # The sum of all cluster probabilities must be 1.0
            partition_function = np.sum(list(clusterprobs_dict.values()))   # namely, the sum of probabilities of all microstates
            for centroidID in clusterprobs_dict.keys():
                clusterprobs_dict[centroidID] /= partition_function
        else:
            # AVENUE 2: calculate cluster probabilities by simply counting their members
            N = sum(clustersizes_dict.values())     # number of valid conformers
            clusterprobs_dict = { k:v/N for k,v in clustersizes_dict.items() }

        # Finally, convert the cluster probabilities to Entropy
        Sfree = -np.sum( [p*np.log(p) for p in clusterprobs_dict.values()] )    # essentially this is Sfree/kB
        confS = -KT*(0-Sfree)    # -T*DeltaS , assuming that Sbound=0. Now we multiply by kB, thus the descriptor
                                # has units kcal/mol
        return confS

    elif "_nConf" in descriptor:
        # STEP 5: calculate the nConf[2-4]0 descriptor
        energies = list(centroidID_energy_dict.values())
        energy_descriptor = 0   # number of microstates with DeltaG < 20 kcal/mol, excluding the zero energy microstate
        relative_energies= np.array(energies)-energies[0]    # subtract the lowest energy to convert to relative energies
        # print("DEBUG: relative_energies=", relative_energies.tolist())
        # print("DEBUG: relative_energies=", np.exp(relative_energies).tolist())
        energy_cutoff = float(descriptor.split("_nConf")[-1])
        for energy in relative_energies[1:]:
            if 0 <= energy < energy_cutoff:
                energy_descriptor += 1
        return energy_descriptor

def load_ligentropy_file(fname, lowest_entropy_per_basename=False):

    molname_entropy_dict = {}
    with open(fname, 'r') as f:
        if fname.endswith(".csv"):
            csv_reader = csv.reader(f, delimiter=',')
            next(csv_reader)  # discard first line (headers)
            for vector in csv_reader:
                try:
                    molname = vector[0].lower()
                    entropy = float(vector[1])
                    if lowest_entropy_per_basename:
                        molname = get_basemolname(molname)  # work with basenames instead of full molnames
                        if molname in molname_entropy_dict.keys() and entropy < molname_entropy_dict[molname]:
                            molname_entropy_dict[molname] = entropy
                        elif molname not in molname_entropy_dict.keys():
                            molname_entropy_dict[molname] = entropy
                    else:
                        molname_entropy_dict[molname] = entropy
                except ValueError:
                    continue
        else:
            for line in f:
                try:
                    molname, entropy = line.split()
                    molname = molname.lower()
                    entropy = float(entropy)
                    if lowest_entropy_per_basename:
                        molname = get_basemolname(molname)
                        if molname in molname_entropy_dict.keys() and entropy < molname_entropy_dict[molname]:
                            molname_entropy_dict[molname] = entropy
                        elif molname not in molname_entropy_dict.keys():
                            molname_entropy_dict[molname] = entropy
                    else:
                        molname_entropy_dict[molname] = entropy
                except ValueError:
                    continue

    return molname_entropy_dict