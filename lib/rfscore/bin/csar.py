import csv
import glob
import os
import sys
from optparse import OptionParser

from rfscore.config import config, logger
from rfscore.credo import contacts
from rfscore.ob import get_molecule, extract_ligand


def parse_options():
    '''
    '''
    # PARSE COMMAND LINE
    usage  = "%prog [options]"
    parser = OptionParser(usage=usage)

    parser.add_option("--debug",
                      action  = "store_true",
                      dest    = "debug",
                      default = False,
                      help    = 'Set logging level to debug and print(more verbose output.')

    parser.add_option("-B", "--binsize",
                      dest    = "binsize",
                      type    = float,
                      default = 0.0,
                      help    = "Bin size (in Angstrom) to use for binning contacts based on inter-atomic distance.")

    parser.add_option("-F", "--format",
                      dest    = "format",
                      default = 'csv',
                      help    = "Format to use for writing the SIFt of the protein-ligand complex.")

    parser.add_option("-O", "--output",
                      dest    = "output",
                      default = None,
                      help    = "File to which the data will be written (default=STDOUT).")

    parser.add_option("-D", "--descriptor",
                      dest    = "descriptor",
                      default = 'elements',
                      help    = "Descriptor to use. Valid descriptors are 'credo', 'elements' and 'sybyl'.")

    # GET COMMAND LINE OPTIONS
    (options, args) = parser.parse_args()

    if options.descriptor not in ('elements', 'credo', 'sybyl'):
        logger.fatal("Invalid descriptor: {0}.".format(options.descriptor))
        parser.print_help()
        sys.exit(1)

    return options


def main():
    """
    """
    options = parse_options()

    # this option will produce more verbose output
    if options.debug: logger.setLevel(logging.DEBUG)

    csarconf = config['csar']

    if options.output: fh = open(options.output,'wb')
    else: fh = sys.stdout

    # choose how the ouptput data will be written
    if options.format == 'csv':
        writer = csv.writer(fh, delimiter=',', quotechar='"',
                            quoting=csv.QUOTE_MINIMAL)

    HEADER = True

    # iterate through all numbered directories
    for directory in os.listdir(csarconf['directory']):
        entrydir = os.path.join(csarconf['directory'], directory)
        
        # parse kd.dat to get the pKd
        kddat_path = os.path.join(entrydir, 'kd.dat')
        
        # exit if kd.dat is missing
        if not os.path.isfile(kddat_path):
            logger.fatal("CSAR directory {} does not contain kd.dat file."
                         .format(directory))
            sys.exit(1)
        
        entry, pdb, pkd = open(kddat_path).read().strip().replace(' ','').split(',')

        protein_path = glob.glob(os.path.join(entrydir, '*_complex.mol2')).pop()
    
        protein = get_molecule(str(protein_path))
        ligand = extract_ligand(protein.OBMol)
   
        # calculate descriptor based on the sum of interacting element pairs
        if options.descriptor == 'elements':

            # calculate element pair descriptor for this complex
            descriptor, labels = contacts.element_descriptor(protein, ligand,
                                                             binsize=options.binsize)
            
        # calculate descriptor based on the sum of interacting element pairs
        elif options.descriptor == 'sybyl':

            # calculate element pair descriptor for this complex
            descriptor, labels = contacts.sybyl_atom_type_descriptor(protein, ligand,
                                                                     binsize=options.binsize)
   
        # calculate descriptor using structural interaction fingerprints
        elif options.descriptor == 'credo':

            # get the protein-ligand structural interaction fingerprint
            descriptor, labels = contacts.sift_descriptor(protein, ligand,
                                                          binsize=options.binsize)

        if HEADER:

            # UPDATE COLUMN LABELS
            labels.insert(0,'pKd/pKi')
            labels.append('pdb')

            writer.writerow(labels)

            HEADER = False

        if options.format == 'csv':

            # FIRST COLUMN OF OUTPUT ROW
            row = [pkd] + descriptor.tolist() + [pdb]

            writer.writerow(row)
main()
