#!/usr/bin/env python
from argparse import ArgumentParser, RawDescriptionHelpFormatter

from library.molfile.mol2_parser import *
from library.molfile.sdf_parser import *
from library.features.ligand_descriptors.Sconf_descriptors import calc_Sconf_descriptors


def cmdlineparse():
    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter,
                            description="""
    A script to extract molecular free Energies from a MOL2 file and to compute Sconf descriptors, namely:
    'structvar', 'DeltaG', 'DeltaG_0to1', 'DeltaG_1to2', 'DeltaG_2to3', 'DeltaG_3to4', 'DeltaG_4to5', 'DeltaG_5to6'
    """,
    epilog="""



    """)

    parser.add_argument("-i", dest="MOLFILE", required=True, type=str, default=None,
                        help="SDF or MOL2 files to be patched with extra properties.")
    parser.add_argument("-ocsv", dest="CSV_PROP_FILE", required=True, type=str, default=None,
                        help="CSV file where Sconf descriptors will be written.")

    args=parser.parse_args()
    return args

if __name__ == "__main__":

    args = cmdlineparse()

    extract_props_from_mol2(mol2=args.MOLFILE,
                            csv_prop_file=args.CSV_PROP_FILE + '_energies.csv',
                            propname='Energy:',
                            value_index=1)
    calc_Sconf_descriptors(args.CSV_PROP_FILE + '_energies.csv').to_csv(args.CSV_PROP_FILE, index=False)
    os.remove(args.CSV_PROP_FILE + '_energies.csv')