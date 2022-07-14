from lib.global_fun import *
from lib.utils.print_functions import ColorPrint


def delete_chain(pdb, chains2remove=[], chains2keep=[], new_pdb=""):
    """
    Removes specific chains from the structure.
    """
    assert (chains2remove and not chains2keep) or (not chains2remove and chains2keep), \
        ColorPrint("ERROR: you must either provide a list for chains to remove or to keep, not both!", "FAIL")

    records = ('ATOM', 'HETATM', 'TER', 'ANISOU')
    new_lines = []
    if chains2remove:
        for line in open(pdb, 'r'):
            if line.startswith(records):
                if line[21] in chains2remove:
                    continue
            new_lines.append(line)
    elif chains2keep:
        for line in open(pdb, 'r'):
            if line.startswith(records):
                if not line[21] in chains2keep:
                    continue
            new_lines.append(line)

    if not new_pdb:
        new_pdb = pdb.replace(".pdb", "_delchains.pdb")

    writelist2file(new_lines, new_pdb)
