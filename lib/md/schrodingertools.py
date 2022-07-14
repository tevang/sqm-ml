import shutil

from lib.global_fun import *
from lib.utils.print_functions import ColorPrint


class SchrodingerTools():

    def __init__(self):
        pass

    @staticmethod
    def split_complex_pdb(complex_pdb_fname):
        """
        Method to split a protein-ligand complex PDB file into a PDB file of the receptor and a MOL2 file for the ligand,
        using SCHRODINGER's tools.
        :param complex_pdb_fname:
        :return: the function will write
        """

        # ATTENTION: **ALWAYS** use SCHRODINGER tools to ensure that the MOL2 file has the right atom types.
        assert "SCHRODINGER" in os.environ, ColorPrint("FAIL: Environment variable 'SCHRODINGER' is not set!",
                                                       "FAIL")
        assert os.path.exists(os.environ.get('SCHRODINGER') + "/utilities/structconvert"), \
            ColorPrint("FAIL: could not find SCHRODINGER tools in directory %s !" % os.environ.get('SCHRODINGER'),
                       "FAIL")

        # Split protein-ligand PDB file into mae files
        if os.path.exists("tmp/"):    shutil.rmtree("tmp/")
        os.mkdir("tmp/")
        shutil.copy2(complex_pdb_fname, "tmp/%s" % (os.path.basename(complex_pdb_fname)))
        os.chdir("tmp/")

        basename = os.path.splitext(os.path.basename(complex_pdb_fname))[0]
        run_commandline("%s/run %s/mmshare-v4.5/python/common/split_structure.py \
        -mode pdb %s.pdb %s.mae -many_files" %
                        (os.environ.get('SCHRODINGER'), os.environ.get('SCHRODINGER'),
                         basename, basename))

        # Assign the right protonation to the ligand
        run_commandline("%s/utilities/applyhtreat %s_ligand1.mae "
                        "%s_ligand1.mol2" %
                        (os.environ.get('SCHRODINGER'), basename, basename))
        run_commandline("rm %s_ligand1.mae" % basename)

        # Convert the receptor to PDB format for MD
        run_commandline("%s/utilities/structconvert %s_receptor1.mae "
                        "%s_receptor1.pdb" % (os.environ.get('SCHRODINGER'), basename, basename))
        run_commandline("rm %s_receptor1.mae" % basename)

        # Change the molname and resname to 'LIG'
        run_commandline("antechamber -i %s_ligand1.mol2 -fi mol2 -o "
                        "%s_ligand1.mol2 -fo mol2 -rn LIG -at sybyl" %
                        (basename, basename))

        os.rename("%s_ligand1.mol2" % basename, "../%s_ligand1.mol2" % basename)
        os.rename("%s_receptor1.pdb" % basename, "../%s_receptor1.pdb" % basename)
        os.chdir("../")
        shutil.rmtree("tmp/")

        return os.path.realpath("%s_receptor1.pdb" % basename), \
               os.path.realpath("%s_ligand1.mol2" % basename)