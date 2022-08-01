#!/usr/bin/env python

import copy
import math
from collections import defaultdict

from rdkit import Chem
from rdkit.Chem import rdMolAlign, AllChem


class Ligand_RMSD():
    """
    Class to compute the RMSD between two ligands.
    """

    # def __init__(self, refFile, prbFile, dontMove=True, symm=True):
    def __init__(self, prbMol, refMol, dontMove=True, symm=True):
        """

        :param refFile:
        :param prbFile:
        :param dontMove: triggers the computation "in place", i.e., without  realigning poses.
        :param symm: triggers consideration of symmetric atoms when doing the RMSD computation (as in RDKit's GetBestRMS()).
        """
        # refMol = Chem.rdmolfiles.MolFromMolFile(refFile, sanitize=True, removeHs=True)
        # prbMol = Chem.rdmolfiles.MolFromMolFile(prbFile, sanitize=True, removeHs=True)
        self.refMol = AllChem.RemoveHs(refMol)
        self.prbMol = AllChem.RemoveHs(prbMol)
        self.dontMove = dontMove
        self.symm = symm

    def FindSymmetryClasses(self, m):
        """
        Finds symmetric (equivalent) atoms WITHIN THE SAME MOLECULE.
        :param m:
        :return:
        """
        equivs = defaultdict(set)
        matches = m.GetSubstructMatches(m, uniquify=False)
        for match in matches:
            for idx1, idx2 in enumerate(match): equivs[idx1].add(idx2)
        classes = set()
        for s in list(equivs.values()): classes.add(tuple(s))
        return classes

    def symmUniqueFit(self, prbMol, refMol, prbConfId=-1, refConfId=-1):
        matches = prbMol.GetSubstructMatches(refMol, uniquify=False)
        if not matches:
            raise ValueError('mols don\'t match')
        amaps = [[tuple(reversed(p)) for p in list(enumerate(match))] for match in matches]
        first = True
        rmsdMin = 0.0
        amapBest = []
        prbConf = prbMol.GetConformer()
        refConf = refMol.GetConformer()
        for amap in amaps:
            conf = self.saveConformer(prbMol, prbConfId)
            rmsd = Chem.rdMolAlign.AlignMol(prbMol, refMol,
                                            prbConfId, refConfId, atomMap=amap)
            # print("DEBUG: rmsd after alignment=%f" % rmsd)
            # print("DEBUG: amap=", amap)
            prbMol = self.loadConformer(prbMol, prbConfId, conf)
            if (first or (rmsd < rmsdMin)):
                first = False
                rmsdMin = rmsd
                amapBest = amap
        return amapBest

    def get_rmsd(self):
        prbMolCopy = copy.copy(self.prbMol)
        amap = None
        if self.symm:
            amap = self.symmUniqueFit(self.prbMol, self.refMol)
            if (len(amap) != self.refMol.GetNumAtoms()):
                raise ValueError("FAIL: atomMap does not contain all atoms!")
        if self.dontMove:
            rmsd = self.getRmsdImmobile(prbMolCopy, self.refMol, atomMap=amap)
        else:
            (rmsd, trans) = Chem.rdMolAlign.GetAlignmentTransform \
                (prbMolCopy, self.refMol, atomMap=amap)
        return rmsd

    def saveConformer(self, mol, confId):
        conf = mol.GetConformer(confId)
        confCopy = Chem.Conformer(conf.GetNumAtoms())
        for i in range(conf.GetNumAtoms()):
            confCopy.SetAtomPosition(i, conf.GetAtomPosition(i))
        return confCopy

    def loadConformer(self, mol, confId, confCopy):
        conf = mol.GetConformer(confId)
        for i in range(conf.GetNumAtoms()):
            conf.SetAtomPosition(i, confCopy.GetAtomPosition(i))
        return mol

    def getRmsdImmobile(self, prbMol, refMol, prbConfId=-1, refConfId=-1, atomMap=None):
        refConf = refMol.GetConformer(refConfId)
        prbConf = prbMol.GetConformer(prbConfId)
        if (not atomMap):
            atomMap = []
            for i in range(0, refMol.GetNumAtoms()):
                if (refMol.GetAtomWithIdx(i).GetAtomicNum() == 1):
                    continue
                atomMap.append((i, i))
        sqDist = 0.0
        for pair in atomMap:
            sqDist += (prbConf.GetAtomPosition(pair[0]) \
                       - refConf.GetAtomPosition(pair[1])).LengthSq()
        sqDist /= float(len(atomMap))
        return math.sqrt(sqDist)

#
# if __name__ == '__main__':
#     from rdkit import Chem
#     qMOL = Chem.SDMolSupplier("/home2/thomas/Documents/QM_Scoring/Docking/Smina_Rigid/hs90a/actives_complexes_forscoring_100ps_noWAT/301178_CHEMBL178130_pose9_noWAT_aln0_LIG.sdf_LIG_215.sdf",
#                               removeHs=True, sanitize=True).next()
#     # qMOL = AllChem.RemoveHs(qMOL)
#     # qMOL = AllChem.AddHs(qMOL)
#     rMOL = Chem.SDMolSupplier("/home2/thomas/Documents/QM_Scoring/Docking/Smina_Rigid/hs90a/PDB_DIR/1yc1_LIG.sdf",
#                               removeHs=True, sanitize=True).next()
#     # rMOL = AllChem.RemoveHs(rMOL)
#     # rMOL = AllChem.AddHs(rMOL)
#
#     print("RMSD = %f" % Ligand_RMSD(qMOL, rMOL).get_rmsd()))