import numpy as np
from collections import defaultdict
from ete3 import Tree
import gc, sys
from itertools import permutations
from operator import itemgetter
from lib.global_fun import *

class Macrocycle():

    def __init__(self):
        self.connectivities_dict = defaultdict(set)
        self.all_chains_set = set()
    
    @staticmethod
    def is_macrocycle(mol, minringsize=10):
        """
        Method to return if a molecule is a macrocycle. The condition to meet is: mol should contain at least
        one ring with >= minringsize atoms.

        :param mol:
        :param minringsize:
        :return:
        """
        rings = mol.GetRingInfo().AtomRings()
        if len(rings) == 0:
            return False
        else:
            return np.max([len(ring) for ring in rings]) >= minringsize

    @staticmethod
    def is_atom_in_macrocyclic_ring(mol, atom, minringsize=10):
        atomIdx = atom.GetIdx()
        macro_rings = np.array([ring for ring in mol.GetRingInfo().AtomRings() if len(ring)>=minringsize])
        return atomIdx in macro_rings.flatten()
    
    def populate_leaves(self, Assignment_Tree):
        """
        Method that adds new branches to the leaves of the Tree.

        :param Assignment_Tree: the Tree structure with connectivities
        :param connectivities_dict:
        :return: (Assignment_Tree, BOOLEAN):    A tuple with elements the input Tree structure with new branches (if applicable), and a BOOLEAN value which is True if the function added
                                           new leaves to the Tree, or False otherwise
        """

        number_of_new_leaves = 0
        # ATTENTION: never use Assignment_Tree.iter_leaf_names(), it doesn't return the names in the order
        # ATTENTION: corresponding to Assignment_Tree.get_leaves()!!!
        for leaf in Assignment_Tree.get_leaves():
            try:
                ancestors_list = [ancestor.name for ancestor in leaf.get_ancestors()] + [leaf.name]
                for ring_ind in self.connectivities_dict[leaf.name]:
                    if ring_ind in ancestors_list:  # if the current ring index is already a node or leaf in the Tree, continue to the next
                        continue
                    new_child = leaf.add_child(name=ring_ind)  # add a new branch to the current ring index
                    number_of_new_leaves += 1
            except KeyError:
                continue

        # print(Assignment_Tree.get_ascii(show_internal=True, compact=False, attributes=["name"]))
        # print Assignment_Tree.get_ascii(show_internal=True, compact=False)
        if number_of_new_leaves > 0:
            return (Assignment_Tree, True)
        else:
            return (Assignment_Tree, False)

    def build_Chain_Tree(self, start):
        """

        :param start:
        :return:
        """
        # print("Building Tree starting from ring index", start, "...")
        expand_tree = True
        Assignment_Tree = Tree()
        Root = Assignment_Tree.get_tree_root()
        Root.add_feature("name", start)
        level = 1
        # sys.stdout.write("Expanding tree from level ")
        while expand_tree:
            # sys.stdout.write("l%i " % level)
            # sys.stdout.flush()
            Assignment_Tree, expand_tree = self.populate_leaves(Assignment_Tree)
            level += 1
        # Print the Tree
        # print Assignment_Tree.get_ascii(show_internal=True, compact=False)
        # print Assignment_Tree.get_ascii(show_internal=True, compact=False, attributes=["name", "dist", "occupancy", "numOfResonances"])

        # print("\nSaving chains from Tree...")

        for leaf in Assignment_Tree.get_leaves():
            chain = []
            chain.append(leaf.name)
            # print("DEBUG: leaf.name=", leaf.name, [a.name for a in leaf.get_ancestors()])
            for ancestor in leaf.get_ancestors():
                chain.append(ancestor.name)
            self.all_chains_set.add(tuple(chain))
            del chain
            del ancestor
            del leaf
            # Assignment_Tree = None
        del Assignment_Tree
        gc.collect()

    def is_complex_cyclic(self, mol, minringsize=12):
        """
        Method to return if a molecule is a complex cyclic (superset that include large molecules with many connected rings
        or macrocycles). The condition to meet is: mol should contain are least
        minringsize atoms belonging to one big ring (macrocycles) or connected rings.

        :param mol:
        :param minringsize:
        :return:
        """
        self.connectivities_dict = defaultdict(set) # reset the dict
        self.all_chains_set = set()                  # reset the dict
        rings = [set(r) for r in mol.GetRingInfo().AtomRings()]
        if len(rings) == 0: # if no rings
            return False
        elif Macrocycle.is_macrocycle(mol):  # macrocycles are also considered as complex cyclic
            return True
        else:   # otherwise check for chains of rings comprising at least minringsize atoms
            for i in range(len(rings)-1):
                for j in range(i+1, len(rings)):
                    ri = rings[i]
                    rj = rings[j]
                    if len(ri.intersection(rj)) >= 2:
                        self.connectivities_dict[i].add(j)
                        self.connectivities_dict[j].add(i)
            # create chains of rings
            for i in list(self.connectivities_dict.keys()):
                self.build_Chain_Tree(start=i)
            # Find the largest ring chain and return if it comprises >=minringsize unique heavy atoms
            if len(self.all_chains_set) == 0:
                return False
            max_chainsize = np.max([len(c) for c in self.all_chains_set])
            for chain in self.all_chains_set:
                chain_atoms_set = set(flatten([rings[r] for r in chain]))
                if len(chain_atoms_set) >= minringsize: # if at least one ring chain satisfies the condition, return True
                    return True
            return False


# # JUST FOR TESTING
# if __name__ == "__main__":
#     from rdkit import Chem
#     # non-macrocycle
#     mol = Chem.MolFromSmiles('CS(=O)(=O)N1CCc2c(C1)c(nn2CCCN1CCOCC1)c1ccc(Cl)c(C#Cc2ccc3C[C@H](NCc3c2)C(=O)N2CCCCC2)c1')
#     # macrocycle
#     mol = Chem.MolFromSmiles('CC(C)c5cc(CNC[C@@H](O)[C@@H]4C[C@H](C)CCCCCN([C@H](C)c1ccccc1)C(=O)c2cc(cc(c2)C3=NC=CO3)C(=O)N4)ccc5')
#     macro = Macrocycle()
#     macro.is_complex_cyclic(mol)
#     for c in macro.all_chains_set:
#         print("DEBUG: chain=", c)