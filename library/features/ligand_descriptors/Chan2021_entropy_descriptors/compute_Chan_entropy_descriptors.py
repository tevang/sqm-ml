from library.features.ligand_descriptors.Chan2021_entropy_descriptors.Hbond_foldability import compute_Hbond_foldability
from library.features.ligand_descriptors.Chan2021_entropy_descriptors.functional_groups import compute_functional_group_count
from library.features.ligand_descriptors.Chan2021_entropy_descriptors.pipi_stacking_foldability import \
    compute_pipi_stacking_foldability
from library.features.ligand_descriptors.Chan2021_entropy_descriptors.ring_flexibility import compute_ring_flexibility
from library.features.ligand_descriptors.Chan2021_entropy_descriptors.rotor import compute_rotor_count
import pandas as pd

from library.features.ligand_descriptors.Chan2021_entropy_descriptors.terminal_CH3 import compute_terminal_CH3_count


def compute_Chan_entropy_descriptors(mol):
    """
    rotor count
    terminal CH3 count
    H-bond Foldability
    pi-pi stacking Foldability
    Ring Flexibility (according to the authors needs improvement)
    """
    return pd.DataFrame([[mol.GetProp('_Name'), compute_rotor_count(mol), compute_terminal_CH3_count(mol),
                          compute_functional_group_count(mol), compute_Hbond_foldability(mol),
                          compute_pipi_stacking_foldability(mol), compute_ring_flexibility(mol)]],
                        columns=['structvar', 'rotor_count', 'terminal_CH3_count',
                                 'function_group_count', 'ring_flexibility',
                                 'Hbond_foldability', 'pipi_stacking_foldability'])