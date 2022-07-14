import os


def get_interface_surfaces_from_pdb(pdb_file, LIG_RESNAME="LIG"):
    import pymol as pml

    pdb_name = os.path.basename(pdb_file).replace(".pdb", "")
    pml.cmd.load(pdb_file)

    pml.cmd.create("complex", "polymer or resn LIG")    # isolate the protein and ligand from the whole structure
    pml.cmd.set("dot_solvent", 0)
    complex_whole_surf = pml.cmd.get_area("complex")
    pml.cmd.set("dot_solvent", 1)
    complex_whole_SASA = pml.cmd.get_area("complex")
    pml.cmd.delete("complex")

    pml.cmd.create("prot", "polymer")
    pml.cmd.create("ligand", "resn %s" % LIG_RESNAME)
    prot_atomnum = pml.cmd.count_atoms("prot")
    lig_atomnum = pml.cmd.count_atoms("ligand")
    if prot_atomnum == 0 or lig_atomnum == 0:   # One of the selections was empty!
        return [None]*6
    pml.cmd.delete(pdb_name)
    pml.cmd.select("prot_interface", "prot within 3.5 of ligand")
    pml.cmd.set("dot_solvent", 0)
    prot_interface_surf = pml.cmd.get_area("prot_interface")
    prot_whole_surf = pml.cmd.get_area("prot")
    pml.cmd.set("dot_solvent", 1)
    prot_interface_SASA = pml.cmd.get_area("prot_interface")
    prot_whole_SASA = pml.cmd.get_area("prot")

    pml.cmd.select("lig_interface", "ligand within 3.5 of prot")
    pml.cmd.set("dot_solvent", 0)
    lig_interface_surf = pml.cmd.get_area("lig_interface")
    lig_whole_surf = pml.cmd.get_area("ligand")
    pml.cmd.set("dot_solvent", 1)
    lig_interface_SASA = pml.cmd.get_area("lig_interface")
    lig_whole_SASA = pml.cmd.get_area("ligand")
    pml.cmd.delete("all")   # IMPORTANT: otherwise the molecules will remain in the next round too!

    mean_interface_surf = (prot_whole_surf+lig_whole_surf-complex_whole_surf)/2
    mean_interface_SASA = (prot_whole_SASA+lig_whole_SASA-complex_whole_SASA)/2

    return [prot_interface_surf, prot_interface_SASA, lig_interface_surf, lig_interface_SASA,
            mean_interface_surf, mean_interface_SASA]