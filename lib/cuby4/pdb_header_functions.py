def get_sqm_region_properties_from_header(pdb):
    with open(pdb) as f:
        for line in f:
            if line.startswith('HEADER SQM_REGION_CUTOFF ='):
                _, _, _, cutoff, _, _, _, charge, _, _, _, resid_selection, _, _, _, _ = line.split()
                yield float(cutoff), int(charge), resid_selection