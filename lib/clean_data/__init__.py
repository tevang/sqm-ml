

class Clean_Data():

    def __init__(self):
        pass

    def _filter_replicate_compounds(self):
        """
        Method to find compounds existing multiple times within the same assayID. If their copies have the same affinity, then
        keep only one copy, otherwise remove all copies of that compound.

        :param self:
        :return:
        """

        args = fake_Args(args_dict=self.args_dict)  # to fool the StructureLoader and GroupDataLoader
        assert len(self.args_dict['-traincf']) == 1, \
            Debuginfo("FAIL: the current implementation of 'crossval_test_compound_num' option is just for one traincf "
                      "containing a single assay! %s", fail=True)

        datasets = GroupDataLoader(args=args)
        datasets.load_group_affinities()
        all_assayIDs = {t[1] for t in datasets.bind_molID_assayID_Kd_list + datasets.function_molID_assayID_Kd_list}
        self.molID_assayID_Kd_to_remove_list = []  # triplets to remove from bind_molID_assayID_Kd_list and function_molID_assayID_Kd_list
        for assayID in all_assayIDs:
            molname_Kds_dict = defaultdict(list)
            for t in datasets.bind_molID_assayID_Kd_list + datasets.function_molID_assayID_Kd_list:
                if t[1] == assayID:
                    molname_Kds_dict[t[0]].append(t[2])
            for molname, Kds in molname_Kds_dict.items():
                if len(Kds) == 1:
                    continue
                if len(Kds) > 1 and len(set(Kds)) > 1:
                    ColorPrint("WARNING: compound %s will be removed from assay %s as it was found multiple times "
                               "with different affinity." % (molname, assayID))
                    for Kd in Kds:
                        self.molID_assayID_Kd_to_remove_list.append((molname, assayID, Kds))
                elif len(Kds) > 1 and len(set(Kds)) == 1:
                    ColorPrint(
                        "WARNING: only copy of compound %s will be left in assay %s as it was found multiple times "
                        "with the same affinity." % (molname, assayID))
                    for Kd in Kds[1:]:
                        self.molID_assayID_Kd_to_remove_list.append((molname, assayID, Kds))
        # TODO: finish it! Example case: SmCB1 compound WRR-286.