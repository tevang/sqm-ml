from .smina import *
from ..utils.print_functions import ColorPrint


def _create_glide_input(gridfile, ligfile_list, sampling_type):
    # NOTE: 'MACROCYCLE True' will dock only macrocyclic molecules.
    Glide_inputs = {
        'enhanced': """
    GRIDFILE  %s
    LIGANDFILE  %s
    CV_CUTOFF   5.0
    EXPANDED_SAMPLING   True
    MACROCYCLE   False
    NREPORT   1000
    POSE_OUTTYPE   ligandlib_sd
    POSES_PER_LIG   100
    POSTDOCK_NPOSE   1000
    POSTDOCK_XP_DELE   0.5
    PRECISION   SP
    WRITE_XP_DESC   False
    NENHANCED_SAMPLING   2
    MAXREF 800
    MAXKEEP 10000
    SCORING_CUTOFF 200.0
    POSE_DISPLACEMENT   1.5
    POSE_RMSD   1.0
    
    """ % (gridfile, ",".join(ligfile_list))
        ,
        'default': """
    GRIDFILE  %s
    LIGANDFILE  %s
    POSE_OUTTYPE   ligandlib_sd
    POSES_PER_LIG   1
    PRECISION   SP
    POSE_DISPLACEMENT   1.5
    POSE_RMSD   1.0
        """ % (gridfile, ",".join(ligfile_list))
    }

    return Glide_inputs[sampling_type]

class Glide(Smina):

    def __init__(self,
                 pose_sdf=None,
                 pose_sdf_pattern=None,
                 score_file=None,
                 pose_score_file=None,
                 max_poses=None,
                 score_property_name='r_i_docking_score'):
        super(Glide, self).__init__(pose_sdf=pose_sdf, pose_sdf_pattern=pose_sdf_pattern, score_file=score_file)
        self.score_property_name = score_property_name
        self.software = "GLIDE"
        self.pose_score_file = pose_score_file
        self.max_poses = max_poses

        self.structvar_validPoses_dict = defaultdict(list)      # structvar -> list of valid poseIDs
        if self.max_poses and self.pose_score_file:             # populate structvar_validPoses_dict
            self.find_valid_poses()

    @staticmethod
    def write_glide_input(gridfile,
                          ligfile_list,
                          sampling_type='default'):
        with open("glide-dock_SP_%s_sampling.in" % sampling_type, 'w') as f:
            f.writelines(_create_glide_input(gridfile, ligfile_list, sampling_type))

    @staticmethod
    def create_gridfile(receptor, refligand):
        # TODO: complete this function for receptor grid generation.
        pass

    @staticmethod
    def write_docking_script(script_filename="docking.sh", sampling_type='default', cpus=8):
        launch_script = """
    #!/bin/bash
    
    "${SCHRODINGER}/glide" glide-dock_SP_%s_sampling.in -OVERWRITE -adjust -HOST "localhost:%i" -NJOBS %i -TMPLAUNCHDIR -ATTACHED
    
    """ % (sampling_type, cpus, cpus)

        # Glide.create_gridfile()   # TEMPORARILY INACTIVATED
        write2file(launch_script, script_filename)
        run_commandline("chmod 777 %s" % script_filename)
        ColorPrint("Created Glide input files. To launch the docking execute the script: %s" % script_filename, "BOLDGREEN")
