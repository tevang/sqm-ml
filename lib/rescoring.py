import sys, os
import sys

# Import ConsScorTK libraries
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )  # import the top level package directory
from .ConsScoreTK_Statistics import *



def prepare_files(ligand_mol2, receptor_pdb, MGLTOOLS_HOME="/home/thomas/Programs/MGLTools-1.5.6"):
    
    os.chdir(os.path.dirname(os.path.abspath(receptor_pdb)))
    run_commandline(MGLTOOLS_HOME+"/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_ligand4.py -l "+ligand_mol2,
                     logname="log", append=False, return_out=False, error_keywords=[])
    run_commandline(MGLTOOLS_HOME+"/MGLToolsPckgs/AutoDockTools/Utilities24/prepare_receptor4.py -r "+receptor_pdb,
                     logname="log", append=False, return_out=False, error_keywords=[])


def run_NNscore1(receptor_pdbqt, ligand_pdbqt, logfilename="", NNSCORE_HOME="/home/thomas/Programs/autodock_vina_1_1_2_linux_x86"):
    ##
    ## Function to execute NNscore 1
    ##
    
    os.chdir(os.path.dirname(os.path.abspath(receptor_pdbqt)))
    if not logfilename:
        logfilename = ligand_pdbqt.replace(".pdbqt", "")+"_NNscore1.log"
        logfilename = logfilename.replace("\\'", "'")
    for scoring_function in range(1,25):
        commandline = "python "+NNSCORE_HOME+"/NNScore.py -receptor "+receptor_pdbqt+" -ligand "+ligand_pdbqt+" -network "+NNSCORE_HOME+"/networks/top_24_networks/"+str(scoring_function)+".net"
        run_commandline(commandline, logname=logfilename, append=True, return_out=False, error_keywords=[])
        
def read_NNscore1_scores(ENSEMBLE_DOCKING_HOME):
    
    NNscore1_list = []
    with open(ENSEMBLE_DOCKING_HOME+"/ligand.MD_NNscore1.log", 'r+') as f:
        for line in f:
            mo = re.search('.*\/networks\/top_24_networks\/([0-9]+)\.net to predict binding:\s+([-0-9]+\.[0-9]+[-e0-9]*)\s+\([a-z]+ binder\)\n', line) # read network score
            if mo:
                NNscore1_list.append(float(mo.group(2)))   ## asign score to each ligand pose
    return NNscore1_list


def read_NNscore2_scores(ENSEMBLE_DOCKING_HOME):
    
    NNscore2_list = []
    with open(ENSEMBLE_DOCKING_HOME+"/ligand.MD_NNscore2.log", 'r+') as f:
        for line in f:
            mo = re.search('Best pose scored by network \#([0-9]+): MODEL [0-9]+ \(Score = ([0-9.e-]+) =', line) # read network score
            if mo:
                # print("isomer",isomer, "poseNo",poseNo,"net",mo.group(1),"-1*score",float(-1*float(mo.group(2))))
                NNscore2_list.append(float(mo.group(2)))   ## asign score to each ligand pose
    return NNscore2_list


def run_NNscore2(receptor_pdbqt, ligand_pdbqt, logfilename="", NNSCORE_HOME="/home/thomas/Programs/autodock_vina_1_1_2_linux_x86",
                 VINA_HOME="/home/thomas/Programs/autodock_vina_1_1_2_linux_x86"):
##
## Function to execute NNscore 2
##
#print("DEBUG: ligand_pdbqt = "+ligand_pdbqt
    os.chdir(os.path.dirname(os.path.abspath(receptor_pdbqt)))
    if not logfilename:
        logfilename = ligand_pdbqt.replace(".pdbqt", "")+"_NNscore2.log"
        logfilename = logfilename.replace("\\'", "'")
    commandline = "python "+NNSCORE_HOME+"/NNScore2.01.py -receptor "+receptor_pdbqt+" -ligand "+ligand_pdbqt+" -vina_executable "+VINA_HOME+"/bin/vina"
    run_commandline(commandline, logname=logfilename, append=True, return_out=False, error_keywords=[])


def run_DSX(receptor_pdbqt, ligand_pdbqt):
##
## Function to execute DSX
##
#print("DEBUG: ligand_pdbqt = "+ligand_pdbqt
    receptor_pdb=receptor_pdbqt.replace('.pdbqt', '.pdb')
    ligand_mol2=ligand_pdbqt.replace('.pdbqt', '.mol2')
    #fout=open(ENSEMBLE_DOCKING_HOME+"/rescoring/"+os.path.basename(ligand_pdbqt.replace(".pdbqt", ""))+"_DSX.log", 'w')
    #print("DEBUG: "+self.DSX_EXE+" -P "+receptor_pdb+" -L "+ligand_mol2+" -D "+self.DSX_POT_DIR+" -s"
    commandline = DSX_EXE+" -P "+receptor_pdb+" -L "+ligand_mol2+" -D "+DSX_POT_DIR+" -s"
    return_code = subprocess.call(commandline, shell=True)
    
    if (return_code != 0):
        print("WARNING, THE FOLLOWING COMMAND FAILED TO RUN:")
        print(DSX_EXE+" -P "+receptor_pdb+" -L "+ligand_mol2+" -D "+DSX_POT_DIR)
    #fout.close()