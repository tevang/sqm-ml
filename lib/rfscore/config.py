import json
import logging
import os

MODULE_PATH = os.path.dirname(os.path.realpath(__file__))
CONFIG_PATH = os.path.join(MODULE_PATH, 'data', 'config.json')
RES_PATH    = os.path.join(MODULE_PATH, 'data', 'atomtypes.json')

# LOGGING
logger = logging.getLogger("pdbbind.py")
logger.setLevel(logging.INFO)

# CREATE CONSOLE HANDLER AND SET LEVEL TO DEBUG
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# CREATE FORMATTER
formatter = logging.Formatter("%(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# LOAD THE CONFIGURATION FROM EXTERNAL FILE
if os.path.exists(CONFIG_PATH):
    config = json.loads(open(CONFIG_PATH).read())
else:
    logger.error('The configuration file %s does not exist.' % CONFIG_PATH)
    sys.exit(1)

# GET THE PROTEIN ATOM TYPE DEFINITIONS
res_atom_types = json.loads(open(RES_PATH).read())


# INITIALIZE A MULTIDICT WITH ALL THE RESIDUES AND ATOM NAMES THAT CAN BE FOUND IN RECEPTOR FILES
resname_atomnames_dict = {}
for k1 in list(res_atom_types.keys()):
    if k1=="DESCRIPTION": continue
    resname_atomnames_dict[k1] = [str(i).replace(" ", "") for i in list(res_atom_types[k1].keys())]

# GET THE CONTACT TYPE DEFINITIONS
contact_types = config['contact types']