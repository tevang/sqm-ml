
import requests

# # Import ConsScorTK libraries
# sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath( os.path.abspath(__file__) ) ) ) )  # import the top level package directory

# OLD FUNCTION
# def fetch_data(url, query_string):
#     """
#     Method to retry infinite times to fetch data from the server.
#     :param url:
#     :param query_string:
#     :return:
#     """
#     try:
#         print("DEBUG: url=%s" % url)
#         print("DEBUG: query_string=%s" % query_string)
#         req = urllib.request.Request(url, data=query_string)
#         f = urllib.request.urlopen(req)
#         result = f.readlines()
#         return result
#     except:
#         ColorPrint("Failed to fetch data, trying again...", "OKRED")
#         return fetch_data(url, query_string)
from lib.utils.print_functions import ColorPrint


def fetch_data(url, query_string):
    """
    Method to retry infinite times to fetch data from the server.
    :param url:
    :param query_string:
    :return:
    """
    try:
        # print("DEBUG: url=%s" % url)
        # print("DEBUG: query_string=%s" % query_string)
        response = requests.post(
            url,
            data=query_string,
            headers={'Content-Type': 'application/x-www-form-urlencoded'}
        )
        return [r for r in response.text.split('\n') if len(r)>0]
    except:
        ColorPrint("Failed to fetch data, trying again...", "OKRED")
        return fetch_data(url, query_string)


def get_pdbIDs_from_ligID(lig_ID):
    url = 'http://www.rcsb.org/pdb/rest/search'
    # ATTENTION: no indentation in XML queries!!!
    ProtLig_query = """
<?xml version="1.0" encoding="UTF-8"?>
<orgPdbQuery>
<queryType>org.pdb.query.simple.ChemCompIdQuery</queryType>
<chemCompId>""" + lig_ID + """</chemCompId>
<polymericType>Any</polymericType>
</orgPdbQuery>
    """

    result4 = fetch_data(url, ProtLig_query)
    return result4


def get_PDBs_with_similar_ligands(SMILES, lig_sim):
    url = 'http://www.rcsb.org/pdb/rest/search'
    # ATTENTION: no indentation in XML queries!!!
    LigSim_query = """
<?xml version="1.0" encoding="UTF-8"?>
<orgPdbQuery>
<queryType>org.pdb.query.simple.ChemSmilesQuery</queryType>
<smiles>""" + SMILES + """</smiles>
<searchType>Similar</searchType>
<similarity>""" + str(lig_sim) + """</similarity>
<polymericType>Any</polymericType>
<target>Ligand</target>
</orgPdbQuery>
    """

    print("Fetching data from PDB (speed depends on your internet connection) ...")
    print(SMILES)
    result3 = fetch_data(url, LigSim_query)
    # print("DEBUG: result3=", result3)
    if len(result3) == 0:
        return []

    pdbIDs_with_ligand = []
    for lig_ID in result3:
        lig_ID = lig_ID.rstrip()
        result4 = get_pdbIDs_from_ligID(lig_ID)
        pdbIDs_with_ligand.extend(result4)
    pdbIDs_with_ligand = set([p.rstrip() for p in pdbIDs_with_ligand])

    return pdbIDs_with_ligand


def get_homologues(prot_seq, seq_identity, result2_set, highres_set=set([])):
    url = 'http://www.rcsb.org/pdb/rest/search'
    # ATTENTION: no indentation in XML queries!!!
    ProtSeq_query = """
<?xml version="1.0" encoding="UTF-8"?>
<orgPdbQuery>
<queryType>org.pdb.query.simple.SequenceQuery</queryType>
<structureId></structureId>
<chainId>A</chainId>
<sequence>""" + prot_seq + """</sequence>
<eCutOff>10.0</eCutOff>
<searchTool>blast</searchTool>
<sequenceIdentityCutoff>""" + str(seq_identity) + """</sequenceIdentityCutoff>
</orgPdbQuery>
    """

    result1 = fetch_data(url, ProtSeq_query)
    result1_set = set([r.split(':')[0] for r in result1])
    # Get all high resolution structures

    # Keep only the PDBs with ligands
    if len(highres_set) == 0:
        highres_set = result2_set
    homologues_with_ligand = result2_set.intersection(result1_set).intersection(highres_set)
    return homologues_with_ligand


def get_structures_with_resolution(min_resolution=0.0, max_resolution=2.0):
    url = 'http://www.rcsb.org/pdb/rest/search'
    # ATTENTION: no indentation in XML queries!!!
    ProtSeq_query = """
<?xml version="1.0" encoding="UTF-8"?>
<orgPdbQuery>
<queryType>org.pdb.query.simple.ResolutionQuery</queryType>
<refine.ls_d_res_high.comparator>between</refine.ls_d_res_high.comparator>
<refine.ls_d_res_high.min>%f</refine.ls_d_res_high.min>
<refine.ls_d_res_high.max>%f</refine.ls_d_res_high.max>
</orgPdbQuery>
    """ % (min_resolution, max_resolution)

    result = fetch_data(url, ProtSeq_query)
    result_set = set([r.split(':')[0].rstrip() for r in result])
    return result_set


def get_all_pdb_with_ligands():
    # Get all PDBs complexes with ligands
    url = 'http://www.rcsb.org/pdb/rest/search'
    # ATTENTION: no indentation in XML queries!!!
    HasLig_query = """
<?xml version="1.0" encoding="UTF-8"?>
<orgPdbQuery>
<queryType>org.pdb.query.simple.NoLigandQuery</queryType>
<haveLigands>yes</haveLigands>
</orgPdbQuery>
            """

    result2 = fetch_data(url, HasLig_query)
    result2_set = set([r.rstrip() for r in result2])
    return result2_set