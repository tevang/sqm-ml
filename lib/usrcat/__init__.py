"""
These are the SMARTS patterns that are used to define the pharmacophore subsets.
Make sure to update the similarity function accordingly if you make any changes
here: The order of patterns corresponds to the order of weights in the similarity
function.
"""

# SMARTS definition of pharmacophore subsets
PHARMACOPHORES = [
    "[#6+0!$(*~[#7,#8,F]),SH0+0v2,s+0,S^3,Cl+0,Br+0,I+0]", # hydrophobic
    "[a]", # aromatic
    "[$([O,S;H1;v2]-[!$(*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N&v3;H1,H2]-[!$(*=[O,N,P,S])]),$([N;v3;H0]),$([n,o,s;+0]),F]", # acceptor
    "[N!H0v3,N!H0+v4,OH+0,SH+0,nH+0]" # donor
    ]

# total number of moments // +1 to include the all atom set
NUM_MOMENTS = (len(PHARMACOPHORES) + 1) * 12