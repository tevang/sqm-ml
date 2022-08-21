import os
import sys
import numpy as np
sys.path.append('/Users/iwatobipen/develop/chemoenv/molsg')

#calculate descriptor
from scipy.spatial import distance

from molsg.laplacemesh import compute_lb_fem
from molsg.localgeometry import WKS_descriptor
import molsg.covariance as cov
evals = 32

mol_names = sorted(os.listdir('data/ampc/npy/'))
mols = (np.load('data/ampc/npy/{}'.format(m)) for m in mol_names)
eigs = (compute_lb_fem(vertices=m[0], faces=m[1], k=100) for m in mols)
wks = (WKS_descriptor(e, evals=evals) for e in eigs)
covs = np.asarray([cov.covariance_descriptor(w).flatten() for w in wks])


