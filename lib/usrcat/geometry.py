from numpy import array, concatenate, sqrt
from scipy.special import cbrt
from scipy.stats.stats import skew, kurtosis

def distance_to_point(coords, point):
    """
    Returns an array containing the distances of each coordinate in the input
    coordinates to the input point.
    """
    return sqrt(((coords-point)**2).sum(axis=1))


# def plane_equation(x1, y1, z1, x2, y2, z2, x3, y3, z3):
#     """
#         A plane equation is defined as: a*x + b*y + c*z = d. 
#     """
#     # Determine two vectors from the 3 points defining the plane
#     vector1 = [x2 - x1, y2 - y1, z2 - z1]
#     vector2 = [x3 - x1, y3 - y1, z3 - z1]
#     
#     # Determine the cross product of the two vectors
#     cross_product = [vector1[1] * vector2[2] - vector1[2] * vector2[1], -1 * (vector1[0] * vector2[2] - vector1[2] * vector2[0]), vector1[0] * vector2[1] - vector1[1] * vector2[0]]
#     
#     a = cross_product[0]
#     b = cross_product[1]
#     c = cross_product[2]
#     d = - (cross_product[0] * x1 + cross_product[1] * y1 + cross_product[2] * z1)
#     
#     return a,b,c,d
#     
# 
# def signed_dist_to_plane(coords, point1, point2, point3):
#     """
#         Calculates the signed distances between all N points with coordinates given by coords array(3xN) and the plane defined by the
#         points (x1,y1,z1), (x2,y2,z2), (x3,y3,z3).
#         The distance is positive if P0 is on the same side of the plane as the normal vector v and negative if it is on the opposite side. 
#     """
#     # x0, y0, z0 = point0.tolist()
#     x1, y1, z1 = point1.tolist()
#     x2, y2, z2 = point2.tolist()
#     x3, y3, z3 = point3.tolist()
#     
#     a, b, c, d = plane_equation(x1, y1, z1, x2, y2, z2, x3, y3, z3)
#     
#     # return (a*x0 + b*y0 + c*z0) / sqrt(a**2 + b**2 + c**2)
#     return (coords * array([a,b,c])).sum(axis=1) / sqrt(a**2 + b**2 + c**2)  
    

def usr_moments(coords, masses, moment_number=4, core_ctd=None, core_com=None):
    """
    Calculates the USR moments for a set of input coordinates as well as the four
    USR reference atoms.

    :param coords: numpy.ndarray
    """
    
    dist_list = []  # list with the distance distributions of every point from all the atoms of the molecule
    # print("DEBUG: coords=", coords.tolist())
    # print("DEBUG: masses=", masses.tolist())
    # centroid of the input coordinates
    ctd = coords.mean(axis=0)
    
    # get the distances to the centroid
    dist_ctd = distance_to_point(coords, ctd)
    dist_list.append(dist_ctd)

    # get the closest and furthest coordinate to/from the centroid
    cst, fct = coords[dist_ctd.argmin()], coords[dist_ctd.argmax()]

    # get the distance distributions for the points that are closest/furthest
    # to/from the centroid
    dist_cst = distance_to_point(coords, cst)
    dist_fct = distance_to_point(coords, fct)
    # save the new points
    dist_list.append(dist_cst)
    dist_list.append(dist_fct)

    # get the point that is the furthest from the point that is furthest from
    # the centroid
    ftf = coords[dist_fct.argmax()]
    dist_ftf = distance_to_point(coords, ftf)
    dist_list.append(dist_ftf)
    
    ##
    ## NEW POINTS
    ##
    point_list = []
    
    """
    # get the points that are furtherst and closest to the plane defined by cst, fct, ftf in poth directions
    plane_distances = signed_dist_to_plane(coords, cst, fct, ftf)
    pos_plane_distances = plane_distances[plane_distances>0]
    neg_plane_distances = plane_distances[plane_distances<0]
    plane_centroid = mean([cst, fct, ftf], axis=0)    # centroid of cst, fct, ftf (lies on the same plane)
    if len(pos_plane_distances) > 0:
        ftpp1 = coords[pos_plane_distances.argmax()]
        ctpp1 = coords[pos_plane_distances.argmin()]
    else:
        ftpp1 = plane_centroid
        ctpp1 = plane_centroid
    if len(neg_plane_distances) > 0:
        ftnp1 = coords[neg_plane_distances.argmin()]
        ctnp1 = coords[neg_plane_distances.argmax()]
    else:
        ftnp1 = plane_centroid
        ctnp1 = plane_centroid
    
    # save the new points
    point_list.append(ftpp1)
    # point_list.append(ctpp1)
    point_list.append(ftnp1)
    # point_list.append(ctnp1)
    
    # get the points that are furtherst and closest to the plane defined by ctd, fct, ftf in poth directions
    plane_distances = signed_dist_to_plane(coords, ctd, fct, ftf)
    pos_plane_distances = plane_distances[plane_distances>0]
    neg_plane_distances = plane_distances[plane_distances<0]
    plane_centroid = mean([ctd, fct, ftf], axis=0)    # centroid of ctd, fct, ftf (lies on the same plane)
    if len(pos_plane_distances) > 0:
        ftpp2 = coords[pos_plane_distances.argmax()]
        ctpp2 = coords[pos_plane_distances.argmin()]
    else:
        ftpp2 = plane_centroid
        ctpp2 = plane_centroid
    if len(neg_plane_distances) > 0:
        ftnp2 = coords[neg_plane_distances.argmin()]
        ctnp2 = coords[neg_plane_distances.argmax()]
    else:
        ftnp2 = plane_centroid
        ctnp2 = plane_centroid
    
    # save the new points
    point_list.append(ftpp2)
    # point_list.append(ctpp2)
    point_list.append(ftnp2)
    # point_list.append(ctnp2)
    
    ##~~~~~~~~~~~~~~~~~~~~~~~~~~~~ COM ~~~~~~~~~~~~~~~~~~~~~~~~~##
    
    # COM of the input coordinates
    # print("DEBUG: masses=", masses.tolist())
    # print("DEBUG: coords=", coords.tolist())
    # replicate mass values (Nx1) 3 times to match the coords array dimensions (Nx3)
    com = (array([masses, masses, masses]).transpose() * coords).sum(axis=0) / masses.sum()
    
    # get the distances to the center of mass
    dist_com = distance_to_point(coords, com)
    dist_list.append(dist_com)

    # get the closest and furthest coordinate to/from the centroid
    cstm, fctm = coords[dist_com.argmin()], coords[dist_com.argmax()]

    # get the distance distributions for the points that are closest/furthest
    # to/from the centroid
    dist_cstm = distance_to_point(coords, cstm)
    dist_fctm = distance_to_point(coords, fctm)
    # save the new points
    dist_list.append(dist_cstm)
    dist_list.append(dist_fctm)

    # get the point that is the furthest from the point that is furthest from
    # the centroid
    ftfm = coords[dist_fctm.argmax()]
    dist_ftfm = distance_to_point(coords, ftfm)
    dist_list.append(dist_ftfm)
    
    if core_ctd != None:
        # get the distances to the centroid
        dist_core_ctd = distance_to_point(coords, core_ctd)
        dist_list.append(dist_core_ctd)
    
        # get the closest and furthest coordinate to/from the centroid
        core_cst, core_fct = coords[dist_core_ctd.argmin()], coords[dist_core_ctd.argmax()]
    
        # get the distance distributions for the points that are closest/furthest
        # to/from the centroid
        dist_core_cst = distance_to_point(coords, core_cst)
        dist_core_fct = distance_to_point(coords, core_fct)
        # save the new points
        dist_list.append(dist_core_cst)
        dist_list.append(dist_core_fct)
    
        # get the point that is the furthest from the point that is furthest from
        # the centroid
        core_ftf = coords[dist_core_fct.argmax()]
        dist_core_ftf = distance_to_point(coords, core_ftf)
        dist_list.append(dist_core_ftf)
        
    if core_com != None:
        # get the distances to the center of mass
        dist_core_com = distance_to_point(coords, core_com)
        dist_list.append(dist_core_com)
    
        # get the closest and furthest coordinate to/from the centroid
        core_cstm, core_fctm = coords[dist_core_com.argmin()], coords[dist_core_com.argmax()]
    
        # get the distance distributions for the points that are closest/furthest
        # to/from the center of mass
        dist_core_cstm = distance_to_point(coords, core_cstm)
        dist_core_fctm = distance_to_point(coords, core_fctm)
        # save the new points
        dist_list.append(dist_core_cstm)
        dist_list.append(dist_core_fctm)
    
        # get the point that is the furthest from the point that is furthest from
        # the center of mass
        core_ftfm = coords[dist_core_fctm.argmax()]
        dist_core_ftfm = distance_to_point(coords, core_ftfm)
        dist_list.append(dist_core_ftfm)
        
        # Calculate the distance distributions between each new point and all atoms of the molecule
        for point in point_list:
            dist_list.append(distance_to_point(coords, point))
    """
    
    ##
    ##  Reduce 3D to 2D using Isomap and calculate the centroid
    ##
    from sklearn.manifold import SpectralEmbedding
    # im =  Isomap(n_components=2, n_neighbors=4, n_jobs=-1)
    # im = MDS(n_components=2, n_init=1, max_iter=100)
    # im = TSNE(n_components=2, random_state=0)
    im = SpectralEmbedding(n_components=2, n_neighbors=4)
    coords_2D = im.fit_transform(coords)
    # centroid of the input coordinates
    ctd_2D = coords_2D.mean(axis=0)
    
    # get the distances to the centroid
    dist_ctd_2D = distance_to_point(coords_2D, ctd_2D)
    dist_list.append(dist_ctd_2D)

    # get the closest and furthest coordinate to/from the centroid
    cst_2D, fct_2D = coords_2D[dist_ctd_2D.argmin()], coords_2D[dist_ctd_2D.argmax()]

    # get the distance distributions for the points that are closest/furthest
    # to/from the centroid
    dist_cst_2D = distance_to_point(coords_2D, cst_2D)
    dist_fct_2D = distance_to_point(coords_2D, fct_2D)
    # save the new points
    dist_list.append(dist_cst_2D)
    dist_list.append(dist_fct_2D)

    # get the point that is the furthest from the point that is furthest from
    # the centroid
    ftf_2D = coords_2D[dist_fct_2D.argmax()]
    dist_ftf_2D = distance_to_point(coords_2D, ftf_2D)
    dist_list.append(dist_ftf_2D)
    
    
    # add the 4 original points at the beginning of the point list in order to keep the same order as in dist_list
    # point_list = [ctd, cst, fct, ftf, core_ctd, core_cst, core_fct, core_ftf] + point_list
    point_list = [ctd, cst, fct, ftf, ctd_2D, cst_2D, fct_2D, ftf_2D] + point_list
    
    # calculate the first three moments for each of the four distance distributions
    if moment_number == 4:
        moments = concatenate([(ar.mean(), ar.std(), cbrt(skew(ar)), kurtosis(ar) )
                           for ar in dist_list])
    else:
        moments = concatenate([(ar.mean(), ar.std(), cbrt(skew(ar)))
                           for ar in dist_list])
    
    # return the USR moments as well as the four points for later re-use
    return point_list, moments


def usr_moments_with_existing(coords, *point_args, **kargs):
    """
    Calculates the USR moments for a set of coordinates and an already existing
    set of four USR reference points.
    """
    moment_number = kargs['moment_number']
    dist_list = []  # list with the distance distributions of every point from all the atoms of the molecule
    # print("DEBUG: point_args=", point_args)
    for point in point_args[:4]:
        dist_list.append(distance_to_point(coords, point))
    
    # print("DEBUG usr_moments_with_existing: dist_list=", dist_list)
    if len(coords) > 3:
        from sklearn.manifold import SpectralEmbedding
        # im =  Isomap(n_components=2, n_neighbors=2, n_jobs=-1)
        # im = MDS(n_components=2, n_init=1, max_iter=100)
        # im = TSNE(n_components=2, random_state=0)
        im = SpectralEmbedding(n_components=2, n_neighbors=2)
        coords_2D = im.fit_transform(coords)
        for point in point_args[4:]:    # iterate over the 2D points
            dist_list.append(distance_to_point(coords_2D, point))
    else:
        for point in point_args[:4]:
            dist_list.append(array(coords.shape[0]*[0]))
    # print("DEBUG: dist_ctd=np.array(", dist_ctd.tolist(), ")")
    # print("DEBUG: dist_cst=np.array(", dist_cst.tolist(), ")")
    # print("DEBUG: dist_fct=np.array(", dist_fct.tolist(), ")")
    # print("DEBUG: dist_ftf=np.array(", dist_ftf.tolist(), ")")
    
    if moment_number == 4:
        moments = concatenate([(ar.mean(), ar.std(), cbrt(skew(ar)), kurtosis(ar) )
                           for ar in dist_list])
    else:
        moments = concatenate([(ar.mean(), ar.std(), cbrt(skew(ar)))
                           for ar in dist_list])

    return moments