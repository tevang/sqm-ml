"""
This module contains similarity functions that can be used to compare USRCAT
moments.
"""
from numpy import abs, arange, zeros
from usrcat import NUM_MOMENTS

def similarity(m1, m2, ow=1.0, hw=1.0, rw=1.0, aw=1.0, dw=1.0):
    """
    Returns the maximum USRCAT similarity between two arrays of USR moments. The
    function will only calculate the similarity between the first row of the first
    moment array (should be the LEC) and all the other rows of the second. This
    function should be seen as a starting point of how to roll your own similarity
    metric; you could modify it to calculate an all-by-all matrix or change the
    way the pharmacophore weights are used to scale the final result.

    :param m1: numpy.ndarray
    :param m2: numpy.ndarray
    """
    weights = [ow, hw, rw, aw, dw]

    # the scale term is used to normalize the distance between the moments by
    # the USRCAT weights that were used. For example if all weights are 1.0 then
    # the scale will be 60. On the other hand, if only ow is 1.0 and the rest 0
    # then the scale will be 12, defaulting to classic USR.
    scale = 12 * sum(weights)

    # a list of the subset indices in the USRCAT array, e.g. 0, 12, 24, 36...
    idx = arange(0, NUM_MOMENTS+1, 12)

    # array that will hold the temporary similarity scores
    scores = zeros(m2.shape[0])

    # this loop will only calculate the similarity between the first row of the
    # first moment array (which should be the LEC) and all the rows of the second.
    for i in range(m2.shape[0]):
        score = 0

        for x, y, w in zip(idx, idx[1:], weights):
            score += w * abs(m1[0,x:y]-m2[i,x:y]).sum()

        scores[i] = 1.0 / (1.0 + score / scale)

    # return the highest similarity score as well as the conformer number
    return (scores.argmax(), scores.max())