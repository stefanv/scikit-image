""" cpmorphology.py - morphological operations not in scipy

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""

import numpy as np
from scipy import ndimage
import scipy.sparse

from outline import outline
from ..filter import rank_order
from index import Indexes

import _cpmorphology
from _cpmorphology2 import skeletonize_loop, table_lookup_index
from _cpmorphology2 import grey_reconstruction_loop
from _cpmorphology2 import _all_connected_components
from _cpmorphology2 import index_lookup, prepare_for_index_lookup
from _cpmorphology2 import extract_from_image_lookup, fill_labeled_holes_loop

try:
    from _cpmorphology2 import ptrsize
except:
    pass

from .selem import disk as strel_disk

from .. import get_log as _get_log

logger = _get_log('scikits.image.morphology')

'''A structuring element for eight-connecting a neigborhood'''
_eight_connect = ndimage.generate_binary_structure(2, 2)

'''A structuring element for four-connecting a neigborhood'''
_four_connect = ndimage.generate_binary_structure(2, 1)

def fill_labeled_holes(labels, mask=None, size_fn = None):
    '''Fill all background pixels that are holes inside the foreground

    A pixel is a hole inside a foreground object if

    * there is no path from the pixel to the edge AND

    * there is no path from the pixel to any other non-hole
      pixel AND

    * there is no path from the pixel to two similarly-labeled pixels that
      are adjacent to two differently labeled non-hole pixels.

    labels - the current labeling

    mask - mask of pixels to ignore

    size_fn - if not None, it is a function that takes a size and a boolean
              indicating whether it is foreground (True) or background (False)
              The function should return True to analyze and False to ignore

    returns a filled copy of the labels matrix
    '''
    #
    # The algorithm:
    #
    # Label the background to get distinct background objects
    # Construct a graph of both foreground and background objects.
    # Walk the graph according to the rules.
    #
    labels_type = labels.dtype
    background = labels == 0
    if mask is not None:
        background &= mask

    blabels, count = ndimage.label(background, _four_connect)
    lcount = np.max(labels)
    labels = labels.copy().astype(int)
    labels[blabels != 0] = blabels[blabels != 0] + lcount + 1
    lmax = lcount + count + 1
    is_not_hole = np.ascontiguousarray(np.zeros(lmax + 1, np.uint8))
    #
    # Find the indexes on the edge and use to populate the to-do list
    #
    to_do = np.unique(np.hstack((
        labels[0, :], labels[:, 0], labels[-1, :], labels[:, -1])))
    to_do = to_do[to_do != 0]
    is_not_hole[to_do] = True
    to_do = list(to_do)
    #
    # An array that names the first non-hole object
    #
    adjacent_non_hole = np.ascontiguousarray(np.zeros(lmax + 1), np.uint32)
    #
    # Find all 4-connected adjacent pixels
    #
    a = np.unique(labels[:-1, :] + labels[1:, :] * (lmax + 1))
    a = np.unique(np.hstack([a, np.unique(labels[:, :-1] + labels[:, 1:] *
                                          (lmax + 1))]))
    i, j = (a % (lmax + 1), (a / (lmax + 1)).astype(int))
    i, j = i[i != j], j[i != j]
    if (len(i)) > 0:
        order = np.lexsort((j, i))
        i = i[order]
        j = j[order]
        #
        # Now we make a ragged array of i and j
        #
        i_count = np.bincount(i)
        if len(i_count) < lmax + 1:
            i_count = np.hstack((i_count, np.zeros(lmax + 1 - len(i_count),
                                 int)))
        indexer = Indexes([i_count])
        #
        # Filter using the size function passed, if any
        #
        if size_fn is not None:
            areas = np.bincount(labels.flatten())
            for ii, area in enumerate(areas):
                if (ii > 0 and area > 0 and not is_not_hole[ii] and
                    not size_fn(area, ii <= lcount)):
                    is_not_hole[ii] = True
                    to_do.append(ii)

        to_do_count = len(to_do)
        if len(to_do) < len(is_not_hole):
            to_do += [ 0 ] * (len(is_not_hole) - len(to_do))
        to_do = np.ascontiguousarray(np.array(to_do), np.uint32)
        fill_labeled_holes_loop(
            np.ascontiguousarray(i, np.uint32),
            np.ascontiguousarray(j, np.uint32),
            np.ascontiguousarray(indexer.fwd_idx, np.uint32),
            np.ascontiguousarray(i_count, np.uint32),
            is_not_hole, adjacent_non_hole, to_do, lcount, to_do_count)
    #
    # Make an array that assigns objects to themselves and background to 0
    #
    new_indexes = np.arange(len(is_not_hole)).astype(np.uint32)
    new_indexes[(lcount+1):] = 0
    #
    # Fill the holes by replacing the old value by the value of the
    # enclosing object.
    #
    is_not_hole = is_not_hole.astype(bool)
    new_indexes[~is_not_hole] = adjacent_non_hole[~ is_not_hole]
    if mask is not None:
        labels[mask] = new_indexes[labels[mask]]
    else:
        labels = new_indexes[labels]
    return labels.astype(labels_type)

def adjacent(labels):
    '''Return a binary mask of all pixels which are adjacent to a pixel of
       a different label.

    '''
    high = labels.max()+1
    if high > np.iinfo(labels.dtype).max:
        labels = labels.astype(np.int)
    image_with_high_background = labels.copy()
    image_with_high_background[labels == 0] = high
    min_label = ndimage.minimum_filter(image_with_high_background,
                                     footprint=np.ones((3,3),bool),
                                     mode = 'constant',
                                     cval = high)
    max_label = ndimage.maximum_filter(labels,
                                     footprint=np.ones((3,3),bool),
                                     mode = 'constant',
                                     cval = 0)
    return (min_label != max_label) & (labels > 0)

def binary_thin(image, strel1, strel2):
    """Morphologically thin an image
    strel1 - the required values of the pixels in order to survive
    strel2 - at each pixel, the complement of strel1 if we care about the value
    """
    hit_or_miss = ndimage.binary_hit_or_miss(image, strel1, strel2)
    return np.logical_and(image,np.logical_not(hit_or_miss))

_binary_shrink_top_right = None
_binary_shrink_bottom_left = None
def binary_shrink_old(image, iterations=-1):
    """Shrink an image by repeatedly removing pixels which have partners
       above, to the left, to the right and below until the image doesn't change

       image - binary image to be manipulated
       iterations - # of times to shrink, -1 to shrink until idempotent

       There are horizontal/vertical thinners which detect a pixel on
       an edge with an interior pixel either horizontally or vertically
       attached like this:
       0  0  0
       X  1  X
       X  1  X
       and there are much more specific diagonal thinners which detect
       a pixel on the edge of a diagonal, like this:
       0  0  0
       0  1  0
       0  0  1
       Rotate each of these 4x to get the four directions for each
    """
    global _binary_shrink_top_right, _binary_shrink_bottom_left
    if _binary_shrink_top_right is None:
        #
        # None of these patterns can remove both of two isolated
        # eight-connected pixels. Taken together, they can remove any
        # pixel touching a background pixel.
        #
        # The top right pixels:
        #
        # 0xx
        # ..0
        # ...
        #
        _binary_shrink_top_right = _make_table(False,
                                               np.array([[0,0,0],
                                                         [0,1,0],
                                                         [0,1,0]],bool),
                                               np.array([[1,1,1],
                                                         [0,1,0],
                                                         [0,1,0]],bool))
        _binary_shrink_top_right &= _make_table(False,
                                                np.array([[0,0,0],
                                                          [0,1,0],
                                                          [1,0,0]], bool),
                                                np.array([[1,1,1],
                                                          [0,1,1],
                                                          [1,0,1]], bool))
        _binary_shrink_top_right &= _make_table(False,
                                                np.array([[0,0,0],
                                                          [1,1,0],
                                                          [0,0,0]], bool),
                                                np.array([[0,0,1],
                                                          [1,1,1],
                                                          [0,0,1]], bool))
        _binary_shrink_top_right &= _make_table(False,
                                                np.array([[0,0,0],
                                                          [1,1,0],
                                                          [0,1,1]], bool),
                                                np.array([[0,0,1],
                                                          [1,1,1],
                                                          [0,1,0]], bool))
        _binary_shrink_top_right &= _make_table(False,
                                                np.array([[0,0,0],
                                                          [0,1,0],
                                                          [0,0,1]], bool),
                                                np.array([[1,1,1],
                                                          [1,1,0],
                                                          [1,0,1]], bool))
        _binary_shrink_top_right &= _make_table(False,
                                                np.array([[0,0,0],
                                                          [0,1,0],
                                                          [1,1,1]], bool),
                                                np.array([[1,1,1],
                                                          [1,1,0],
                                                          [0,1,1]], bool))
        #
        # bottom left pixels
        #
        # ...
        # 0..
        # xx0
        _binary_shrink_bottom_left = _make_table(False,
                                                 np.array([[0,1,0],
                                                           [0,1,0],
                                                           [0,0,0]],bool),
                                                 np.array([[0,1,0],
                                                           [0,1,0],
                                                           [1,1,1]],bool))
        _binary_shrink_bottom_left &= _make_table(False,
                                                  np.array([[0,0,1],
                                                            [0,1,0],
                                                            [0,0,0]], bool),
                                                  np.array([[1,0,1],
                                                            [1,1,0],
                                                            [1,1,1]], bool))
        _binary_shrink_bottom_left &= _make_table(False,
                                                  np.array([[0,0,0],
                                                            [0,1,1],
                                                            [0,0,0]], bool),
                                                  np.array([[1,0,0],
                                                            [1,1,1],
                                                            [1,0,0]], bool))
        _binary_shrink_bottom_left &= _make_table(False,
                                                  np.array([[1,1,0],
                                                            [0,1,1],
                                                            [0,0,0]], bool),
                                                  np.array([[0,1,0],
                                                            [1,1,1],
                                                            [1,0,0]], bool))
        _binary_shrink_bottom_left &= _make_table(False,
                                                  np.array([[1,0,0],
                                                            [0,1,0],
                                                            [0,0,0]], bool),
                                                  np.array([[1,0,1],
                                                            [0,1,1],
                                                            [1,1,1]], bool))
        _binary_shrink_bottom_left &= _make_table(False,
                                                  np.array([[1,1,1],
                                                            [0,1,0],
                                                            [0,0,0]], bool),
                                                  np.array([[1,1,0],
                                                            [0,1,1],
                                                            [1,1,1]], bool))
    orig_image = image
    index_i, index_j, image = prepare_for_index_lookup(image, False)
    if iterations == -1:
        iterations = len(index_i)
    for i in range(iterations):
        pixel_count = len(index_i)
        for table in (binary_shrink_top_right,
                      binary_shrink_bottom_left):
            index_i, index_j = index_lookup(index_i, index_j,
                                            image, table, 1)
        if len(index_i) == pixel_count:
            break
    image = extract_from_image_lookup(orig_image, index_i, index_j)
    return image

_binary_shrink_ulr_table = None
_binary_shrink_urb_table = None
_binary_shrink_lrl_table = None
_binary_shrink_llt_table = None
_erode_table = None
def binary_shrink(image, iterations=-1):
    """Shrink an image by repeatedly removing pixels which have
    partners above, to the left, to the right and below until the
    image doesn't change.

    image - binary image to be manipulated
    iterations - # of times to shrink, -1 to shrink until idempotent

    There are horizontal/vertical thinners which detect a pixel on
    an edge with an interior pixel either horizontally or vertically
    attached like this:

    0  0  0
    X  1  X
    X  1  X

    and there are much more specific diagonal thinners which detect
    a pixel on the edge of a diagonal, like this:

    0  0  0
    0  1  0
    0  0  1

    Rotate each of these 4x to get the four directions for each

    """
    global _erode_table, _binary_shrink_ulr_table, _binary_shrink_lrl_table
    global _binary_shrink_urb_table, _binary_shrink_llt_table
    if _erode_table is None:
        #
        # The erode table hits all patterns that can be eroded without
        # changing the euler_number
        _erode_table = np.array([_pattern_of(index)[1,1] and
                                 (ndimage.label(_pattern_of(index-16))[1] != 1)
                                 for index in range(512)])
        _erode_table[_index_of(np.ones((3,3), bool))] = True
        #
        # Each other table is more specific: a specific corner or a specific
        # edge must be on where the corner and edge are not adjacent
        #
        _binary_shrink_ulr_table = (
            _erode_table |
            (_make_table(False, np.array([[0,0,0],
                                         [1,1,0],
                                         [0,0,0]], bool),
                        np.array([[0,0,0],
                                  [1,1,1],
                                  [0,0,0]],bool)) &
             _make_table(False, np.array([[1,0,0],
                                          [0,1,0],
                                          [0,0,0]],bool),
                         np.array([[1,0,0],
                                   [0,1,1],
                                   [0,1,1]],bool))))
        _binary_shrink_urb_table = (
            _erode_table |
            (_make_table(False, np.array([[0,1,0],
                                          [0,1,0],
                                          [0,0,0]], bool),
                         np.array([[0,1,0],
                                   [0,1,0],
                                   [0,1,0]],bool)) &
             _make_table(False, np.array([[0,0,1],
                                          [0,1,0],
                                          [0,0,0]],bool),
                         np.array([[0,0,1],
                                   [1,1,0],
                                   [1,1,0]],bool))))
        _binary_shrink_lrl_table = (
            _erode_table |
            (_make_table(False, np.array([[0,0,0],
                                          [0,1,1],
                                          [0,0,0]], bool),
                         np.array([[0,0,0],
                                   [1,1,1],
                                   [0,0,0]],bool)) &
             _make_table(False, np.array([[0,0,0],
                                          [0,1,0],
                                          [0,0,1]], bool),
                         np.array([[1,1,0],
                                   [1,1,0],
                                   [0,0,1]], bool))))
        _binary_shrink_llt_table = (
            _erode_table |
            (_make_table(False, np.array([[0,0,0],
                                          [0,1,0],
                                          [0,1,0]], bool),
                         np.array([[0,1,0],
                                   [0,1,0],
                                   [0,1,0]],bool)) &
             _make_table(False, np.array([[0,0,0],
                                          [0,1,0],
                                          [1,0,0]], bool),
                         np.array([[0,1,1],
                                   [0,1,1],
                                   [1,0,0]], bool))))

    orig_image = image
    index_i, index_j, image = prepare_for_index_lookup(image, False)
    if iterations == -1:
        iterations = len(index_i)
    for i in range(iterations):
        pixel_count = len(index_i)
        for table in (_binary_shrink_ulr_table,
                      _binary_shrink_urb_table,
                      _binary_shrink_lrl_table,
                      _binary_shrink_llt_table):
            index_i, index_j = index_lookup(index_i, index_j,
                                            image, table, 1)
        if len(index_i) == pixel_count:
            break
    image = extract_from_image_lookup(orig_image, index_i, index_j)
    return image

def maximum(image, structure=np.ones((3,3),dtype=bool), offset=None):
    """Find the local maximum at each point in the image,
    using the given structuring element

    image - a 2-d array of doubles
    structure - a boolean structuring element indicating which
                local elements should be sampled
    offset - the offset to the center of the structuring element
    """
    if not offset:
        offset = (structure.shape[0]/2,structure.shape[1]/2)
    offset = tuple(offset)
    return _cpmorphology.cpmaximum(image,structure,offset)

def relabel(image):
    """Given a labeled image, relabel each of the objects consecutively

    image - a labeled 2-d integer array
    returns - (labeled image, object count)
    """
    #
    # Build a label table that converts an old label # into
    # labels using the new numbering scheme
    #
    unique_labels = np.unique(image[image!=0])
    if len(unique_labels) == 0:
        return (image,0)
    consecutive_labels = np.arange(len(unique_labels))+1
    label_table = np.zeros(unique_labels.max()+1, int)
    label_table[unique_labels] = consecutive_labels
    #
    # Use the label table to remap all of the labels
    #
    new_image = label_table[image]
    return (new_image,len(unique_labels))

def convex_hull(labels, indexes=None):
    """Given a labeled image, return a list of points per object ordered by
    angle from an interior point, representing the convex hull.

    labels - the label matrix
    indexes - an array of label #s to be processed, defaults to all non-zero
              labels

    Returns a matrix and a vector. The matrix consists of one row per
    point in the convex hull. Each row has three columns, the label #,
    the i coordinate of the point and the j coordinate of the point. The
    result is organized first by label, then the points are arranged
    counter-clockwise around the perimeter.
    The vector is a vector of #s of points in the convex hull per label
    """
    if indexes == None:
        indexes = np.unique(labels)
        indexes.sort()
        indexes=indexes[indexes!=0]
    else:
        indexes=np.array(indexes)
    if len(indexes) == 0:
        return np.zeros((0,2),int),np.zeros((0,),int)
    #
    # Reduce the # of points to consider
    #
    outlines = outline(labels)
    coords = np.argwhere(outlines > 0).astype(np.int32)
    if len(coords)==0:
        # Every outline of every image is blank
        return (np.zeros((0,3),int),
                np.zeros((len(indexes),),int))

    i = coords[:,0]
    j = coords[:,1]
    labels_per_point = labels[i,j]
    pixel_labels = np.column_stack((i,j,labels_per_point))
    return convex_hull_ijv(pixel_labels, indexes)

def convex_hull_ijv(pixel_labels, indexes):
    """Return the convex hull for each label using an ijv labeling

    pixel_labels: the labeling of the pixels in i,j,v form where
                  i & j are the coordinates of a pixel and v is
                  the pixel's label number
    indexes: the indexes at which to measure the convex hull

    Returns a matrix and a vector. The matrix consists of one row per
    point in the convex hull. Each row has three columns, the label #,
    the i coordinate of the point and the j coordinate of the point. The
    result is organized first by label, then the points are arranged
    counter-clockwise around the perimeter.
    The vector is a vector of #s of points in the convex hull per label
    """

    if len(indexes) == 0:
        return np.zeros((0,3),int),np.zeros((0,),int)
    #
    # An array that converts from label # to index in "indexes"
    anti_indexes = np.zeros((np.max(indexes)+1,),int)
    anti_indexes[indexes] = range(len(indexes))

    coords = pixel_labels[:,:2]
    i = coords[:, 0]
    j = coords[:, 1]
    # This disgusting copy spooge appears to be needed for scipy 0.7.0
    labels_per_point = np.zeros(len(pixel_labels), int)
    labels_per_point[:] = pixel_labels[:,2]
    #
    # Calculate the centers for each label
    #
    center_i = _fixup_scipy_ndimage_result(
        ndimage.mean(i.astype(float), labels_per_point, indexes))
    center_j = _fixup_scipy_ndimage_result(
        ndimage.mean(j.astype(float), labels_per_point, indexes))
    centers = np.column_stack((center_i, center_j))
    #
    # Now make an array with one outline point per row and the following
    # columns:
    #
    # index of label # in indexes array
    # angle of the point relative to the center
    # i coordinate of the point
    # j coordinate of the point
    #
    anti_indexes_per_point = anti_indexes[labels_per_point]
    centers_per_point = centers[anti_indexes_per_point]
    angle = np.arctan2(i-centers_per_point[:,0],j-centers_per_point[:,1])
    a = np.zeros((len(i),3), np.int32)
    a[:,0] = anti_indexes_per_point
    a[:,1:] = coords
    #
    # Sort the array first by label # (sort of), then by angle
    #
    order = np.lexsort((angle,anti_indexes_per_point))
    #
    # Make unique
    #
    same_as_next = np.hstack([np.all(a[order[:-1],:] == a[order[1:],:], 1),
                                [False]])
    order = order[~same_as_next]
    a=a[order]
    anti_indexes_per_point = anti_indexes_per_point[order]
    angle = angle[order]
    centers_per_point = centers_per_point[order]
    #
    # Make the result matrix, leaving enough space so that all points might
    # be on the convex hull.
    #
    result = np.zeros((len(order),3), np.int32)
    result[:,0] = labels_per_point[order]
    #
    # Create an initial count vector
    #
    v = np.ones((a.shape[0],),dtype = np.int32)
    result_counts = scipy.sparse.coo_matrix((v,(a[:,0],v*0)),
                                            shape=(len(indexes),1))
    result_counts = result_counts.toarray().flatten()
    r_anti_indexes_per_point = anti_indexes_per_point # save this
    #
    # Create a vector that indexes into the results for each label
    #
    result_index = np.zeros(result_counts.shape, np.int32)
    result_index[1:]=np.cumsum(result_counts[:-1])
    #
    # Initialize the counts of convex hull points to a ridiculous number
    #
    counts = np.iinfo(np.int32).max
    first_pass = True
    while True:
        #
        # Figure out how many putative convex hull points there are for
        # each label.
        #
        # If the count for a label is 3 or less, it's a convex hull or
        # degenerate case.
        #
        # If the count hasn't changed in an iteration, then we've done
        # as well as we can hope to do.
        #
        v = np.ones((a.shape[0],),dtype = np.int32)
        new_counts = scipy.sparse.coo_matrix((v,(a[:,0],v*0)),
                                             shape=(len(indexes),1))
        new_counts = new_counts.toarray().flatten()
        done_count = (2 if first_pass else 3)
        finish_me = ((new_counts > 0) &
                     ((new_counts <= done_count) |
                      (new_counts == counts)))
        indexes_to_finish = np.argwhere(finish_me).astype(np.int32)
        keep_me = (new_counts > done_count) & (new_counts < counts)
        indexes_to_keep = np.argwhere(keep_me).astype(np.int32)
        if len(indexes_to_finish):
            result_counts[finish_me] = new_counts[finish_me]
            #
            # Store the coordinates of each of the points to finish
            #
            finish_this_row = finish_me[a[:,0]]
            rows_to_finish = np.argwhere(finish_this_row).flatten()
            a_to_finish = a[rows_to_finish]
            atf_indexes = a_to_finish[:,0]
            #
            # Map label #s to the index into indexes_to_finish of that label #
            #
            anti_indexes_to_finish = np.zeros((len(indexes),), np.int32)
            anti_indexes_to_finish[indexes_to_finish] = \
                                            range(len(indexes_to_finish))
            #
            # Figure out the indices of each point in a label to be finished.
            # We figure out how much to subtract for each label, then
            # subtract that much from 0:N to get successive indexes at
            # each label.
            # Then we add the result_index to figure out where to store it
            # in the result table.
            #
            finish_idx_base = np.zeros((len(indexes_to_finish),), np.int32)
            finish_idx_base[1:]=np.cumsum(new_counts[indexes_to_finish])[:-1]
            finish_idx_bases = finish_idx_base[anti_indexes_to_finish
                                                [atf_indexes]]
            finish_idx = (np.array(range(a_to_finish.shape[0]))-
                          finish_idx_bases)
            finish_idx = finish_idx + result_index[atf_indexes]
            result[finish_idx,1:] = a_to_finish[:,1:]
        if len(indexes_to_keep) == 0:
            break
        #
        # Figure out which points are still available
        #
        rows_to_keep = np.argwhere(keep_me[a[:,0].astype(np.int32)]).flatten()
        rows_to_keep = rows_to_keep.astype(np.int32)
        a = a[rows_to_keep]
        centers_per_point = centers_per_point[rows_to_keep]
        counts = new_counts
        #
        # The rule is that the area of the triangle from the center to
        # point N-1 to point N plus the area of the triangle from the center
        # to point N to point N+1 must be greater than the area of the
        # triangle from the center to point N-1 to point N+1 for a point
        # to be on the convex hull.
        # N-1 and N+1 have to be modulo "counts", so we make special arrays
        # to address those situations.
        #
        anti_indexes_to_keep = np.zeros((len(indexes),), np.int32)
        anti_indexes_to_keep[indexes_to_keep] = range(len(indexes_to_keep))
        idx_base = np.zeros((len(indexes_to_keep),), np.int32)
        idx_base[1:]=np.cumsum(counts[keep_me])[0:-1]
        idx_bases = idx_base[anti_indexes_to_keep[a[:,0]]]
        counts_per_pt = counts[a[:,0]]
        idx = np.array(range(a.shape[0]), np.int32)-idx_bases
        n_minus_one = np.mod(idx+counts_per_pt-1,counts_per_pt)+idx_bases
        n_plus_one  = np.mod(idx+1,counts_per_pt)+idx_bases
        #
        # Compute the triangle areas
        #
        t_left = _triangle_areas(centers_per_point,
                                 a[n_minus_one,1:],
                                 a[:,1:])
        t_right = _triangle_areas(centers_per_point,
                                  a[:,1:],
                                  a[n_plus_one,1:])
        t_lr = _triangle_areas(centers_per_point,
                               a[n_minus_one,1:],a[n_plus_one,1:])
        #
        # Keep the points where the area of the left triangle plus the
        # area of the right triangle is bigger than the area of the triangle
        # composed of the points to the left and right. This means that
        # there's a little triangle sitting on top of t_lr with our point
        # on top and convex in relation to its neighbors.
        #
        keep_me = t_left+t_right > t_lr
        #
        # If all points on a line are co-linear with the center, then the
        # whole line goes away. Special handling for this to find the points
        # most distant from the center and on the same side
        #
        consider_me = t_left+t_right == 0
        if np.any(consider_me):
            diff_i = a[:,1]-centers_per_point[:,0]
            diff_j = a[:,2]-centers_per_point[:,1]
            #
            # The manhattan distance is good enough
            #
            dist = np.abs(diff_i)+np.abs(diff_j)
            # The sign is different on different sides of a line including
            # the center. Multiply j by 2 to keep from colliding with i
            #
            # If both signs are zero, then the point is in the center
            #
            sign = np.sign(diff_i) + np.sign(diff_j)*2
            n_minus_one_consider = n_minus_one[consider_me]
            n_plus_one_consider = n_plus_one[consider_me]
            left_is_worse = (
                (dist[consider_me] > dist[n_minus_one_consider]) |
                (sign[consider_me] != sign[n_minus_one_consider]))
            right_is_worse = ((dist[consider_me] > dist[n_plus_one_consider]) |
                              (sign[consider_me] != sign[n_plus_one_consider]))
            to_keep = left_is_worse & right_is_worse & (sign[consider_me] != 0)
            keep_me[consider_me] = to_keep
        a = a[keep_me,:]
        centers_per_point = centers_per_point[keep_me]
        first_pass = False
    #
    # Finally, we have to shrink the results. We number each of the
    # points for a label, then only keep those whose indexes are
    # less than the count for their label.
    #
    within_label_index = np.array(range(result.shape[0]), np.int32)
    counts_per_point = result_counts[r_anti_indexes_per_point]
    result_indexes_per_point = result_index[r_anti_indexes_per_point]
    within_label_index = (within_label_index - result_indexes_per_point)
    result = result[within_label_index < counts_per_point]
    return result, result_counts


def _triangle_areas(p1,p2,p3):
    """Compute an array of triangle areas given three arrays of triangle pts
    
    p1,p2,p3 - three Nx2 arrays of points
    """
    v1 = p2-p1
    v2 = p3-p1
    cross1 = v1[:,1] * v2[:,0]
    cross2 = v2[:,1] * v1[:,0]
    a = (cross1-cross2) / 2
    #
    # Handle small round-off errors
    #
    a[a<np.finfo(np.float32).eps] = 0
    return a  



def _fixup_scipy_ndimage_result(whatever_it_returned):
    """Convert a result from scipy.ndimage to a numpy array

    scipy.ndimage has the annoying habit of returning a single, bare
    value instead of an array if the indexes passed in are of length 1.
    For instance:
    ndimage.maximum(image, labels, [1]) returns a float
    but
    ndimage.maximum(image, labels, [1,2]) returns a list
    """
    if getattr(whatever_it_returned,"__getitem__",False):
        return np.array(whatever_it_returned)
    else:
        return np.array([whatever_it_returned])


def centers_of_labels(labels):
    '''Return the i,j coordinates of the centers of a labels matrix
    
    The result returned is an 2 x n numpy array where n is the number
    of the label minus one, result[0,x] is the i coordinate of the center
    and result[x,1] is the j coordinate of the center.
    You can unpack the result as "i,j = centers_of_labels(labels)"
    '''
    max_labels = np.max(labels)
    if max_labels == 0:
        return np.zeros((2,0),int)
    
    result = ndimage.center_of_mass(np.ones(labels.shape),
                                  labels,
                                  np.arange(max_labels)+1)
    result = np.array(result)
    if result.ndim == 1:
        result.shape = (2,1)
        return result
    return result.transpose()


def farthest_from_edge(labels, indices):
    """Return coords of the pixel in each object farthest from the edge

    labels - find the centers in this

    Returns a 2 x n matrix of the i and j positions
    """
    return maximum_position_of_labels(distance_to_edge(labels), labels, indices)


def minimum_enclosing_circle(labels, indexes = None,
                             hull_and_point_count = None):
    """Find the location of the minimum enclosing circle and its radius

    labels - a labels matrix
    indexes - an array giving the label indexes to be processed
    hull_and_point_count - convex_hull output if already done. None = calculate

    returns an Nx3 array organized as i,j of the center and radius
    Algorithm from
    http://www.personal.kent.edu/~rmuhamma/Compgeometry/MyCG/CG-Applets/Center/centercli.htm
    who calls it the Applet's Algorithm and ascribes it to Pr. Chrystal
    The original citation is Professor Chrystal, "On the problem to construct
    the minimum circle enclosing n given points in a plane", Proceedings of
    the Edinburgh Mathematical Society, vol 3, 1884
    """
    if indexes == None:
        if hull_and_point_count is not None:
            indexes = np.array(np.unique(hull_and_point_count[0][:,0]),dtype=np.int32)
        else:
            max_label = np.max(labels)
            indexes = np.array(range(1,max_label+1),dtype=np.int32)
    else:
        indexes = np.array(indexes,dtype=np.int32)
    if indexes.shape[0] == 0:
        return np.zeros((0,2)),np.zeros((0,))

    if hull_and_point_count is None:
        hull, point_count = convex_hull(labels, indexes)
    else:
        hull, point_count = hull_and_point_count
    centers = np.zeros((len(indexes),2))
    radii = np.zeros((len(indexes),))
    #
    # point_index is the index to the first point in "hull" for a label
    #
    point_index = np.zeros((indexes.shape[0],),int)
    point_index[1:] = np.cumsum(point_count[:-1])
    #########################################################################
    #
    # The algorithm is this:
    # * Choose a line S from S0 to S1 at random from the set of adjacent
    #   S0 and S1
    # * For every vertex (V) other than S, compute the angle from S0
    #   to V to S. If this angle is obtuse, the vertex V lies within the
    #   minimum enclosing circle and can be ignored.
    # * Find the minimum angle for all V.
    #   If the minimum angle is obtuse, stop and accept S as the diameter of
    #   the circle.
    # * If the vertex with the minimum angle makes angles S0-S1-V and
    #   S1-S0-V that are acute and right, then take S0, S1 and V as the
    #   triangle within the circumscribed minimum enclosing circle.
    # * Otherwise, find the largest obtuse angle among S0-S1-V and
    #   S1-S0-V (V is the vertex with the minimum angle, not all of them).
    #   If S0-S1-V is obtuse, make V the new S1, otherwise make V the new S0
    #
    ##########################################################################
    #
    # anti_indexes is used to transform a label # into an index in the above array
    # anti_indexes_per_point gives the label index of any vertex
    #
    anti_indexes=np.zeros((np.max(indexes)+1,),int)
    anti_indexes[indexes] = range(indexes.shape[0])
    anti_indexes_per_point = anti_indexes[hull[:,0]]
    #
    # Start out by eliminating the degenerate cases: 0, 1 and 2
    #
    centers[point_count==0,:]= np.NaN
    if np.all(point_count == 0):
        # Bail if there are no points in any hull to prevent
        # index failures below.
        return centers,radii

    centers[point_count==1,:]=hull[point_index[point_count==1],1:]
    radii[point_count < 2]=0
    centers[point_count==2,:]=(hull[point_index[point_count==2],1:]+
                               hull[point_index[point_count==2]+1,1:])/2
    distance = centers[point_count==2,:] - hull[point_index[point_count==2],1:]
    radii[point_count==2]=np.sqrt(distance[:,0]**2+distance[:,1]**2)
    #
    # Get rid of the degenerate points
    #
    keep_me = point_count > 2
    #
    # Pick S0 as the first point in each label
    # and S1 as the second.
    #
    s0_idx = point_index.copy()
    s1_idx = s0_idx+1
    #
    # number each of the points in a label with an index # which gives
    # the order in which we'll get their angles. We use this to pick out
    # points # 2 to N which are the candidate vertices to S
    #
    within_label_indexes = (np.array(range(hull.shape[0]),int) -
                            point_index[anti_indexes_per_point])

    while(np.any(keep_me)):
        #############################################################
        # Label indexing for active labels
        #############################################################
        #
        # labels_to_consider contains the labels of the objects which
        # have not been completed
        #
        labels_to_consider = indexes[keep_me]
        #
        # anti_indexes_to_consider gives the index into any vector
        # shaped similarly to labels_to_consider (for instance, min_angle
        # below) for every label in labels_to_consider.
        #
        anti_indexes_to_consider =\
            np.zeros((np.max(labels_to_consider)+1,),int)
        anti_indexes_to_consider[labels_to_consider] = \
            np.array(range(labels_to_consider.shape[0]))
        ##############################################################
        # Vertex indexing for active vertexes other than S0 and S1
        ##############################################################
        #
        # The vertices are hull-points with indexes of 2 or more
        # keep_me_vertices is a mask of the vertices to operate on
        # during this iteration
        #
        keep_me_vertices = np.logical_and(keep_me[anti_indexes_per_point],
                                             within_label_indexes >= 2)
        #
        # v is the vertex coordinates for each vertex considered
        #
        v  = hull[keep_me_vertices,1:]
        #
        # v_labels is the label from the label matrix for each vertex
        #
        v_labels = hull[keep_me_vertices,0]
        #
        # v_indexes is the index into "hull" for each vertex (and similarly
        # shaped vectors such as within_label_indexes
        #
        v_indexes=np.argwhere(keep_me_vertices).flatten().astype(np.int32)
        #
        # anti_indexes_per_vertex gives the index into "indexes" and
        # any similarly shaped array of per-label values
        # (for instance s0_idx) for each vertex being considered
        #
        anti_indexes_per_vertex = anti_indexes_per_point[keep_me_vertices]
        #
        # anti_indexes_to_consider_per_vertex gives the index into any
        # vector shaped similarly to labels_to_consider for each
        # vertex being analyzed
        #
        anti_indexes_to_consider_per_vertex = anti_indexes_to_consider[v_labels]
        #
        # Get S0 and S1 per vertex
        #
        s0 = hull[s0_idx[keep_me],1:]
        s1 = hull[s1_idx[keep_me],1:]
        s0 = s0[anti_indexes_to_consider_per_vertex]
        s1 = s1[anti_indexes_to_consider_per_vertex]
        #
        # Compute the angle S0-S1-V
        #
        # the first vector of the angles is between S0 and S1
        #
        s01 = (s0 - s1).astype(float)
        #
        # compute V-S1 and V-S0 at each of the vertices to be considered
        #
        vs0 = (v - s0).astype(float)
        vs1 = (v - s1).astype(float)
        #
        #` Take the dot product of s01 and vs1 divided by the length of s01 *
        # the length of vs1. This gives the cosine of the angle between.
        #
        dot_vs1s0 = (np.sum(s01*vs1,1) /
                     np.sqrt(np.sum(s01**2,1)*np.sum(vs1**2,1)))
        angle_vs1s0 = np.abs(np.arccos(dot_vs1s0))
        s10 = -s01
        dot_vs0s1 = (np.sum(s10*vs0,1) /
                     np.sqrt(np.sum(s01**2,1)*np.sum(vs0**2,1)))
        angle_vs0s1 = np.abs(np.arccos(dot_vs0s1))
        #
        # S0-V-S1 is pi - the other two
        #
        angle_s0vs1 = np.pi - angle_vs1s0 - angle_vs0s1
        assert np.all(angle_s0vs1 >= 0)
        #
        # Now we find the minimum angle per label
        #
        min_angle = ndimage.minimum(angle_s0vs1,v_labels,
                                  labels_to_consider)
        min_angle = _fixup_scipy_ndimage_result(min_angle)
        min_angle_per_vertex = min_angle[anti_indexes_to_consider_per_vertex]
        #
        # Calculate the index into V of the minimum angle per label.
        # Use "indexes" instead of labels_to_consider so we get something
        # with the same shape as keep_me
        #
        min_position = ndimage.minimum_position(angle_s0vs1,v_labels,
                                              indexes)
        min_position = _fixup_scipy_ndimage_result(min_position).astype(int)
        min_position = min_position.flatten()
        #
        # Case 1: minimum angle is obtuse or right. Accept S as the diameter.
        # Case 1a: there are no vertices. Accept S as the diameter.
        #
        vertex_counts = ndimage.sum(keep_me_vertices,
                                  hull[:,0],
                                  labels_to_consider)
        vertex_counts = _fixup_scipy_ndimage_result(vertex_counts)
        case_1 = np.logical_or(min_angle >= np.pi / 2,
                                  vertex_counts == 0)

        if np.any(case_1):
            # convert from a boolean over indexes_to_consider to a boolean
            # over indexes
            finish_me = np.zeros((indexes.shape[0],),bool)
            finish_me[anti_indexes[labels_to_consider[case_1]]] = True
            s0_finish_me = hull[s0_idx[finish_me],1:].astype(float)
            s1_finish_me = hull[s1_idx[finish_me],1:].astype(float)
            centers[finish_me] = (s0_finish_me + s1_finish_me)/2
            radii[finish_me] = np.sqrt(np.sum((s0_finish_me -
                                                     s1_finish_me)**2,1))/2
            keep_me[finish_me] = False
        #
        # Case 2: all angles for the minimum angle vertex are acute
        #         or right.
        #         Pick S0, S1 and the vertex with the
        #         smallest angle as 3 points on the circle. If you look at the
        #         geometry, the diameter is the length of S0-S1 divided by
        #         the cosine of 1/2 of the angle. The center of the circle
        #         is at the circumcenter of the triangle formed by S0, S1 and
        #         V.
        case_2 = keep_me.copy()
        case_2[angle_vs1s0[min_position] > np.pi/2] = False
        case_2[angle_vs0s1[min_position] > np.pi/2] = False
        case_2[angle_s0vs1[min_position] > np.pi/2] = False

        if np.any(case_2):
            #
            # Wikipedia (http://en.wikipedia.org/wiki/Circumcircle#Cartesian_coordinates)
            # gives the following:
            # D = 2(S0y Vx + S1y S0x - S1y Vx - S0y S1x - S0x Vy + S1x Vy)
            # D = 2(S0x (S1y-Vy) + S1x(Vy-S0y) + Vx(S0y-S1y)
            # x = ((S0x**2+S0y**2)(S1y-Vy)+(S1x**2+S1y**2)(Vy-S0y)+(Vx**2+Vy**2)(S0y-S1y)) / D
            # y = ((S0x**2+S0y**2)(Vx-S1x)+(S1x**2+S1y**2)(S0y-Vy)+(Vx**2+Vy**2)(S1y-S0y)) / D
            #
            ss0 = hull[s0_idx[case_2],1:].astype(float)
            ss1 = hull[s1_idx[case_2],1:].astype(float)
            vv  = v[min_position[case_2]].astype(float)
            Y = 0
            X = 1
            D = 2*(ss0[:,X] * (ss1[:,Y] - vv[:,Y]) +
                   ss1[:,X] * (vv[:,Y]  - ss0[:,Y]) +
                   vv[:,X]  * (ss0[:,Y] - ss1[:,Y]))
            x = (np.sum(ss0**2,1)*(ss1[:,Y] - vv[:,Y]) +
                 np.sum(ss1**2,1)*(vv[:,Y]  - ss0[:,Y]) +
                 np.sum(vv**2,1) *(ss0[:,Y] - ss1[:,Y])) / D
            y = (np.sum(ss0**2,1)*(vv[:,X]  - ss1[:,X]) +
                 np.sum(ss1**2,1)*(ss0[:,X] - vv[:,X]) +
                 np.sum(vv**2,1) *(ss1[:,X] - ss0[:,X])) / D
            centers[case_2,X] = x
            centers[case_2,Y] = y
            distances = ss0-centers[case_2]
            radii[case_2] = np.sqrt(np.sum(distances**2,1))
            keep_me[case_2] = False
        #
        # Finally, for anybody who's left, for each of S0-S1-V and
        # S1-S0-V, for V, the vertex with the minimum angle,
        # find the largest obtuse angle. The vertex of this
        # angle (S0 or S1) is inside the enclosing circle, so take V
        # and either S1 or S0 as the new S.
        #
        # This involves a relabeling of within_label_indexes. We replace
        # either S0 or S1 with V and assign V either 0 or 1
        #
        if np.any(keep_me):
            labels_to_consider = indexes[keep_me]
            indexes_to_consider = anti_indexes[labels_to_consider]
            #
            # Index into within_label_indexes for each V with the
            # smallest angle
            #
            v_obtuse_indexes = v_indexes[min_position[keep_me]]
            angle_vs0s1_to_consider = angle_vs0s1[min_position[keep_me]]
            angle_vs1s0_to_consider = angle_vs1s0[min_position[keep_me]]
            #
            # Do the cases where S0 is larger
            #
            s0_is_obtuse = angle_vs0s1_to_consider > np.pi/2
            if np.any(s0_is_obtuse):
                #
                # The index of the obtuse S0
                #
                v_obtuse_s0_indexes = v_obtuse_indexes[s0_is_obtuse]
                obtuse_s0_idx = s0_idx[indexes_to_consider[s0_is_obtuse]]
                #
                # S0 gets the within_label_index of the vertex
                #
                within_label_indexes[obtuse_s0_idx] = \
                    within_label_indexes[v_obtuse_s0_indexes]
                #
                # Assign V as the new S0
                #
                s0_idx[indexes_to_consider[s0_is_obtuse]] = v_obtuse_s0_indexes
                within_label_indexes[v_obtuse_s0_indexes] = 0
            #
            # Do the cases where S1 is larger
            #
            s1_is_obtuse = np.logical_not(s0_is_obtuse)
            if np.any(s1_is_obtuse):
                #
                # The index of the obtuse S1
                #
                v_obtuse_s1_indexes = v_obtuse_indexes[s1_is_obtuse]
                obtuse_s1_idx = s1_idx[indexes_to_consider[s1_is_obtuse]]
                #
                # S1 gets V's within_label_index and goes onto the list
                # of considered vertices.
                #
                within_label_indexes[obtuse_s1_idx] = \
                    within_label_indexes[v_obtuse_s1_indexes]
                #
                # Assign V as the new S1
                #
                s1_idx[indexes_to_consider[s1_is_obtuse]] = v_obtuse_s1_indexes
                within_label_indexes[v_obtuse_s1_indexes] = 1
    return centers, radii

def associate_by_distance(labels_a, labels_b, distance):
    '''Find the objects that are within a given distance of each other

    Given two labels matrices and a distance, find pairs of objects that
    are within the given distance of each other where the distance is
    the minimum distance between any point in the convex hull of the
    two objects.

    labels_a - first labels matrix
    labels_b - second labels matrix
    distance - distance to measure

    returns a n x 2 matrix where m[x,0] is the label number in labels1 and
    m[x,1] is the label number in labels2

    Algorithm for computing distance between convex polygons taken from
    Chin, "Optimal Algorithms for the Intersection and the Minimum Distance
    Problems Between Planar Polygons", IEEE Transactions on Computers,
    vol. C-32, # 12, December 1983
    '''
    if np.max(labels_a) == 0 or np.max(labels_b) == 0:
        return np.zeros((0,2),int)

    hull_a, point_counts_a = convex_hull(labels_a)
    hull_b, point_counts_b = convex_hull(labels_b)
    centers_a, radii_a = minimum_enclosing_circle(
        labels_a, hull_and_point_count = (hull_a, point_counts_a))
    centers_b, radii_b = minimum_enclosing_circle(
        labels_b, hull_and_point_count = (hull_b, point_counts_b))
    #
    # Make an indexer into the hull tables
    #
    indexer_a = np.cumsum(point_counts_a)
    indexer_a[1:] = indexer_a[:-1]
    indexer_a[0] = 0
    indexer_b = np.cumsum(point_counts_b)
    indexer_b[1:] = indexer_b[:-1]
    indexer_b[0] = 0
    #
    # Compute the distances between minimum enclosing circles =
    # distance - radius_a - radius_b
    #
    i,j = np.mgrid[0:len(radii_a),0:len(radii_b)]
    ab_distance = np.sqrt((centers_a[i,0]-centers_b[j,0])**2 +
                          (centers_a[i,1]-centers_b[j,1])**2)
    ab_distance_minus_radii = ab_distance - radii_a[i] - radii_b[j]
    # Account for roundoff error
    ab_distance_minus_radii -= np.sqrt(np.finfo(float).eps)
    #
    # Exclude from consideration ab_distance > distance and automatically
    # choose those whose centers are within the distance
    #
    ab_easy_wins = ab_distance <= distance
    ij_wins = np.dstack((hull_a[indexer_a[i[ab_easy_wins]],0],
                         hull_b[indexer_b[j[ab_easy_wins]],0]))
    ij_wins.shape = ij_wins.shape[1:]

    ab_consider = (ab_distance_minus_radii <= distance) & (~ ab_easy_wins)
    ij_consider = np.dstack((i[ab_consider], j[ab_consider]))
    ij_consider.shape = ij_consider.shape[1:]
    if np.product(ij_consider.shape) == 0:
        return ij_wins
    if True:
        wins = []
        distance2 = distance**2
        for ii,jj in ij_consider:
            a = hull_a[indexer_a[ii]:indexer_a[ii]+point_counts_a[ii],1:]
            b = hull_b[indexer_b[jj]:indexer_b[jj]+point_counts_b[jj],1:]
            d = minimum_distance2(a,centers_a[ii,:],
                                  b,centers_b[jj,:])
            if d <= distance2:
                wins.append((hull_a[indexer_a[ii],0],
                             hull_b[indexer_b[jj],0]))
        ij_wins = np.vstack((ij_wins, np.array(wins)))
        return ij_wins
    else:
        #
        # For each point in the hull, get the next point mod # of points in hull
        #
        hull_next_a = np.arange(hull_a.shape[0])+1
        hull_next_a[indexer_a+point_counts_a-1] = indexer_a
        hull_next_b = np.arange(hull_b.shape[0])+1
        hull_next_b[indexer_b+point_counts_b-1] = indexer_b
        #
        # Parallelize the algorithm for overlap
        #
        # For each pair of points i, i+1 mod n in the hull, and the test point t
        # the cross product of the vector from i to i+1 and the vector from i+1
        # to t should have the same sign.
        #
        next_b = hull_b[hull_next_b,1:]
        vector_b = hull_b[:,1:] - next_b
        #
        # For each i,j, we have to compare the centers_a against point_counts_b[j]
        # crosses.
        #
        b_len = point_counts_b[ij_consider[:,1]]
        b_index = np.cumsum(point_counts_b)
        b_elems = b_index[-1]
        b_index[1:] = b_index[:-1]
        b_index[0] = 0
        #
        # First create a vector that's b_elems long and every element contains an
        # index into the ij_consider vector. How we do this:
        # 1) mark the first element at a particular index by 1, all others = 0
        # 2) Erase the first 1
        # 3) Take the cumulative sum which will increment to 1 when it hits the
        #    first 1, again when it hits the second...etc.
        #
        b_indexer = np.zeros(b_elems, int)
        b_indexer[b_index[1:]] = 1
        b_indexer = np.cumsum(b_indexer)
        #
        # The sub-index is the index from 1 to n for each of the vertices
        # per b convex hull
        #
        b_sub_index = np.arange(b_elems) - b_index[b_indexer]
        #
        # For each element of b_indexer, get the i and j at that index
        #
        b_i = ij_consider[b_indexer,0]
        b_j = ij_consider[b_indexer,1]
        #
        # Compute the cross-products now
        #
        b_vector_b = vector_b[indexer_b[b_j]+b_sub_index,:]
        b_center_vector = (next_b[indexer_b[b_j]+b_sub_index,:] -
                           centers_a[indexer_a[b_i]])
        cross = (b_vector_b[:,0] * b_center_vector[:,1] -
                 b_vector_b[:,1] * b_center_vector[:,0])
        hits = (_all_true(cross > 0, b_index) | _all_true(cross < 0, b_index))

        ij_wins = np.vstack((ij_wins, ij_consider[hits,:]))
        ij_consider = ij_consider[~hits,:]
        if ij_consider.shape[0] == 0:
            return ij_wins

def minimum_distance2(hull_a, center_a, hull_b, center_b):
    '''Return the minimum distance or 0 if overlap between 2 convex hulls

    hull_a - list of points in clockwise direction
    center_a - a point within the hull
    hull_b - list of points in clockwise direction
    center_b - a point within the hull
    '''
    if hull_a.shape[0] < 3 or hull_b.shape[0] < 3:
        return _slow_minimum_distance2(hull_a, hull_b)
    else:
        return _faster_minimum_distance2(hull_a, center_a, hull_b, center_b)

def _slow_minimum_distance2(hull_a, hull_b):
    '''Do the minimum distance by exhaustive examination of all points'''
    d2_min = np.iinfo(int).max
    for a in hull_a:
        if within_hull(a, hull_b):
            return 0
    for b in hull_b:
        if within_hull(b, hull_a):
            return 0
    for pt_a in hull_a:
        for pt_b in hull_b:
            d2_min = min(d2_min, np.sum((pt_a - pt_b)**2))

    for h1, h2 in ((hull_a, hull_b), (hull_b, hull_a)):
        # Find the distance from a vertex in h1 to an edge in h2
        for pt1 in h1:
            prev_pt2 = h2[-1,:]
            for pt2 in h2:
                if (np.dot(pt2-prev_pt2,pt1-prev_pt2) > 0 and
                    np.dot(prev_pt2-pt2,pt1-pt2) > 0):
                    # points form an acute triangle, so edge is closer
                    # than vertices
                    d2_min = min(d2_min, distance2_to_line(pt1, prev_pt2, pt2))
                prev_pt2 = pt2
    return d2_min

def _faster_minimum_distance2(hull_a, center_a, hull_b, center_b):
    '''Do the minimum distance using the bimodal property of hull ordering

    '''
    #
    # Find the farthest vertex in b from some point within A. Find the
    # vertices within A visible from this point in B. If the point in A
    # is within B or the farthest vertex in B is within A, then the objects
    # intersect.
    #
    if within_hull(center_a, hull_b):
        return 0
    farthest_b = find_farthest(center_a, hull_b)
    if within_hull(hull_b[farthest_b,:], hull_a):
        return 0
    visible_b = find_visible(hull_b, center_a, farthest_b)
    #
    # Do the same for B
    if within_hull(center_b, hull_a):
        return 0
    farthest_a = find_farthest(center_b, hull_a)
    if within_hull(hull_a[farthest_a,:], hull_b):
        return 0
    visible_a = find_visible(hull_a, center_b, farthest_a)
    #
    # Now go from the first in A and last in B measuring distances
    # which should decrease as we move toward the best
    #
    i = visible_a[0]
    i_next = (i+1) % hull_a.shape[0]
    j = visible_b[1]
    j_next = (j+hull_b.shape[0]-1) % hull_b.shape[0]
    a = hull_a[i,:]
    a_next = hull_a[i_next,:]
    b = hull_b[j,:]
    b_next = hull_b[j_next,:]
    d2_min = np.sum((a-b)**2)

    while i != visible_a[1] and j != visible_b[0]:
        if _lines_intersect(a, a_next, b, b_next):
            return 0
        if (np.dot(b-b_next,a-b_next) > 0 and
            np.dot(b_next-b,a-b) > 0):
            # do the edge if better than the vertex
            d2a = distance2_to_line(b, a, a_next)
        else:
            # try the next vertex of a
            d2a = np.sum((a_next-b)**2)
        if (np.dot(a-a_next,b-a_next) > 0 and
            np.dot(a_next-a,b-a) > 0):
            d2b = distance2_to_line(a, b, b_next)
        else:
            d2b = np.sum((b_next-a)**2)
        if d2a < d2_min and d2a < d2b:
            # The edge of A is closer than the best or the b-edge
            # Take it and advance A
            d2_min = d2a
            a = a_next
            i = i_next
            i_next = (i+1) % hull_a.shape[0]
            a_next = hull_a[i_next,:]
        elif d2b < d2_min:
            # B is better. Take it and advance
            d2_min = d2b
            b = b_next
            j = j_next
            j_next = (j+hull_b.shape[0]-1) % hull_b.shape[0]
            b_next = hull_b[j_next,:]
        else:
            return d2_min
    #
    # Some more to do... either one more i or one more j
    #
    while i != visible_a[1]:
        d2_min = min(d2_min, np.sum((a_next-b)**2))
        a = a_next
        i = i_next
        i_next = (i+1) % hull_a.shape[0]
        a_next = hull_a[i_next,:]

    while j != visible_b[0]:
        d2_min = min(d2_min, np.sum((b_next-a)**2))
        b = b_next
        j = j_next
        j_next = (j+ hull_b.shape[0]-1) % hull_b.shape[0]
        b_next = hull_b[j_next,:]
    return d2_min

def _lines_intersect(pt1_p, pt2_p, pt1_q, pt2_q):
    '''Return true if two line segments intersect
    pt1_p, pt2_p - endpoints of first line segment
    pt1_q, pt2_q - endpoints of second line segment
    '''
    #
    # The idea here is to do the cross-product of the vector from
    # point 1 to point 2 of one segment against the cross products from
    # both points of the other segment. If any of the cross products are zero,
    # the point is colinear with the line. If the cross products differ in
    # sign, then one point is on one side of the line and the other is on
    # the other. If that happens for both, then the lines must cross.
    #
    for pt1_a, pt2_a, pt1_b, pt2_b in ((pt1_p, pt2_p, pt1_q, pt2_q),
                                       (pt1_q, pt2_q, pt1_p, pt2_p)):
        v_a = pt2_a-pt1_a
        cross_a_1b = np.cross(v_a, pt1_b-pt2_a)
        if cross_a_1b == 0 and _colinear_intersection_check(pt1_a, pt2_a, pt1_b):
            return True
        cross_a_2b = np.cross(v_a, pt2_b-pt2_a)
        if cross_a_2b == 0 and _colinear_intersection_check(pt1_a, pt2_a, pt2_b):
            return True
        if (cross_a_1b < 0) == (cross_a_2b < 0):
            return False
    return True

def _colinear_intersection_check(pt1_a, pt2_a, pt_b):
    '''Check that co-linear pt_b lies between pt1_a and pt2_a'''
    da = np.sum((pt2_a-pt1_a)**2)
    return np.sum((pt1_a - pt_b)**2) < da and np.sum((pt2_a - pt_b)**2) < da

def find_farthest(point, hull):
    '''Find the vertex in hull farthest away from a point'''
    d_start = np.sum((point-hull[0,:])**2)
    d_end = np.sum((point-hull[-1,:])**2)
    if d_start > d_end:
        # Go in the forward direction
        i = 1
        inc = 1
        term = hull.shape[0]
        d2_max = d_start
    else:
        # Go in the reverse direction
        i = hull.shape[0]-2
        inc = -1
        term = -1
        d2_max = d_end
    while i != term:
        d2 = np.sum((point - hull[i,:])**2)
        if d2 < d2_max:
            break
        i += inc
        d2_max = d2
    return i-inc

def find_visible(hull, observer, background):
    '''Given an observer location, find the first and last visible
       points in the hull

       The observer at "observer" is looking at the hull whose most distant
       vertex from the observer is "background. Find the vertices that are
       the furthest distance from the line between observer and background.
       These will be the start and ends in the vertex chain of vertices
       visible by the observer.
       '''
    pt_background = hull[background,:]
    vector = pt_background - observer
    i = background
    dmax = 0
    while True:
        i_next = (i+1) % hull.shape[0]
        pt_next = hull[i_next,:]
        d = -np.cross(vector, pt_next-pt_background)
        if d < dmax or i_next == background:
            i_min = i
            break
        dmax = d
        i = i_next
    dmax = 0
    i = background
    while True:
        i_next = (i+hull.shape[0]-1) % hull.shape[0]
        pt_next = hull[i_next,:]
        d = np.cross(vector, pt_next-pt_background)
        if d < dmax or i_next == background:
            i_max = i
            break
        dmax = d
        i = i_next
    return (i_min, i_max)

def distance2_to_line(pt, l0, l1):
    '''The perpendicular distance squared from a point to a line

    pt - point in question
    l0 - one point on the line
    l1 - another point on the line
    '''
    pt = np.atleast_1d(pt)
    l0 = np.atleast_1d(l0)
    l1 = np.atleast_1d(l1)
    reshape = pt.ndim == 1
    if reshape:
        pt.shape = l0.shape = l1.shape = (1, pt.shape[0])
    result = (((l0[:,0] - l1[:,0]) * (l0[:,1] - pt[:,1]) -
               (l0[:,0] - pt[:,0]) * (l0[:,1] - l1[:,1]))**2 /
              np.sum((l1-l0)**2, 1))
    if reshape:
        result = result[0]
    return result


def within_hull(point, hull):
    '''Return true if the point is within the convex hull'''
    h_prev_pt = hull[-1,:]
    for h_pt in hull:
        if np.cross(h_pt-h_prev_pt, point - h_pt) >= 0:
            return False
        h_prev_pt = h_pt
    return True

def _all_true(a, indexes):
    '''Find which vectors have all-true elements

    Given an array, "a" and indexes into the first elements of vectors
    within that array, return an array where each element is true if
    all elements of the corresponding vector are true.

    Example: a = [ 1,1,0,1,1,1,1], indexes=[0,3]
             vectors = [[1,1,0],[1,1,1,1]]
             return = [False, True]
    '''
    if len(indexes) == 0:
        return np.zeros(0,bool)
    elif len(indexes) == 1:
        return np.all(a)
    cs = np.zeros(len(a)+1,int)
    cs[1:] = np.cumsum(a)
    augmented_indexes = np.zeros(len(indexes)+1, int)
    augmented_indexes[0:-1] = indexes + 1
    augmented_indexes[-1] = len(a) + 1
    counts = augmented_indexes[1:]-augmented_indexes[0:-1]
    hits = cs[augmented_indexes[1:]-1] - cs[augmented_indexes[0:-1]-1]
    return counts == hits

def ellipse_from_second_moments(image, labels, indexes,
                                wants_compactness=False):
    """Calculate measurements of ellipses equivalent to the second
    moments of labels.

    image  - the intensity at each point
    labels - for each labeled object, derive an ellipse
    indexes - sequence of indexes to process

    returns the following arrays:
       coordinates of the center of the ellipse
       eccentricity
       major axis length
       minor axis length
       orientation
       compactness (if asked for)

    some definitions taken from "Image Moments-Based Structuring and Tracking
    of Objects", LOURENA ROCHA, LUIZ VELHO, PAULO CEZAR P. CARVALHO,
    http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2002/10.23.11.34/doc/35.pdf
    particularly equation 5 (which has some errors in it).
    These yield the rectangle with equivalent second moments. I translate
    to the ellipse by multiplying by 1.154701 which is Matlab's calculation
    of the major and minor axis length for a square of length X divided
    by the actual length of the side of a square of that length.

    eccentricity is the distance between foci divided by the major axis length
    orientation is the angle of the major axis with respect to the X axis
    compactness is the variance of the radial distribution normalized by the area
    """
    if len(indexes) == 0:
        return (np.zeros((0,2)), np.zeros((0,)), np.zeros((0,)),
                np.zeros((0,)),np.zeros((0,)))
    i,j = np.argwhere(labels != 0).transpose()
    return ellipse_from_second_moments_ijv(i,j,image[i,j], labels[i,j], indexes, wants_compactness)

def ellipse_from_second_moments_ijv(i, j, image, labels, indexes,
                                    wants_compactness=False):
    """Calculate measurements of ellipses equivalent to the second
    moments of labels.

    i,j - coordinates of each point
    image  - the intensity at each point
    labels - for each labeled object, derive an ellipse
    indexes - sequence of indexes to process

    returns the following arrays:
       coordinates of the center of the ellipse
       eccentricity
       major axis length
       minor axis length
       orientation

    some definitions taken from "Image Moments-Based Structuring and Tracking
    of Objects", LOURENA ROCHA, LUIZ VELHO, PAULO CEZAR P. CARVALHO,
    http://sibgrapi.sid.inpe.br/col/sid.inpe.br/banon/2002/10.23.11.34/doc/35.pdf
    particularly equation 5 (which has some errors in it).
    These yield the rectangle with equivalent second moments. I translate
    to the ellipse by multiplying by 1.154701 which is Matlab's calculation
    of the major and minor axis length for a square of length X divided
    by the actual length of the side of a square of that length.

    eccentricity is the distance between foci divided by the major axis length
    orientation is the angle of the major axis with respect to the X axis
    """
    if len(indexes) == 0:
        return [np.zeros((0,2))] + [np.zeros((0,))] * (5 if wants_compactness else 4)
    if len(i) == 0:
        return ([np.zeros((len(indexes), 2)), np.ones(len(indexes))] +
                [np.zeros(len(indexes))] * (4 if wants_compactness else 3))
    #
    # Normalize to center of object for stability
    #
    nlabels = np.max(indexes)+1
    m = np.array([[None, 0, None],
                  [0, None, None],
                  [None, None, None]], object)
    if np.all(image == 1):
        image = 1
        m[0,0] = intensity = np.bincount(labels)
    else:
        m[0,0] = intensity = np.bincount(labels, image)
    ic = np.bincount(labels, i * image) / intensity
    jc = np.bincount(labels, j * image) / intensity
    i = i - ic[labels]
    j = j - jc[labels]
    #
    # Start by calculating the moments m[p][q] of the image
    # sum(i**p j**q)
    #
    # m[1,0] = 0 via normalization
    # m[0,1] = 0 via normalization
    m[1,1] = np.bincount(labels, i*j*image)
    m[2,0] = np.bincount(labels, i*i*image)
    m[0,2] = np.bincount(labels, j*j*image)

    a = m[2,0] / m[0,0]
    b = 2*m[1,1]/m[0,0]
    c = m[0,2] / m[0,0]

    theta = np.arctan2(b,c-a) / 2
    temp = np.sqrt(b**2+(a-c)**2)
    #
    # If you do a linear regression of the circles from 1 to 50 radius
    # in Matlab, the resultant values fit a line with slope=.9975 and
    # intercept .095. I'm adjusting the lengths accordingly.
    #
    mystery_constant = 0.095
    mystery_multiplier = 0.9975
    major_axis_len = (np.sqrt(8*(a+c+temp)) * mystery_multiplier +
                      mystery_constant)
    minor_axis_len = (np.sqrt(8*(a+c-temp)) * mystery_multiplier +
                      mystery_constant)
    eccentricity = np.sqrt(1-(minor_axis_len / major_axis_len)**2)
    compactness = 2 * np.pi * (a + c) / m[0,0]
    return ([np.column_stack((ic[indexes], jc[indexes])),
             eccentricity[indexes],
             major_axis_len[indexes],
             minor_axis_len[indexes],
             theta[indexes]] +
            ([compactness[indexes]] if wants_compactness else []))

def calculate_extents(labels, indexes):
    """Return the area of each object divided by the area of its bounding box"""
    fix = _fixup_scipy_ndimage_result
    areas = fix(ndimage.sum(np.ones(labels.shape),labels,np.array(indexes, dtype=np.int32)))
    y,x = np.mgrid[0:labels.shape[0],0:labels.shape[1]]
    xmin = fix(ndimage.minimum(x, labels, indexes))
    xmax = fix(ndimage.maximum(x, labels, indexes))
    ymin = fix(ndimage.minimum(y, labels, indexes))
    ymax = fix(ndimage.maximum(y, labels, indexes))
    bbareas = (xmax-xmin+1)*(ymax-ymin+1)
    return areas / bbareas

# The perimeter scoring matrix provides the distance to the next point
#
#   To use this, the value at [i-1,j-1] is bit 0, [i-1,j] is bit 1, [i-1,j+1]
#   is bit 2, etc. of an index into the perimeter_scoring
#   the distance from the center point to the next point clockwise on the
#   perimeter. The values must be the label matrix == shifted label matrix
#
#   I came up with the idea for this independently, but while Googling,
#   found a reference to the same idea. The perimeter matrix is taken from
#   the reference:
#   Prashker, "An Improved Algorithm for Calculating the Perimeter and Area
#   of Raster Polygons", GeoComputation 99.
#    http://www.geovista.psu.edu/sites/geocomp99/Gc99/076/gc_076.htm
def _calculate_perimeter_scoring():
    """Return a 512 element vector which gives the perimeter given surrounding pts

    """
    #
    # This is the array from the paper - a 256 - element array leaving out
    # the center point. The first value is the index, the second, the perimeter
    #
    prashker = np.array([
        [0 ,4    ],[32,4    ],[64,3    ],[96 ,1.414],[128,4    ],[160,4    ],[192,1.414],[224,2.828],
        [1 ,4    ],[33,4    ],[65,3    ],[97 ,1.414],[129,4    ],[161,4    ],[193,3    ],[225,3    ],
        [2 ,3    ],[34,3    ],[66,2    ],[98 ,2    ],[130,3    ],[162,3    ],[194,2    ],[226,2    ],
        [3 ,1.414],[35,1.414],[67,2    ],[99 ,2    ],[131,3    ],[163,3    ],[195,2    ],[227,2    ],
        [4 ,4    ],[36,4    ],[68,3    ],[100,3    ],[132,4    ],[164,4    ],[196,1.414],[228,3    ],
        [5 ,4    ],[37,4    ],[69,3    ],[101,3    ],[133,4    ],[165,4    ],[197,3    ],[229,3    ],
        [6 ,1.414],[38,3    ],[70,2    ],[102,2    ],[134,1.414],[166,3    ],[198,2    ],[230,2    ],
        [7 ,2.828],[39,3    ],[71,2    ],[103,2    ],[135,3    ],[167,3    ],[199,2    ],[231,1.414],
        [8 ,3    ],[40,1.414],[72,2    ],[104,2    ],[136,3    ],[168,1.414],[200,1.414],[232,1.414],
        [9 ,1.414],[41,2.828],[73,1.414],[105,1.414],[137,3    ],[169,3    ],[201,1.414],[233,1.414],
        [10,2    ],[42,1.414],[74,1    ],[106,1    ],[138,2    ],[170,2    ],[202,1    ],[234,1.414],
        [11,2    ],[43,1.414],[75,1    ],[107,1    ],[139,2    ],[171,2    ],[203,1    ],[235,1    ],
        [12,3    ],[44,3    ],[76,2    ],[108,2    ],[140,3    ],[172,3    ],[204,2    ],[236,2    ],
        [13,1.414],[45,3    ],[77,2    ],[109,2    ],[141,3    ],[173,3    ],[205,1.414],[237,1.414],
        [14,1.414],[46,1.414],[78,1    ],[110,1    ],[142,2    ],[174,1.414],[206,2    ],[238,1    ],
        [15,1.414],[47,1.414],[79,1.414],[111,1    ],[143,2    ],[175,1.414],[207,1    ],[239,1    ],
        [16,3    ],[48,3    ],[80,2    ],[112,1.414],[144,1.414],[176,1.414],[208,2    ],[240,1.414],
        [17,3    ],[49,3    ],[81,2    ],[113,2    ],[145,3    ],[177,3    ],[209,2    ],[241,2    ],
        [18,2    ],[50,2    ],[82,1    ],[114,1    ],[146,1.414],[178,2    ],[210,1    ],[242,1.414],
        [19,1.414],[51,2    ],[83,1    ],[115,2    ],[147,1.414],[179,1.414],[211,1    ],[243,1    ],
        [20,1.414],[52,3    ],[84,1.414],[116,1.414],[148,2.828],[180,3    ],[212,1.414],[244,1.414],
        [21,1.414],[53,3    ],[85,2    ],[117,1.414],[149,3    ],[181,3    ],[213,2    ],[245,1.414],
        [22,2    ],[54,2    ],[86,1    ],[118,1    ],[150,1.414],[182,2    ],[214,1    ],[246,1    ],
        [23,1.414],[55,2    ],[87,1.414],[119,1    ],[151,1.414],[183,1.414],[215,1    ],[247,1    ],
        [24,2    ],[56,2    ],[88,1    ],[120,1    ],[152,2    ],[184,2    ],[216,1    ],[248,1    ],
        [25,2    ],[57,2    ],[89,1    ],[121,1.414],[153,2    ],[185,2    ],[217,1    ],[249,1    ],
        [26,1    ],[58,1    ],[90,0    ],[122,0    ],[154,1    ],[186,2    ],[218,0    ],[250,0    ],
        [27,1    ],[59,1.414],[91,0    ],[123,0    ],[155,1    ],[187,1    ],[219,0    ],[251,0    ],
        [28,2    ],[60,2    ],[92,1    ],[124,1    ],[156,2    ],[188,2    ],[220,1.414],[252,1    ],
        [29,2    ],[61,2    ],[93,2    ],[125,1    ],[157,2    ],[189,1.414],[221,1    ],[253,1    ],
        [30,1    ],[62,1    ],[94,0    ],[126,0    ],[158,1.414],[190,1    ],[222,0    ],[254,0    ],
        [31,1    ],[63,1    ],[95,0    ],[127,0    ],[159,1    ],[191,1    ],[223,0    ],[255,0]])
    score = np.zeros((512,))
    i = np.zeros((prashker.shape[0]),int)
    for j in range(4): # 1,2,4,8
        i = i+((prashker[:,0].astype(int) / 2**j)%2)*2**j
    i = i+16
    for j in range(4,8):
        i = i+((prashker[:,0].astype(int) / 2**j)%2)*2**(j+1)
    score[i.astype(int)] = prashker[:,1]
    return score

_perimeter_scoring = _calculate_perimeter_scoring()

def calculate_perimeters(labels, indexes):
    """Count the distances between adjacent pixels in the perimeters of the labels"""
    #
    # Create arrays that tell whether a pixel is like its neighbors.
    # index = 0 is the pixel -1,-1 from the pixel of interest, 1 is -1,0, etc.
    #
    m=np.zeros((labels.shape[0],labels.shape[1]),int)
    exponent = 0
    for i in range(-1,2):
        ilow = (i==-1 and 1) or 0
        iend = (i==1 and labels.shape[0]-1) or labels.shape[0]
        for j in range(-1,2):
            jlow = (j==-1 and 1) or 0
            jend = (j==1 and labels.shape[1]-1) or labels.shape[1]
            #
            # Points outside of bounds are different from what's outside,
            # so set untouched points to "different"
            #
            mask = np.zeros(labels.shape, bool)
            mask[ilow:iend, jlow:jend] = (labels[ilow:iend,jlow:jend] ==
                                          labels[ilow+i:iend+i,jlow+j:jend+j])
            m[mask] += 2**exponent
            exponent += 1
    pixel_score = _perimeter_scoring[m]
    return _fixup_scipy_ndimage_result(ndimage.sum(pixel_score, labels, np.array(indexes,dtype=np.int32)))

def calculate_convex_hull_areas(labels,indexes=None):
    """Calulculate the area of the convex hull of each labeled object

    labels - a label matrix
    indexes - None: calculate convex hull area over entire image
              number: calculate convex hull for a single label
              sequence: calculate convex hull for labels matching a sequence
                        member and return areas in same order.
    """
    if getattr(indexes,"__getitem__",False):
        indexes = np.array(indexes,dtype=np.int32)
    elif indexes != None:
        indexes = np.array([indexes],dtype=np.int32)
    else:
        labels = labels !=0
        indexes = np.array([1],dtype=np.int32)
    hull, counts = convex_hull(labels, indexes)
    result = np.zeros((counts.shape[0],))
    #
    # Get rid of the degenerate cases
    #
    result[counts==1] = 1 # a single point has area 1
    if not np.any(counts >1):
        return result
    #
    # Given a label number "index_of_label" indexes into the result
    #
    index_of_label = np.zeros((hull[:,0].max()+1),int)
    index_of_label[indexes] = np.array(range(indexes.shape[0]))
    #
    # hull_index is the index into hull of the first point on the hull
    # per label
    #
    hull_index = np.zeros((counts.shape[0],),int)
    hull_index[1:] = np.cumsum(counts[:-1])
    #
    # A 2-point case is a line. The area of a line is its length * 1
    # and its length needs to be expanded by 1 because the end-points are
    # at the limits, not the ends.
    #
    if np.any(counts==2):
        diff_2 = hull[hull_index[counts==2],1:]-hull[hull_index[counts==2]+1,1:]
        result[counts==2] = np.sqrt(np.sum(diff_2**2,1))+1
    if not np.any(counts>=3):
        return result
    #
    # Now do the non-degenerate cases (_nd)
    #
    counts_per_label = np.zeros((hull[:,0].max()+1),counts.dtype)
    counts_per_label[indexes] = counts
    hull_nd = hull[counts_per_label[hull[:,0]] >=3]
    counts_nd = counts[counts>=3]
    indexes_nd = indexes[counts>=3]
    index_of_label_nd = np.zeros((index_of_label.shape[0],),int)
    index_of_label_nd[indexes_nd] = np.array(range(indexes_nd.shape[0]))
    #
    # Figure out the within-label index of each point in a label. This is
    # so we can do modulo arithmetic when pairing a point with the next
    # when determining an edge
    #
    hull_index_nd = np.zeros((counts_nd.shape[0],),int)
    if hull_index_nd.shape[0] > 1:
        hull_index_nd[1:] = np.cumsum(counts_nd[:-1])
    index_of_label_per_pixel_nd = index_of_label_nd[hull_nd[:,0]]
    hull_index_per_pixel_nd = hull_index_nd[index_of_label_per_pixel_nd]
    within_label_index = (np.array(range(hull_nd.shape[0])) -
                          hull_index_per_pixel_nd)
    #
    # Find some point within each convex hull.
    #
    within_hull = np.zeros((counts_nd.shape[0],2))
    within_hull[:,0] = ndimage.sum(hull_nd[:,1],
                                   hull_nd[:,0],
                                   indexes_nd) / counts_nd
    within_hull[:,1] = ndimage.sum(hull_nd[:,2],
                                   hull_nd[:,0],
                                   indexes_nd) / counts_nd
    within_hull_per_pixel = within_hull[index_of_label_per_pixel_nd]
    #
    # Now, we do a little, slightly wierd fixup, arguing that the
    # edge of a pixel is +/- .5 of its coordinate. So we move the ones
    # left of center to the left by .5, right of center to the right by .5
    # etc.
    #
    # It works for a square...
    #
    hull_nd = hull_nd.astype(float)
    hull_nd[hull_nd[:,1] < within_hull_per_pixel[:,0],1]  -= .5
    hull_nd[hull_nd[:,2] < within_hull_per_pixel[:,1],2]  -= .5
    hull_nd[hull_nd[:,1] >= within_hull_per_pixel[:,0],1] += .5
    hull_nd[hull_nd[:,2] >= within_hull_per_pixel[:,1],2] += .5
    #
    # Finally, we go around the circle, computing triangle areas
    # from point n to point n+1 (modulo count) to the point within
    # the hull.
    #
    plus_one_idx = np.array(range(hull_nd.shape[0]))+1
    modulo_mask = within_label_index+1 == counts_nd[index_of_label_per_pixel_nd]
    plus_one_idx[modulo_mask] = hull_index_per_pixel_nd[modulo_mask]
    area_per_pt_nd = _triangle_areas(hull_nd[:,1:],
                                     hull_nd[plus_one_idx,1:],
                                     within_hull_per_pixel)
    #
    # The convex area is the sum of these triangles
    #
    result[counts>=3] = ndimage.sum(area_per_pt_nd, hull_nd[:,0], indexes_nd)
    return result

def calculate_solidity(labels,indexes=None):
    """Calculate the area of each label divided by the area of its convex hull

    labels - a label matrix
    indexes - the indexes of the labels to measure
    """
    if indexes is not None:
        """ Convert to compat 32bit integer """
        indexes = np.array(indexes,dtype=np.int32)
    areas = ndimage.sum(np.ones(labels.shape),labels,indexes)
    convex_hull_areas = calculate_convex_hull_areas(labels, indexes)
    return areas / convex_hull_areas

def euler_number(labels, indexes=None):
    """Calculate the Euler number of each label

    labels - a label matrix
    indexes - the indexes of the labels to measure or None to
              treat the labels matrix as a binary matrix
    """
    if indexes == None:
        labels = labels != 0
        indexes = np.array([1],dtype=np.int32)
    elif getattr(indexes,'__getitem__',False):
        indexes = np.array(indexes,dtype=np.int32)
    else:
        indexes = np.array([indexes],dtype=np.int32)
    fix = _fixup_scipy_ndimage_result
    #
    # The algorithm here is from the following reference:
    # S.B. Gray, "Local Properties of Binary Images in Two Dimensions",
    # IEEE Transactions on Computers, Vol c-20 # 5 p 551, May 1971
    #
    # The general idea is that crossings into objects can be measured locally
    # through counting numbers of patterns resulting in crossings. There
    # are three sets that are applicable in Euler Numbers:
    # Q1: 1 0  0 1  0 0  0 0 (or more simply, 1 bit per quad)
    #     0 0  0 0  1 0  0 1
    #
    # Q3: 0 1  1 0  1 1  1 1 (or 3 bits per quad)
    #     1 1  1 1  1 0  0 1
    #
    # QD: 1 0  0 1
    #     0 1  1 0
    #
    # and the Euler number = W of an object is
    #
    # 4W = n(Q1) - n(Q3) - 2n(QD) (equation 34)
    # W  = (n(Q1) - n(Q3) - 2n(QD))/4
    #
    # We shift the label matrix to make matrices, padded by zeros on the
    # sides for each of the four positions of the quad:
    # I00 I01
    # I10 I11
    #
    # We can then assign each bitquad to a label based on the value
    # of the label at one of the "on" bits. For example, the first pattern
    # of Q1 has the label I00 because that bit is on. It's truth value is
    # I00 != I01 and I00 != I02 and I00 != I03.
    #
    I_shape = (labels.shape[0]+3,labels.shape[1]+3)
    I00 = np.zeros(I_shape,int)
    I01 = np.zeros(I_shape,int)
    I10 = np.zeros(I_shape,int)
    I11 = np.zeros(I_shape,int)
    slice_00 = [slice(1,labels.shape[0]+1),slice(1,labels.shape[1]+1)]
    slice_01 = [slice(1,labels.shape[0]+1),slice(0,labels.shape[1])]
    slice_10 = [slice(labels.shape[0]),slice(1,labels.shape[1]+1)]
    slice_11 = [slice(0,labels.shape[0]),slice(0,labels.shape[1])]
    I00[slice_00] = labels
    I01[slice_01] = labels
    I10[slice_10] = labels
    I11[slice_11] = labels
    #
    # There are 6 binary comparisons among the four bits
    #
    EQ00_01 = I00 == I01;              EQ01_00 = EQ00_01
    EQ00_10 = I00 == I10;              EQ10_00 = EQ00_10
    EQ00_11 = I00 == I11;              EQ11_00 = EQ00_11
    EQ01_10 = I01 == I10;              EQ10_01 = EQ01_10
    EQ01_11 = I01 == I11;              EQ11_01 = EQ01_11
    EQ10_11 = I10 == I11;              EQ11_10 = EQ10_11
    NE00_01 = np.logical_not(EQ00_01); NE01_00 = NE00_01
    NE00_10 = np.logical_not(EQ00_10); NE10_00 = NE00_10
    NE00_11 = np.logical_not(EQ00_11); NE11_00 = NE00_11
    NE01_10 = np.logical_not(EQ01_10); NE10_01 = NE01_10
    NE01_11 = np.logical_not(EQ01_11); NE11_01 = NE01_11
    NE10_11 = np.logical_not(EQ10_11); NE11_10 = NE10_11
    #
    # Q1: 1 0
    #     0 0
    Q1_condition = (NE00_01 & NE00_10 & NE00_11).astype(int)
    #     0 1
    #     0 0
    Q1_condition[slice_00] += (NE01_00 & NE01_10 & NE01_11)[slice_01]
    #     0 0
    #     1 0
    Q1_condition[slice_00] += (NE10_00 & NE10_01 & NE10_11)[slice_10]
    #     0 0
    #     0 1
    Q1_condition[slice_00] += (NE11_00 & NE11_01 & NE11_10)[slice_11]
    Q1 = fix(ndimage.sum(Q1_condition, I00, indexes))
    #
    # Q3: 1 1
    #     1 0
    Q3_condition = (EQ00_10 & EQ00_01 & NE00_11).astype(int)
    #     0 1
    #     1 1
    Q3_condition[slice_00] += (NE11_00 & EQ11_10 & EQ11_01)[slice_11]
    #     1 0
    #     1 1
    Q3_condition += (NE00_01 & EQ00_10 & EQ00_11)
    #     1 1
    #     0 1
    Q3_condition += (NE00_10 & EQ00_01 & EQ00_11)
    Q3 = fix(ndimage.sum(Q3_condition, I00, indexes))
    # QD: 1 0
    #     0 1
    QD_condition = (NE00_01 & NE00_10 & EQ00_11).astype(int)
    #     0 1
    #     1 0
    QD_condition[slice_00] += (NE01_00 & NE01_11 & EQ01_10)[slice_01]
    QD  = fix(ndimage.sum(QD_condition, I00, indexes))
    W = (Q1 - Q3 - 2*QD).astype(float)/4.0
    if indexes is None:
        return W[0]
    return W

def _table_lookup(image, table, border_value, iterations = None):
    '''Perform a morphological transform on an image, directed by its neighbors

    image - a binary image
    table - a 512-element table giving the transform of each pixel given
            the values of that pixel and its 8-connected neighbors.
    border_value - the value of pixels beyond the border of the image.
                   This should test as True or False.

    The pixels are numbered like this:

    0 1 2
    3 4 5
    6 7 8
    The index at a pixel is the sum of 2**<pixel-number> for pixels
    that evaluate to true.
    '''
    #
    # Test for a table that never transforms a zero into a one:
    #
    center_is_zero = np.array([(x & 2**4) == 0 for x in range(2**9)])
    use_index_trick = False
    if (not np.any(table[center_is_zero]) and
        (np.issubdtype(image.dtype, bool) or np.issubdtype(image.dtype, int))):
        # Use the index trick
        use_index_trick = True
        invert = False
    elif (np.all(table[~center_is_zero]) and np.issubdtype(image.dtype, bool)):
        # All ones stay ones, invert the table and the image and do the trick
        use_index_trick = True
        invert = True
        image = ~ image
        # table index 0 -> 511 and the output is reversed
        table = ~ table[511-np.arange(512)]
        border_value = not border_value
    if use_index_trick:
        orig_image = image
        index_i, index_j, image = prepare_for_index_lookup(image, border_value)
        index_i, index_j = index_lookup(index_i, index_j,
                                        image, table, iterations)
        image = extract_from_image_lookup(orig_image, index_i, index_j)
        if invert:
            image = ~ image
        return image

    counter = 0
    while counter != iterations:
        counter += 1
        #
        # We accumulate into the indexer to get the index into the table
        # at each point in the image
        #
        if image.shape[0] < 3 or image.shape[1] < 3:
            image = image.astype(bool)
            indexer = np.zeros(image.shape,int)
            indexer[1:,1:]   += image[:-1,:-1] * 2**0
            indexer[1:,:]    += image[:-1,:] * 2**1
            indexer[1:,:-1]  += image[:-1,1:] * 2**2

            indexer[:,1:]    += image[:,:-1] * 2**3
            indexer[:,:]     += image[:,:] * 2**4
            indexer[:,:-1]   += image[:,1:] * 2**5

            indexer[:-1,1:]  += image[1:,:-1] * 2**6
            indexer[:-1,:]   += image[1:,:] * 2**7
            indexer[:-1,:-1] += image[1:,1:] * 2**8
        else:
            indexer = table_lookup_index(np.ascontiguousarray(image,np.uint8))
        if border_value:
            indexer[0,:]   |= 2**0 + 2**1 + 2**2
            indexer[-1,:]  |= 2**6 + 2**7 + 2**8
            indexer[:,0]   |= 2**0 + 2**3 + 2**6
            indexer[:,-1]  |= 2**2 + 2**5 + 2**8
        new_image = table[indexer]
        if np.all(new_image == image):
            break
        image = new_image
    return image

def _pattern_of(index):
    '''Return the pattern represented by an index value'''
    return np.array([[index & 2**0,index & 2**1,index & 2**2],
                     [index & 2**3,index & 2**4,index & 2**5],
                     [index & 2**6,index & 2**7,index & 2**8]], bool)

def _index_of(pattern):
    '''Return the index of a given pattern'''
    return (pattern[0,0] * 2**0 + pattern[0,1] * 2**1 + pattern[0,2] * 2**2 +
            pattern[1,0] * 2**3 + pattern[1,1] * 2**4 + pattern[1,2] * 2**5 +
            pattern[2,0] * 2**6 + pattern[2,1] * 2**7 + pattern[2,2] * 2**8)

def _make_table(value, pattern, care=np.ones((3,3),bool)):
    """Return a table suitable for table_lookup

    value - set all table entries matching 'pattern' to 'value', all others
            to not 'value'
    pattern - a 3x3 boolean array with the pattern to match
    care    - a 3x3 boolean array where each value is true if the pattern
              must match at that position and false if we don't care if
              the pattern matches at that position.
    """
    def fn(index, p,i,j):
        '''Return true if bit position "p" in index matches pattern'''
        return ((((index & 2**p) > 0) == pattern[i,j]) or not care[i,j])
    return np.array([value
                     if (fn(i,0,0,0) and fn(i,1,0,1) and fn(i,2,0,2) and
                         fn(i,3,1,0) and fn(i,4,1,1) and fn(i,5,1,2) and
                         fn(i,6,2,0) and fn(i,7,2,1) and fn(i,8,2,2))
                     else not value
                     for i in range(512)], bool)

'''The table for computing the branchpoints of a skeleton'''
#
# A skeleton will only contain 3-connected pixels if the connected
# pixels are on three separate branches. The operation is subtractive
# so the middle pixel must always be on.
# Removing the middle pixel should create three objects after labeling
# using 4-connectivity.
#
_branchpoints_table = np.array([_pattern_of(index)[1,1] and
                                ndimage.label(_pattern_of(index-16))[1] > 2
                                for index in range(512)])

def branchpoints(image, mask=None):
    """Remove all pixels from an image except for branchpoints

    image - a skeletonized image
    mask -  a mask of pixels excluded from consideration

    1 0 1    ? 0 ?
    0 1 0 -> 0 1 0
    0 1 0    0 ? 0
    """
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _branchpoints_table, False, 1)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

#####################################
#
# Branchings - this is the count of the number of branches that
#              eminate from a pixel. A pixel with neighbors fore
#              and aft has branches fore and aft = 2. An endpoint
#              has one branch. A fork has 3. Finally, there's
#              the quadrabranch which has 4:
#  1 0 1
#  0 1 0 -> 4
#  1 0 1
#####################################

_branchings_table = np.array([0 if (index & 16) == 0
                              else ndimage.label(_pattern_of(index-16))[1]
                              for index in range(512)])

def branchings(image, mask=None):
    """Count the number of branches eminating from each pixel

    image - a binary image
    mask - optional mask of pixels not to consider

    This is the count of the number of branches that
    eminate from a pixel. A pixel with neighbors fore
    and aft has branches fore and aft = 2. An endpoint
    has one branch. A fork has 3. Finally, there's
    the quadrabranch which has 4:
    1 0 1
    0 1 0 -> 4
    1 0 1
    """
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    #
    # Not a binary operation, so we do a convolution with the following
    # kernel to get the indices into the table.
    #
    kernel = np.array([[1,2,4],
                       [8,16,32],
                       [64,128,256]])
    indexer = ndimage.convolve(masked_image.astype(int), kernel,
                             mode='constant').astype(int)
    result = _branchings_table[indexer]
    return result

'''The table for computing binary bridge'''
#
# Either the center is already true or, if you label the pattern,
# there are two unconnected objects in the pattern
#
_bridge_table = np.array([_pattern_of(index)[1,1] or
                          ndimage.label(_pattern_of(index),
                                        np.ones((3,3),bool))[1] > 1
                          for index in range(512)])

def bridge(image, mask=None, iterations = 1):
    '''Fill in pixels that bridge gaps.

    1 0 0    1 0 0
    0 0 0 -> 0 1 0
    0 0 1    0 0 1
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _bridge_table, False, iterations)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

# Keep all pixels (the first _make_table) except for isolated ones
_clean_table = (_make_table(True, np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                            np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) &
                _make_table(False, np.array([[0,0,0],[0,1,0],[0,0,0]],bool)))

def clean(image, mask=None, iterations = 1):
    '''Remove isolated pixels

    0 0 0     0 0 0
    0 1 0 ->  0 0 0
    0 0 0     0 0 0

    Border pixels and pixels adjoining masks are removed unless one valid
    neighbor is true.
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _clean_table, False, iterations)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

# Keep all pixels. Rotate the following pattern 90 degrees four times
# to 4-connect two pixels that are 8-connected
_diag_table = (_make_table(True, np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                           np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) |
               _make_table(True, np.array([[0,1,0],
                                           [1,0,0],
                                           [0,0,0]]),
                           np.array([[1,1,0],
                                     [1,1,0],
                                     [0,0,0]]))|
               _make_table(True, np.array([[0,1,0],
                                           [0,0,1],
                                           [0,0,0]]),
                           np.array([[0,1,1],
                                     [0,1,1],
                                     [0,0,0]]))|
               _make_table(True, np.array([[0,0,0],
                                           [0,0,1],
                                           [0,1,0]]),
                           np.array([[0,0,0],
                                     [0,1,1],
                                     [0,1,1]]))|
               _make_table(True, np.array([[0,0,0],
                                           [1,0,0],
                                           [0,1,0]]),
                           np.array([[0,0,0],
                                     [1,1,0],
                                     [1,1,0]])))

def diag(image, mask=None, iterations=1):
    '''4-connect pixels that are 8-connected

    0 0 0     0 0 ?
    0 0 1 ->  0 1 1
    0 1 0     ? 1 ?

    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _diag_table, False, iterations)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

#
# Endpoints are on and have at most one neighbor.
#
_endpoints_table = np.array([_pattern_of(index)[1,1] and
                             np.sum(_pattern_of(index)) <= 2
                             for index in range(512)])

def endpoints(image, mask=None):
    '''Remove all pixels from an image except for endpoints

    image - a skeletonized image
    mask -  a mask of pixels excluded from consideration

    1 0 0    ? 0 0
    0 1 0 -> 0 1 0
    0 0 0    0 0 0
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _endpoints_table, False, 1)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

# Fill table - keep all ones. Change a zero surrounded by ones to 1
_fill_table = (_make_table(True, np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                           np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) |
               _make_table(True, np.array([[1,1,1],[1,0,1],[1,1,1]],bool)))

def fill(image, mask=None, iterations=1):
    '''Fill isolated black pixels

    1 1 1     1 1 1
    1 0 1 ->  1 1 1
    1 1 1     1 1 1
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = True
    result = _table_lookup(masked_image, _fill_table, True, iterations)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

#Fill4 table - keep if 1. Change a zero with 1's at N-S-E-W to 1.
_fill4_table = (_make_table(True,
                            np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                            np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) |
                _make_table(True,
                            np.array([[1,1,1],[1,0,1],[1,1,1]],bool),
                            np.array([[0,1,0],[1,1,1],[0,1,0]])))
def fill4(image, mask=None, iterations=1):
    '''Fill 4-connected black pixels

    x 1 x     x 1 x
    1 0 1 ->  1 1 1
    x 1 x     x 1 x
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = True
    result = _table_lookup(masked_image, _fill4_table, True, iterations)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

# Hbreak table - keep all ones except for the hbreak case
_hbreak_table = (_make_table(True, np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                             np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) &
                 [i != _index_of(np.array([[1,1,1],[0,1,0],[1,1,1]],bool))
                  for i in range(512)])

def hbreak(image, mask=None, iterations=1):
    '''Remove horizontal breaks

    1 1 1     1 1 1
    0 1 0 ->  0 0 0 (this case only)
    1 1 1     1 1 1
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _hbreak_table, False)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

# Vbreak table - keep all ones except for the vbreak case
_vbreak_table = (_make_table(True, np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                             np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) &
                 [i != _index_of(np.array([[1,0,1],[1,1,1],[1,0,1]],bool))
                  for i in range(512)])

def vbreak(image, mask=None, iterations=1):
    '''Remove horizontal breaks

    1 1 1     1 1 1
    0 1 0 ->  0 0 0 (this case only)
    1 1 1     1 1 1
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _vbreak_table, False)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

_life_table = np.array([np.sum(_pattern_of(i))==3 or
                        (_pattern_of(i)[1,1] and np.sum(_pattern_of(i))==4)
                        for i in range(512)])

def life(image, mask=None, iterations=1):
    return _table_lookup(image, _life_table, False)

# Majority table - a pixel is 1 if the sum of it and its neighbors is > 4
_majority_table = np.array([np.sum(_pattern_of(i))>4 for i in range(512)])
def majority(image, mask=None, iterations=1):
    '''A pixel takes the value of the majority of its neighbors

    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _majority_table, False, iterations)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

# Remove table - a pixel is changed from 1 to 0 if all of its 4-connected
# neighbors are 1
_remove_table = (_make_table(True,
                             np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                             np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) &
                 _make_table(False,
                             np.array([[0,1,0],
                                       [1,1,1],
                                       [0,1,0]],bool),
                             np.array([[0,1,0],
                                       [1,1,1],
                                       [0,1,0]],bool)))

def remove(image, mask=None, iterations=1):
    '''Turn 1 pixels to 0 if their 4-connected neighbors are all 0

    ? 1 ?     ? 1 ?
    1 1 1  -> 1 0 1
    ? 1 ?     ? 1 ?
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _remove_table, False)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

# A spur pixel has only one neighbor
#
# We have to remove 1/2 of the spur pixels in pass 1 and
# 1/2 in pass 2 in order to keep from removing both members
# of a 2-pixel object. So there are two spur-tables.
#
# spur_table_1 removes if the only neighbor is in the top row or left.
#
_spur_table_1 = np.array([(np.sum(_pattern_of(i)) != 2 or
                           ((i & (2**0 + 2**1 + 2**2 + 2**3)) == 0)) and
                          (i & 2**4)
                          for i in range(512)],bool)
#
# spur_table_2 removes if the only neighbor is in the bottom row or right.
#
_spur_table_2 = np.array([(np.sum(_pattern_of(i)) != 2 or
                           ((i & (2**0 + 2**1 + 2**2 + 2**3)) != 0)) and
                          (i & 2**4)
                          for i in range(512)],bool)

def spur(image, mask=None, iterations=1):
    '''Remove spur pixels from an image

    0 0 0    0 0 0
    0 1 0 -> 0 0 0
    0 0 1    0 0 ?
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    index_i, index_j, masked_image = prepare_for_index_lookup(masked_image,
                                                              False)
    if iterations == None:
        iterations = len(index_i)
    for i in range(iterations):
        for table in (_spur_table_1, _spur_table_2):
            index_i, index_j = index_lookup(index_i, index_j,
                                            masked_image, table, 1)
    masked_image = extract_from_image_lookup(image, index_i, index_j)
    if not mask is None:
        masked_image[~mask] = image[~mask]
    return masked_image

#
# The thicken table turns pixels on if they have a neighbor that's on and
# if adding the pixel does not connect any neighbors
#
_thicken_table = np.array([ndimage.label(_pattern_of(i), _eight_connect)[1] ==
                           ndimage.label(_pattern_of(i | 16), _eight_connect)[1] or
                           ((i & 16) != 0) for i in range(512)])

def thicken(image, mask=None, iterations=1):
    '''Thicken the objects in an image where doing so does not connect them

    0 0 0    ? ? ?
    0 0 0 -> ? 1 ?
    0 0 1    ? ? ?

    1 0 0    ? ? ?
    0 0 0 -> ? 0 ?
    0 0 1    ? ? ?
    '''
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    result = _table_lookup(masked_image, _thicken_table, False, iterations)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

# Thinning tables based on
# algorithm # 1 described in Guo, "Parallel Thinning with Two
# Subiteration Algorithms", Communications of the ACM, Vol 32 #3
# page 359.
#
# Neighborhood pixels are numbered like this:
# p1 p2 p3
# p8    p4
# p7 p6 p5
#
# A pixel changes from 1 to 0 if
#
# 1) labeling its 8-neighborhood finds exactly 1 object
# 2) min(N1,N2) is either 2 or 3 where
#    N1 = (p1 or p2) + (p3 or p4) + (p5 or p6) + (p7 or p8)
#    N2 = (p2 or p3) + (p4 or p5) + (p6 or p7) + (p8 or p1)
# 3) for pass 1: (p2 or p3 or not p5) and p4 is false
#    for pass 2: (p6 or p7 or not p1) and p8 is false
#
#
_thin_table = None

def thin(image, mask=None, iterations=1):
    '''Thin an image to lines, preserving Euler number

    Implements thinning as described in algorithm # 1 from
    Guo, "Parallel Thinning with Two Subiteration Algorithms",
    Communications of the ACM, Vol 32 #3 page 359.
    '''
    global _thin_table, _eight_connect
    if _thin_table is None:
        _thin_table = np.zeros((2,512),bool)
        for i in range(512):
            if (i & 16) == 0:
                # All zeros -> 0
                continue
            pat = _pattern_of(i & ~ 16)
            ipat = pat.astype(int)
            if ndimage.label(pat, _eight_connect)[1] != 1:
                _thin_table[:,i] = True
                continue
            n1 = ((ipat[0,0] or ipat[0,1]) + (ipat[0,2] or ipat[1,2])+
                  (ipat[2,2] or ipat[2,1]) + (ipat[2,0] or ipat[1,0]))
            n2 = ((ipat[0,1] or ipat[0,2]) + (ipat[1,2] or ipat[2,2])+
                  (ipat[2,1] or ipat[2,0]) + (ipat[1,0] or ipat[0,0]))
            if min(n1,n2) not in (2,3):
                _thin_table[:,i] = True
                continue
            _thin_table[0,i] = ((pat[0,1] or pat[0,2] or not pat[2,2]) and
                                pat[1,2])
            _thin_table[1,i] = ((pat[2,1] or pat[2,0] or not pat[0,0]) and
                                pat[1,0])
    if mask is None:
        masked_image = image.copy()
    else:
        masked_image = image.copy()
        masked_image[~mask] = False
    index_i, index_j, masked_image = prepare_for_index_lookup(masked_image, False)
    if iterations is None:
        iterations = len(index_i)
    for i in range(iterations):
        hit_count = len(index_i)
        for j in range(2):
            index_i, index_j, = index_lookup(index_i, index_j,
                                             masked_image,
                                             _thin_table[j], 1)
        if hit_count == len(index_i):
            break
    masked_image = extract_from_image_lookup(image, index_i, index_j)
    if not mask is None:
        masked_image[~mask] = masked_image[~mask]
    return masked_image

def find_neighbors(labels):
    '''Find the set of objects that touch each object in a labels matrix

    Construct a "list", per-object, of the objects 8-connected adjacent
    to that object.
    Returns three 1-d arrays:
    * array of #'s of neighbors per object
    * array of indexes per object to that object's list of neighbors
    * array holding the neighbors.

    For instance, say 1 touches 2 and 3 and nobody touches 4. The arrays are:
    [ 2, 1, 1, 0], [ 0, 2, 3, 4], [ 2, 3, 1, 1]
    '''
    max_label = np.max(labels)
    # Make a labels matrix with zeros around the edges so we can do index
    # offsets without worrying.
    #
    new_labels = np.zeros(np.array(labels.shape)+2, labels.dtype)
    new_labels[1:-1,1:-1] = labels
    labels = new_labels
    # Only consider the points that are next to others
    adjacent_mask = adjacent(labels)
    adjacent_i, adjacent_j = np.argwhere(adjacent_mask).transpose()
    # Get matching vectors of labels and neighbor labels for the 8
    # compass directions.
    count = len(adjacent_i)
    if count == 0:
        return (np.zeros(max_label, int),
                np.zeros(max_label,int),
                np.zeros(0,int))
    # The following bizarre construct does the following:
    # labels[adjacent_i, adjacent_j] looks up the label for each pixel
    # [...]*8 creates a list of 8 references to it
    # np.hstack concatenates, giving 8 repeats of the list
    v_label = np.hstack([labels[adjacent_i, adjacent_j]]*8)
    v_neighbor = np.zeros(count*8,int)
    index = 0
    for i,j in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
        v_neighbor[index:index+count] = labels[adjacent_i+i, adjacent_j+j]
        index += count
    #
    # sort by label and neighbor
    #
    sort_order = np.lexsort((v_neighbor,v_label))
    v_label = v_label[sort_order]
    v_neighbor = v_neighbor[sort_order]
    #
    # eliminate duplicates by comparing each element after the first one
    # to its previous
    #
    first_occurrence = np.ones(len(v_label), bool)
    first_occurrence[1:] = ((v_label[1:] != v_label[:-1]) |
                            (v_neighbor[1:] != v_neighbor[:-1]))
    v_label = v_label[first_occurrence]
    v_neighbor = v_neighbor[first_occurrence]
    #
    # eliminate neighbor = self and neighbor = background
    #
    to_remove = (v_label == v_neighbor) | (v_neighbor == 0)
    v_label = v_label[~ to_remove]
    v_neighbor = v_neighbor[~ to_remove]
    #
    # The count of # of neighbors
    #
    v_count = _fixup_scipy_ndimage_result(ndimage.sum(np.ones(v_label.shape),
                                                   v_label,
                                                   np.arange(max_label,dtype=np.int32)+1))
    v_count = v_count.astype(int)
    #
    # The index into v_neighbor
    #
    v_index = np.cumsum(v_count)
    v_index[1:] = v_index[:-1]
    v_index[0] = 0
    return (v_count, v_index, v_neighbor)

def distance_color_labels(labels):
    '''Recolor a labels matrix so that adjacent labels have distant numbers

    '''
    #
    # Color labels so adjacent ones are most distant
    #
    colors = color_labels(labels, True)
    #
    # Order pixels by color, then label #
    #
    rlabels =  labels.ravel()
    order = np.lexsort((rlabels, colors.ravel()))
    #
    # Construct color indices with the cumsum trick:
    # cumsum([0,0,1,0,1]) = [0,0,1,1,2]
    # and copy back into the color array, using the order.
    #
    different = np.hstack([[False], rlabels[order[1:]] != rlabels[order[:-1]]])
    colors.ravel()[order] = np.cumsum(different)
    return colors.astype(labels.dtype)

def color_labels(labels, distance_transform = False):
    '''Color a labels matrix so that no adjacent labels have the same color

    distance_transform - if true, distance transform the labels to find out
         which objects are closest to each other.

    Create a label coloring matrix which assigns a color (1-n) to each pixel
    in the labels matrix such that all pixels similarly labeled are similarly
    colored and so that no similiarly colored, 8-connected pixels have
    different labels.

    You can use this function to partition the labels matrix into groups
    of objects that are not touching; you can then operate on masks
    and be assured that the pixels from one object won't interfere with
    pixels in another.

    returns the color matrix
    '''
    if distance_transform:
        i,j = ndimage.distance_transform_edt(labels == 0, return_distances=False,
                                           return_indices = True)
        dt_labels = labels[i,j]
    else:
        dt_labels = labels
    # Get the neighbors for each object
    v_count, v_index, v_neighbor = find_neighbors(dt_labels)
    # Quickly get rid of labels with no neighbors. Greedily assign
    # all of these a color of 1
    v_color = np.zeros(len(v_count)+1,int) # the color per object - zero is uncolored
    zero_count = (v_count==0)
    if np.all(zero_count):
        # can assign all objects the same color
        return (labels!=0).astype(int)
    v_color[1:][zero_count] = 1
    v_count = v_count[~zero_count]
    v_index = v_index[~zero_count]
    v_label = np.argwhere(~zero_count).transpose()[0]+1
    # If you process the most connected labels first and use a greedy
    # algorithm to preferentially assign a label to an existing color,
    # you'll get a coloring that uses 1+max(connections) at most.
    #
    # Welsh, "An upper bound for the chromatic number of a graph and
    # its application to timetabling problems", The Computer Journal, 10(1)
    # p 85 (1967)
    #
    sort_order = np.lexsort([-v_count])
    v_count = v_count[sort_order]
    v_index = v_index[sort_order]
    v_label = v_label[sort_order]
    for i in range(len(v_count)):
        neighbors = v_neighbor[v_index[i]:v_index[i]+v_count[i]]
        colors = np.unique(v_color[neighbors])
        if colors[0] == 0:
            if len(colors) == 1:
                # only one color and it's zero. All neighbors are unlabeled
                v_color[v_label[i]] = 1
                continue
            else:
                colors = colors[1:]
        # The colors of neighbors will be ordered, so there are two cases:
        # * all colors up to X appear - colors == np.arange(1,len(colors)+1)
        # * some color is missing - the color after the first missing will
        #   be mislabeled: colors[i] != np.arange(1, len(colors)+1)
        crange = np.arange(1,len(colors)+1)
        misses = crange[colors != crange]
        if len(misses):
            color = misses[0]
        else:
            color = len(colors)+1
        v_color[v_label[i]] = color
    return v_color[labels]

def skeletonize(image, mask=None):
    '''Skeletonize the image

    Take the distance transform.
    Order the 1 points by the distance transform.
    Remove a point if it has more than 1 neighbor and if removing it
    does not change the Euler number.
    '''
    global _eight_connect
    if mask is None:
        masked_image = image
    else:
        masked_image = image.astype(bool).copy()
        masked_image[~mask] = False
    #
    # Lookup table - start with only positive pixels.
    # Keep if # pixels in neighborhood is 2 or less
    # Keep if removing the pixel results in a different connectivity
    #
    table = (_make_table(True,np.array([[0,0,0],[0,1,0],[0,0,0]],bool),
                         np.array([[0,0,0],[0,1,0],[0,0,0]],bool)) &
             (np.array([ndimage.label(_pattern_of(index), _eight_connect)[1] !=
                        ndimage.label(_pattern_of(index & ~ 2**4),
                                      _eight_connect)[1]
                        for index in range(512) ]) |
              np.array([np.sum(_pattern_of(index))<3 for index in range(512)])))

    distance = ndimage.distance_transform_edt(masked_image)
    #
    # The processing order along the edge is critical to the shape of the
    # resulting skeleton: if you process a corner first, that corner will
    # be eroded and the skeleton will miss the arm from that corner. Pixels
    # with fewer neighbors are more "cornery" and should be processed last.
    #
    cornerness_table = np.array([9-np.sum(_pattern_of(index))
                                 for index in range(512)])
    corner_score = _table_lookup(masked_image, cornerness_table, False,1)
    i,j = np.mgrid[0:image.shape[0],0:image.shape[1]]
    result=masked_image.copy()
    distance = distance[result]
    i = np.ascontiguousarray(i[result],np.int32)
    j = np.ascontiguousarray(j[result],np.int32)
    result=np.ascontiguousarray(result,np.uint8)
    #
    # We use a random # for tiebreaking. Assign each pixel in the image a
    # predictable, random # so that masking doesn't affect arbitrary choices
    # of skeletons
    #
    np.random.seed(0)
    tiebreaker=np.random.permutation(np.arange(np.product(masked_image.shape)))
    tiebreaker.shape=masked_image.shape
    order = np.lexsort((tiebreaker[masked_image],
                        corner_score[masked_image],
                        distance))
    order = np.ascontiguousarray(order, np.int32)
    table = np.ascontiguousarray(table, np.uint8)
    skeletonize_loop(result, i, j, order, table)

    result = result.astype(bool)
    if not mask is None:
        result[~mask] = image[~mask]
    return result

def skeletonize_labels(labels):
    '''Skeletonize a labels matrix'''
    #
    # The trick here is to separate touching labels by coloring the
    # labels matrix and then processing each color separately
    #
    colors = color_labels(labels)
    max_color = np.max(colors)
    if max_color == 0:
        return labels
    result = np.zeros(labels.shape, labels.dtype)
    for i in range(1,max_color+1):
        mask = skeletonize(colors==i)
        result[mask] = labels[mask]
    return result

def label_skeleton(skeleton):
    """Label a skeleton so that each edge has a unique label

    This operation produces a labels matrix where each edge between
    two branchpoints has a different label. If the skeleton has been
    properly eroded, there are three kinds of points:
    1) point adjacent to 0 or 1 other points = end of edge
    2) point adjacent to two other points = in middle of edge
    3) point adjacent to more than two other points = at end of edge
            connecting to another edge
    4) a branchpoint

    We do all connected components here where components are 8-connected
    but a point in category 3 can't connect to another point in category 3.

    Returns the labels matrix and the count as a tuple
    """
    bpts = branchpoints(skeleton)
    #
    # Count the # of neighbors per point
    #
    neighbors = ndimage.convolve(skeleton.astype(int), np.ones((3,3),int),
                               mode='constant').astype(int)
    neighbors[~skeleton] = 0
    neighbors[skeleton] -= 1
    #
    # Find the i/j coordinates of the relevant points
    #
    i,j = np.mgrid[0:skeleton.shape[0], 0:skeleton.shape[1]]
    skeleton_minus_bpts = skeleton & ~ bpts
    si = i[skeleton_minus_bpts]
    sj = j[skeleton_minus_bpts]
    bi = i[bpts]
    bj = j[bpts]
    i = np.hstack((bi, si))
    j = np.hstack((bj, sj))
    b_vnum = np.arange(len(bi))
    s_vnum = np.arange(len(si)) + len(bi)
    all_vnum = np.hstack((b_vnum, s_vnum))
    vertex_numbers=np.zeros(skeleton.shape, int)
    vertex_numbers[i,j] = all_vnum
    #
    # src and dest are the vertices linked by edges. Their values are the
    # vertex numbers. First, link every vertex to itself
    #
    src = all_vnum
    dest = all_vnum
    #
    # Now, for the non-branchpoints, link to all 8-connected neighbors
    # while obeying the rules
    #
    for ioff, joff in ((-1,-1), (-1,0), (-1,1),
                       ( 0,-1),         ( 0,1),
                       ( 1,-1), ( 1,0), ( 1,1)):
        consider = np.ones(len(si), bool)
        if ioff == -1:
            consider = si > 0
        elif ioff == 1:
            consider = si < skeleton.shape[0] - 1
        if joff == -1:
            consider = consider & (sj > 0)
        elif joff == 1:
            consider = consider & (sj < skeleton.shape[1] - 1)
        #
        # Forge a link if the offset point is in the skeleton
        #
        ci = si[consider]
        cj = sj[consider]
        link = (skeleton_minus_bpts[ci+ioff, cj+joff] &
                ((neighbors[ci,cj] < 3) | (neighbors[ci+ioff, cj+joff] < 3)))
        ci = ci[link]
        cj = cj[link]
        src = np.hstack((src, vertex_numbers[ci, cj]))
        dest = np.hstack((dest, vertex_numbers[ci+ioff, cj+joff]))

    labeling = _cp_all_connected_components(src, dest)
    vertex_numbers[i,j] = labeling + 1
    return (vertex_numbers,
            0 if len(labeling) == 0 else int(np.max(labeling)) + 1)

def distance_to_edge(labels):
    '''Compute the distance of a pixel to the edge of its object

    labels - a labels matrix

    returns a matrix of distances
    '''
    colors = color_labels(labels)
    max_color = np.max(colors)
    result = np.zeros(labels.shape)
    if max_color == 0:
        return result

    for i in range(1, max_color+1):
        mask = (colors==i)
        result[mask] = ndimage.distance_transform_edt(mask)[mask]
    return result

def regional_maximum(image, mask = None, structure=None, ties_are_ok=False):
    '''Return a binary mask containing only points that are regional maxima

    image     - image to be transformed
    mask      - mask of relevant pixels
    structure - binary structure giving the neighborhood and connectivity
                in which to search for maxima. Default is 8-connected.
    ties_are_ok - if this is true, then adjacent points of the same magnitude
                  are rated as maxima.

    Find locations for which all neighbors as defined by the structure have
    lower values. The algorithm selects only one of a set of adjacent locations
    with identical values, first using a distance transform to find the
    innermost location, then, among equals, selected randomly.

    A location cannot be a local maximum if it is touching the edge or a
    masked pixel.
    '''
    global _eight_connect
    if not ties_are_ok:
        #
        # Get an an initial pass with the ties.
        #
        result = regional_maximum(image, mask, structure, True)
        if not np.any(result):
            return result
        distance = ndimage.distance_transform_edt(result)
        #
        # Rank-order the distances and then add a randomizing factor
        # to break ties for pixels equidistant from the background.
        # Pick the best value within a contiguous region
        #
        labels, label_count = ndimage.label(result, _eight_connect)
        np.random.seed(0)
        ro_distance = rank_order(distance)[0].astype(float)
        count = np.product(ro_distance.shape)
        ro_distance.flat += (np.random.permutation(count).astype(float) /
                             float(count))
        positions = ndimage.maximum_position(ro_distance, labels,
                                           np.arange(label_count)+1)
        positions = np.array(positions, np.uint32)
        result = np.zeros(image.shape, bool)
        if positions.ndim == 1:
            result[positions[0],positions[1]] = True
        else:
            result[positions[:,0],positions[:,1]] = True
        return result
    result = np.ones(image.shape,bool)
    if structure == None:
        structure = ndimage.generate_binary_structure(image.ndim, image.ndim)
    #
    # The edges of the image are losers because they are touching undefined
    # points. Construct a big mask that represents the edges.
    #
    big_mask = np.zeros(np.array(image.shape) + np.array(structure.shape), bool)
    structure_half_shape = np.array(structure.shape)/2
    big_mask[structure_half_shape[0]:structure_half_shape[0]+image.shape[0],
             structure_half_shape[1]:structure_half_shape[1]+image.shape[1]]=\
        mask if not mask is None else True
    for i in range(structure.shape[0]):
        off_i = i-structure_half_shape[0]
        for j in range(structure.shape[1]):
            if i == structure_half_shape[0] and j == structure_half_shape[1]:
                continue
            off_j = j-structure_half_shape[1]
            if structure[i,j]:
                result = np.logical_and(result, big_mask[i:i+image.shape[0],
                                                         j:j+image.shape[1]])
                #
                # Get the boundaries of the source image and the offset
                # image so we can make similarly shaped, but offset slices
                #
                src_i_min = max(0,-off_i)
                src_i_max = min(image.shape[0], image.shape[0]-off_i)
                off_i_min = max(0,off_i)
                off_i_max = min(image.shape[0], image.shape[0]+off_i)
                src_j_min = max(0,-off_j)
                src_j_max = min(image.shape[1], image.shape[1]-off_j)
                off_j_min = max(0,off_j)
                off_j_max = min(image.shape[1], image.shape[1]+off_j)
                min_mask = (image[src_i_min:src_i_max,
                                  src_j_min:src_j_max] <
                            image[off_i_min:off_i_max,
                                  off_j_min:off_j_max])
                result[src_i_min:src_i_max,
                       src_j_min:src_j_max][min_mask] = False
    return result

def _cp_all_connected_components(i, j):
    """Associate each label in i with a component #

    This function finds all connected components given an array of
    associations between labels i and j using a depth-first search.

    i & j give the edges of the graph. The first step of the algorithm makes
    bidirectional edges, (i->j and j<-i), so it's best to only send the
    edges in one direction (although the algorithm can withstand duplicates).

    returns a label for each vertex up to the maximum named vertex in i.
    """
    if len(i) == 0:
        return i
    i1 = np.hstack((i,j))
    j1 = np.hstack((j,i))
    order = np.lexsort((j1,i1))
    i=np.ascontiguousarray(i1[order],np.uint32)
    j=np.ascontiguousarray(j1[order],np.uint32)
    #
    # Get indexes and counts of edges per vertex
    #
    counts = np.ascontiguousarray(np.bincount(i.astype(int)),np.uint32)
    indexes = np.ascontiguousarray(np.cumsum(counts)-counts,np.uint32)
    #
    # This stores the lowest index # during the algorithm - the first
    # vertex to be labeled in a connected component.
    #
    labels = np.zeros(len(counts), np.uint32)
    _all_connected_components(i,j,indexes,counts,labels)
    return labels

def is_local_maximum(image, labels, footprint):
    '''Return a boolean array of points that are local maxima

    image - intensity image
    labels - find maxima only within labels. Zero is reserved for background.
    footprint - binary mask indicating the neighborhood to be examined
                must be a matrix with odd dimensions, center is taken to
                be the point in question.
    '''
    assert((np.all(footprint.shape) & 1) == 1)
    footprint = (footprint != 0)
    footprint_extent = (np.array(footprint.shape)-1) / 2
    if np.all(footprint_extent == 0):
        return labels > 0
    result = (labels > 0).copy()
    #
    # Create a labels matrix with zeros at the borders that might be
    # hit by the footprint.
    #
    big_labels = np.zeros(np.array(labels.shape) + footprint_extent*2,
                          labels.dtype)
    big_labels[[slice(fe,-fe) for fe in footprint_extent]] = labels
    #
    # Find the relative indexes of each footprint element
    #
    image_strides = np.array(image.strides) / image.dtype.itemsize
    big_strides = np.array(big_labels.strides) / big_labels.dtype.itemsize
    result_strides = np.array(result.strides) / result.dtype.itemsize
    footprint_offsets = np.mgrid[[slice(-fe,fe+1) for fe in footprint_extent]]

    fp_image_offsets = np.sum(image_strides[:, np.newaxis] *
                              footprint_offsets[:, footprint], 0)
    fp_big_offsets = np.sum(big_strides[:, np.newaxis] *
                            footprint_offsets[:, footprint], 0)
    #
    # Get the index of each labeled pixel in the image and big_labels arrays
    #
    indexes = np.mgrid[[slice(0,x) for x in labels.shape]][:, labels > 0]
    image_indexes = np.sum(image_strides[:, np.newaxis] * indexes, 0)
    big_indexes = np.sum(big_strides[:, np.newaxis] *
                         (indexes + footprint_extent[:, np.newaxis]), 0)
    result_indexes = np.sum(result_strides[:, np.newaxis] * indexes, 0)
    #
    # Now operate on the raveled images
    #
    big_labels_raveled = big_labels.ravel()
    image_raveled = image.ravel()
    result_raveled = result.ravel()
    #
    # A hit is a hit if the label at the offset matches the label at the pixel
    # and if the intensity at the pixel is greater or equal to the intensity
    # at the offset.
    #
    for fp_image_offset, fp_big_offset in zip(fp_image_offsets, fp_big_offsets):
        same_label = (big_labels_raveled[big_indexes + fp_big_offset] ==
                      big_labels_raveled[big_indexes])
        less_than = (image_raveled[image_indexes[same_label]] <
                     image_raveled[image_indexes[same_label]+ fp_image_offset])
        result_raveled[result_indexes[same_label][less_than]] = False

    return result

def angular_distribution(labels, resolution=100, weights=None):
    '''For each object in labels, compute the angular distribution
    around the centers of mass.  Returns an i x j matrix, where i is
    the number of objects in the label matrix, and j is the resolution
    of the distribution (default 100), mapped from -pi to pi.

    Optionally, the distributions can be weighted by pixel.

    The algorithm approximates the angular width of pixels relative to
    the object centers, in an attempt to be accurate for small
    objects.

    The ChordRatio of an object can be approximated by
    >>> angdist = angular_distribution(labels, resolution)
    >>> angdist2 = angdist[:, :resolution/2] + angdist[:, resolution/2] # map to widths, rather than radii
    >>> chord_ratio = np.sqrt(angdist2.max(axis=1) / angdist2.min(axis=1)) # sqrt because these are sectors, not triangles
    '''
    if weights is None:
        weights = np.ones(labels.shape)
    maxlabel = labels.max()
    ci, cj = centers_of_labels(labels)
    j, i = np.meshgrid(np.arange(labels.shape[0]), np.arange(labels.shape[1]))
    # compute deltas from pixels to object centroids, and mask to labels
    di = i[labels > 0] - ci[labels[labels > 0] - 1]
    dj = j[labels > 0] - cj[labels[labels > 0] - 1]
    weights = weights[labels > 0]
    labels = labels[labels > 0]
    # find angles, and angular width of pixels
    angle = np.arctan2(di, dj)
    # Use pixels of width 2 to get some smoothing
    width = np.arctan(1.0 / np.sqrt(di**2 + dj**2 + np.finfo(float).eps))
    # create an onset/offset array of size 3 * resolution
    lo = np.clip((angle - width) * resolution / (2 * np.pi), -resolution, 2 * resolution).astype(int) + resolution
    hi = np.clip((angle + width) * resolution / (2 * np.pi), -resolution, 2 * resolution).astype(int) + resolution
    # make sure every pixel counts at least once
    hi[lo == hi] += 1
    # normalize weights by their angular width (adding a bit to avoid 0 / 0)
    weights /= (hi - lo)
    onset = scipy.sparse.coo_matrix((weights, (labels - 1, lo)), (maxlabel, 3 * resolution)).toarray()
    offset = scipy.sparse.coo_matrix((weights, (labels - 1, hi)), (maxlabel, 3 * resolution)).toarray()
    # sum onset/offset to get actual distribution
    onoff = onset - offset
    dist = np.cumsum(onoff, axis=1)
    dist = dist[:, :resolution] + dist[:, resolution:2*resolution] + dist[:, 2*resolution:]
    return dist

def feret_diameter(chulls, counts, indexes):
    '''Return the minimum and maximum Feret diameter for each object

    This function takes the convex hull data, as generated by convex_hull
    and returns the minimum and maximum Feret diameter for each convex hull.

    chulls    - an n x 3 matrix giving the label #, the i coordinate and the
                j coordinate for each vertex in the convex hull
    counts    - the number of points in each convex hull

    returns two arrays, the minimum and maximum diameter per object.
    '''
    #
    # Method taken from Alsuwaiyel, "Algorithms, Design Techniques and
    # Analysis", World Scientific, 2003
    #
    # Given a vertex, p1 on the convex hull composed of points p1 to pm,
    # there is a line, pm <-> p1. Going from p(m-1) to p2, find the
    # distance from the point to the line until the next point in the
    # series has a lesser distance. This point forms an antipodal pair
    # with p1. Now take the line from p1 to p2 and continue to add points
    # as long as the point has a greater distance from the line than its
    # successor. The final point forms an antipodal point with both p1
    # and its succesor vertex.
    #
    # Note: as far as I (leek) can figure out, there is a bug here for
    # the case of a square:
    #
    #  0     1
    #
    #
    #  3     2
    #
    #  0 - 1 is generated first, then 0 - 2 and 0 - 3
    #  Then 1 - 3 is generated, missing 1 - 2. The author is only interested
    #  in max Feret diameter as are we.
    #
    # The minimum Feret diameter always has a parallel line of support
    # that runs through two antipodal pairs and the second that either
    # runs through an antipodal pair or a vertex.
    #
    if len(counts) == 0:
        return np.zeros(0)
    save_counts = counts
    counts = np.atleast_1d(counts)
    indexes = np.atleast_1d(indexes)
    chull_idx = np.hstack(([0], np.cumsum(counts[:-1])))
    save_chull_idx = chull_idx
    chull_pts = chulls[:,1:].astype(float)
    #
    # A list of antipode arrays
    #
    antipodes = []
    #
    # Get rid of degenerate cases
    #
    # A single point is its own antipode
    #
    antipodes.append(np.column_stack([chull_idx[counts==1]] * 2))
    #
    # A line has it's two points as antipodes of each other.
    #
    antipodes.append(np.column_stack([chull_idx[counts==2],
                                      chull_idx[counts==2] + 1]))
    chull_idx = chull_idx[counts > 2]
    counts = counts[counts > 2]
    if len(counts) > 0:
        #
        # Calculate distances from every vertex other than pm and p1
        # to the line between pm and p1
        #
        pm_idx = (chull_idx + counts - 1)
        pm = chull_pts[pm_idx,:]
        p1 = chull_pts[chull_idx,:]
        # There are counts - 2 vertices to be examined
        vvv = Indexes([counts-2])
        v_idx = chull_idx[vvv.rev_idx] + vvv.idx[0] + 1
        distances = distance2_to_line(chull_pts[v_idx],
                                      pm[vvv.rev_idx],
                                      p1[vvv.rev_idx])
        #
        # Find the largest distance, settling ties by picking the
        # one with the largest within-object index.
        #
        order = np.lexsort((vvv.idx[0], -distances, vvv.rev_idx))
        antipode_idx = order[vvv.fwd_idx] + 1 - vvv.fwd_idx
        vertex_idx = np.zeros(len(antipode_idx), int)
        while len(chull_idx) > 0:
            #
            # Add the current antipode / vertex pair
            #
            antipodes.append(np.column_stack((chull_idx + vertex_idx,
                                              chull_idx + antipode_idx)))
            if len(chull_idx) == 0:
                break
            #
            # Get the distance from the line to the current antipode
            #
            dc = distance2_to_line(chull_pts[chull_idx + antipode_idx, :],
                                   chull_pts[chull_idx + vertex_idx, :],
                                   chull_pts[chull_idx +vertex_idx + 1, :])
            #
            # Get the distance from the line to the next antipode
            #
            next_antipode_idx = antipode_idx + 1
            next_antipode_idx[next_antipode_idx == counts] = 0
            dn = distance2_to_line(chull_pts[chull_idx + next_antipode_idx, :],
                                   chull_pts[chull_idx + vertex_idx, :],
                                   chull_pts[chull_idx + vertex_idx + 1, :])
            #
            # If the distance to the next candidate is larger, advance
            #
            advance_antipode = dc <= dn
            antipode_idx[advance_antipode] += 1
            #
            # Otherwise, move to the next vertex.
            #
            vertex_idx[~ advance_antipode] += 1
            #
            # Get rid of completed convex hulls
            #
            to_keep = (antipode_idx < counts) & (vertex_idx != antipode_idx)
            antipode_idx = antipode_idx[to_keep]
            vertex_idx = vertex_idx[to_keep]
            chull_idx = chull_idx[to_keep]
            counts = counts[to_keep]
    antipodes = np.vstack(antipodes)
    l = chulls[antipodes[:,0], 0]

    pt1 = chulls[antipodes[:,0],1:]
    pt2 = chulls[antipodes[:,1],1:]
    distances = np.sum((pt1 - pt2) **2, 1)

    
    ## best = np.array(ndimage.maximum_position(distances, l, indexes)).ravel().astype(int)
    ## dt = np.dtype([("label",int,1),("distance",float,1),("pt1",int,2),("pt2",int,2)])
    ## choices = np.array(
    ##     zip(indexes, np.sqrt(distances[best]), pt1[best,:], pt2[best,:]),
    ##     dtype = dt)
    ## logger.debug("Feret diameter results:\n" + repr(choices))


    max_distance = np.sqrt(_fixup_scipy_ndimage_result(ndimage.maximum(distances, l, indexes)))
    #
    # For the minimum distance, we have to take the distance from each
    # antipode to any line between alternate successive antipodes.
    #
    counts = save_counts
    chull_idx = save_chull_idx
    indexer = Indexes(counts)
    # Get rid of degenerates
    #
    degenerates = antipodes[:,0] == antipodes[:,1]
    if np.all(degenerates):
        return np.zeros(len(indexes)), max_distance
    l = l[~degenerates]
    antipodes = antipodes[~degenerates]
    # We duplicate the list so each antipode appears as both a first and second
    #
    l = np.hstack((l,l))
    antipodes = np.vstack((antipodes, antipodes[:,::-1]))
    v2_idx = indexer.idx[0][antipodes[:,1]]
    #
    # Vertex number zero can be the vertex after the last one. We therefore
    # add a second vertex of "count" for each one where the second vertex is zero.
    #
    extras = antipodes[v2_idx == 0,:]
    antipodes = np.vstack([antipodes, extras])
    l = np.hstack((l, indexes[indexer.rev_idx[extras[:,0]]]))
    v2_idx = np.hstack((v2_idx, indexer.counts[0][indexer.rev_idx[extras[:,1]]]))
    #
    # We sort by label, first and second to order them for the search.
    #
    order = np.lexsort((v2_idx, antipodes[:,0], l))
    antipodes = antipodes[order,:]
    l = l[order]
    v2_idx = v2_idx[order]
    #
    # Now, we only want antipode pairs where:
    # * the label is the same as it's successor
    # * the first antipode is the same as it's successor
    # * the second antipode is one less than it's successor
    #
    v1_idx = indexer.idx[0][antipodes[:,0]]
    good = np.hstack(((l[1:] == l[:-1]) &
                      (v1_idx[1:] == v1_idx[:-1]) &
                      (v2_idx[1:] == v2_idx[:-1]+1), [False]))
    if not np.any(good):
        return np.zeros(len(indexes)), max_distance
    antipodes = antipodes[good,:]
    l = l[good]
    v2_idx = v2_idx[good]
    v = chull_pts[antipodes[:,0],:]
    l0 = chull_pts[antipodes[:,1],:]
    #
    # The index of the second point in the line has to be done modulo
    #
    l1_idx = antipodes[:,1] + 1
    needs_modulo = v2_idx == indexer.counts[0][indexer.rev_idx[antipodes[:,1]]] - 1
    l1_idx[needs_modulo] -= indexer.counts[0][indexer.rev_idx[antipodes[needs_modulo,1]]]
    l1 = chull_pts[l1_idx,:]
    #
    # Compute the distances
    #
    distances = distance2_to_line(v, l0, l1)
    min_distance = np.sqrt(_fixup_scipy_ndimage_result(ndimage.minimum(distances, l, indexes)))
    min_distance[np.isnan(min_distance)] = 0
    return min_distance, max_distance
