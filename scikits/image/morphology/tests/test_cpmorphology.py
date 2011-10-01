"""test_cpmorphology - test the functions in cellprofiler.cpmath.cpmorphology

Originally part of CellProfiler, code licensed under both GPL and BSD licenses.
Website: http://www.cellprofiler.org

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Original author: Lee Kamentsky
"""

import numpy as np
import scipy.ndimage as ndi

import scikits.image.morphology.cpmorphology as morph
from scikits.image.morphology.cpmorphology import _fixup_scipy_ndimage_result as fix

from numpy.testing import assert_, assert_equal, assert_almost_equal

#from scikits.image.draw import line

class TestFillLabeledHoles:
    def test_01_00_zeros(self):
        """A label matrix of all zeros has no hole"""
        image = np.zeros((10,10),dtype=int)
        output = morph.fill_labeled_holes(image)
        assert_(np.all(output==0))
    
    def test_01_01_ones(self):
        """Regression test - an image of all ones"""
        image = np.ones((10,10),dtype=int)
        output = morph.fill_labeled_holes(image)
        assert_(np.all(output==1))

    def test_02_object_without_holes(self):
        """The label matrix of a single object without holes has no hole"""
        image = np.zeros((10,10),dtype=int)
        image[3:6,3:6] = 1
        output = morph.fill_labeled_holes(image)
        assert_(np.all(output==image))
    
    def test_03_object_with_hole(self):
        image = np.zeros((20,20),dtype=int)
        image[5:15,5:15] = 1
        image[8:12,8:12] = 0
        output = morph.fill_labeled_holes(image)
        assert_(np.all(output[8:12,8:12] == 1))
        output[8:12,8:12] = 0 # unfill the hole again
        assert_(np.all(output==image))
    
    def test_04_holes_on_edges_are_not_holes(self):
        image = np.zeros((40,40),dtype=int)
        objects = (((15,25),(0,10),(18,22),(0,3)),
                   ((0,10),(15,25),(0,3),(18,22)),
                   ((15,25),(30,39),(18,22),(36,39)),
                   ((30,39),(15,25),(36,39),(18,22)))
        for idx,x in zip(range(1,len(objects)+1),objects):
            image[x[0][0]:x[0][1],x[1][0]:x[1][1]] = idx
            image[x[2][0]:x[2][1],x[3][0]:x[3][1]] = 0
        output = morph.fill_labeled_holes(image)
        for x in objects:
            assert_(np.all(output[x[2][0]:x[2][1],x[3][0]:x[3][1]]==0))
            output[x[2][0]:x[2][1],x[3][0]:x[3][1]] = 1
            assert_(np.all(output[x[0][0]:x[0][1],x[1][0]:x[1][1]]!=0))
            
    def test_05_lots_of_objects_with_holes(self):
        image = np.ones((1020,1020),bool)
        for i in range(0,51):
            image[i*20:i*20+10,:] = ~image[i*20:i*20+10,:]
            image[:,i*20:i*20+10] = ~ image[:,i*20:i*20+10]
        image = ndi.binary_erosion(image, iterations = 2)
        erosion = ndi.binary_erosion(image, iterations = 2)
        image = image & ~ erosion
        labeled_image,nobjects = ndi.label(image)
        output = morph.fill_labeled_holes(labeled_image)
        assert_(np.all(output[erosion] > 0))
    
    def test_06_regression_diamond(self):
        """Check filling the center of a diamond"""
        image = np.zeros((5,5),int)
        image[1,2]=1
        image[2,1]=1
        image[2,3]=1
        image[3,2]=1
        output = morph.fill_labeled_holes(image)
        where = np.argwhere(image != output)
        assert_equal(len(where),1)
        assert_equal(where[0][0],2)
        assert_equal(where[0][1],2)
    
    def test_07_regression_nearby_holes(self):
        """Check filling an object with three holes"""
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,0,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,0,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,0,0,0,0,0,0,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0]])
        expec = np.array([[0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,1,1,1,1,1,1,1,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0]])
        output = morph.fill_labeled_holes(image)
        assert_(np.all(output==expec))
        
    def test_08_fill_small_holes(self):
        """Check filling only the small holes"""
        image = np.zeros((10,20), int)
        image[1:-1,1:-1] = 1
        image[3:8,4:7] = 0     # A hole with area of 5*3 = 15 and not filled
        expected = image.copy()
        image[3:5, 11:18] = 0  # A hole with area 2*7 = 14 is filled
        
        def small_hole_fn(area, is_foreground):
            return area <= 14
        output = morph.fill_labeled_holes(image, size_fn = small_hole_fn)
        assert_(np.all(output == expected))
        
    def test_09_fill_binary_image(self):
        """Make sure that we can fill a binary image too"""
        image = np.zeros((10,20), bool)
        image[1:-1, 1:-1] = True
        image[3:8, 4:7] = False # A hole with area of 5*3 = 15 and not filled
        expected = image.copy()
        image[3:5, 11:18] = False # A hole with area 2*7 = 14 is filled
        def small_hole_fn(area, is_foreground):
            return area <= 14
        output = morph.fill_labeled_holes(image, size_fn = small_hole_fn)
        assert_equal(image.dtype.kind, output.dtype.kind)
        assert_(np.all(output == expected))
        
    def test_10_fill_bullseye(self):
        i,j = np.mgrid[-50:50, -50:50]
        bullseye = i * i + j * j < 2000
        bullseye[i * i + j * j < 1000 ] = False
        bullseye[i * i + j * j < 500 ] = True
        bullseye[i * i + j * j < 250 ] = False
        bullseye[i * i + j * j < 100 ] = True
        labels, count = ndi.label(bullseye)
        result = morph.fill_labeled_holes(labels)
        assert_(np.all(result[result != 0] == bullseye[6, 43]))
        
    def test_11_dont_fill_if_touches_2(self):
        labels = np.array([
            [ 0, 0, 0, 0, 0, 0, 0, 0 ],
            [ 0, 1, 1, 1, 2, 2, 2, 0 ],
            [ 0, 1, 1, 0, 0, 2, 2, 0 ],
            [ 0, 1, 1, 1, 2, 2, 2, 0 ],
            [ 0, 0, 0, 0, 0, 0, 0, 0 ]])
        result = morph.fill_labeled_holes(labels)
        self

class TestAdjacent:
    def test_00_00_zeros(self):
        result = morph.adjacent(np.zeros((10,10), int))
        assert_(np.all(result==False))
    
    def test_01_01_one(self):
        image = np.zeros((10,10), int)
        image[2:5,3:8] = 1
        result = morph.adjacent(image)
        assert_(np.all(result==False))
        
    def test_01_02_not_adjacent(self):
        image = np.zeros((10,10), int)
        image[2:5,3:8] = 1
        image[6:8,3:8] = 2
        result = morph.adjacent(image)
        assert_(np.all(result==False))

    def test_01_03_adjacent(self):
        image = np.zeros((10,10), int)
        image[2:8,3:5] = 1
        image[2:8,5:8] = 2
        expected = np.zeros((10,10), bool)
        expected[2:8,4:6] = True
        result = morph.adjacent(image)
        assert_(np.all(result==expected))
        
    def test_02_01_127_objects(self):
        '''Test that adjacency works for int8 and 127 labels
        
        Regression test of img-1099. Adjacent sets the background to the
        maximum value of the labels matrix + 1. For 127 and int8, it wraps
        around and uses -127.
        '''
        # Create 127 labels
        labels = np.zeros((32,16), np.int8)
        i,j = np.mgrid[0:32, 0:16]
        mask = (i % 2 > 0) & (j % 2 > 0)
        labels[mask] = np.arange(np.sum(mask))
        result = morph.adjacent(labels)
        assert_(np.all(result == False))
        
class TestStrelDisk:
    """Test cellprofiler.cpmath.cpmorphology.strel_disk"""
    
    def test_01_radius2(self):
        """Test strel_disk with a radius of 2"""
        x = morph.strel_disk(2)
        assert_(x.shape[0], 5)
        assert_(x.shape[1], 5)
        y = [0,0,1,0,0,
             0,1,1,1,0,
             1,1,1,1,1,
             0,1,1,1,0,
             0,0,1,0,0]
        ya = np.array(y,dtype=float).reshape((5,5))
        assert_(np.all(x==ya))
    
    def test_02_radius2_point_5(self):
        """Test strel_disk with a radius of 2.5"""
        x = morph.strel_disk(2.5)
        assert_(x.shape[0], 5)
        assert_(x.shape[1], 5)
        y = [0,1,1,1,0,
             1,1,1,1,1,
             1,1,1,1,1,
             1,1,1,1,1,
             0,1,1,1,0]
        ya = np.array(y,dtype=float).reshape((5,5))
        assert_(np.all(x==ya))

class TestBinaryShrink:
    def test_01_zeros(self):
        """Shrink an empty array to itself"""
        input = np.zeros((10,10),dtype=bool)
        result = morph.binary_shrink(input,1)
        assert_(np.all(input==result))
    
    def test_02_cross(self):
        """Shrink a cross to a single point"""
        input = np.zeros((9,9),dtype=bool)
        input[4,:]=True
        input[:,4]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        assert_(len(where)==1)
        assert_(input[where[0][0],where[0][1]])
    
    def test_03_x(self):
        input = np.zeros((9,9),dtype=bool)
        x,y = np.mgrid[-4:5,-4:5]
        input[x==y]=True
        input[x==-y]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        assert_(len(where)==1)
        assert_(input[where[0][0],where[0][1]])
    
    def test_04_block(self):
        """A block should shrink to a point"""
        input = np.zeros((9,9), dtype=bool)
        input[3:6,3:6]=True
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        assert_(len(where)==1)
        assert_(input[where[0][0],where[0][1]])
    
    def test_05_hole(self):
        """A hole in a block should shrink to a ring"""
        input = np.zeros((19,19), dtype=bool)
        input[5:15,5:15]=True
        input[9,9]=False
        result = morph.binary_shrink(input)
        where = np.argwhere(result)
        assert_(len(where) > 1)
        assert_equal(result[9:9], False)

    def test_06_random_filled(self):
        """Shrink random blobs
        
        If you label a random binary image, then fill the holes,
        then shrink the result, each blob should shrink to a point
        """
        np.random.seed(0)
        input = np.random.uniform(size=(300,300)) > .8
        labels,nlabels = ndi.label(input,np.ones((3,3),bool))
        filled_labels = morph.fill_labeled_holes(labels)
        input = filled_labels > 0
        result = morph.binary_shrink(input)
        my_sum = ndi.sum(result.astype(int),filled_labels,np.array(range(nlabels+1),dtype=np.int32))
        my_sum = np.array(my_sum)
        assert_(np.all(my_sum[1:] == 1))
        
    def test_07_all_patterns_of_3x3(self):
        '''Run all patterns of 3x3 with a 1 in the middle
        
        All of these patterns should shrink to a single pixel since
        all are 8-connected and there are no holes
        '''
        for i in range(512):
            a = morph._pattern_of(i)
            if a[1,1]:
                result = morph.binary_shrink(a)
                assert_equal(np.sum(result),1)
    
    def test_08_labels(self):
        '''Run a labels matrix through shrink with two touching objects'''
        labels = np.zeros((10,10),int)
        labels[2:8,2:5] = 1
        labels[2:8,5:8] = 2
        result = morph.binary_shrink(labels)
        assert_equal(np.any(result[labels==0] > 0), False)
        my_sum = fix(ndi.sum(result>0, labels, np.arange(1,3,dtype=np.int32)))
        assert_(np.all(my_sum == 1))
        
class TestMaximum:
    def test_01_zeros(self):
        input = np.zeros((10,10))
        output = morph.maximum(input)
        assert_(np.all(output==input))
    
    def test_01_ones(self):
        input = np.ones((10,10))
        output = morph.maximum(input)
        assert_(np.all(np.abs(output-input)<=np.finfo(float).eps))

    def test_02_center_point(self):
        input = np.zeros((9,9))
        input[4,4] = 1
        expected = np.zeros((9,9))
        expected[3:6,3:6] = 1
        structure = np.ones((3,3),dtype=bool)
        output = morph.maximum(input,structure,(1,1))
        assert_(np.all(output==expected))
    
    def test_03_corner_point(self):
        input = np.zeros((9,9))
        input[0,0]=1
        expected = np.zeros((9,9))
        expected[:2,:2]=1
        structure = np.ones((3,3),dtype=bool)
        output = morph.maximum(input,structure,(1,1))
        assert_(np.all(output==expected))

    def test_04_structure(self):
        input = np.zeros((9,9))
        input[0,0]=1
        input[4,4]=1
        structure = np.zeros((3,3),dtype=bool)
        structure[0,0]=1
        expected = np.zeros((9,9))
        expected[1,1]=1
        expected[5,5]=1
        output = morph.maximum(input,structure,(1,1))
        assert_(np.all(output[1:,1:]==expected[1:,1:]))

    def test_05_big_structure(self):
        big_disk = morph.strel_disk(10).astype(bool)
        input = np.zeros((1001,1001))
        input[500,500] = 1
        expected = np.zeros((1001,1001))
        expected[490:551,490:551][big_disk]=1
        output = morph.maximum(input,big_disk)
        assert_(np.all(output == expected))

class TestRelabel:
    def test_00_relabel_zeros(self):
        input = np.zeros((10,10),int)
        output,count = morph.relabel(input)
        assert_(np.all(input==output))
        assert_equal(count, 0)
    
    def test_01_relabel_one(self):
        input = np.zeros((10,10),int)
        input[3:6,3:6]=1
        output,count = morph.relabel(input)
        assert_(np.all(input==output))
        assert_equal(count,1)
    
    def test_02_relabel_two_to_one(self):
        input = np.zeros((10,10),int)
        input[3:6,3:6]=2
        output,count = morph.relabel(input)
        assert_(np.all((output==1)[input==2]))
        assert_(np.all((input==output)[input!=2]))
        assert_equal(count,1)
    
    def test_03_relabel_gap(self):
        input = np.zeros((20,20),int)
        input[3:6,3:6]=1
        input[3:6,12:15]=3
        output,count = morph.relabel(input)
        assert_(np.all((output==2)[input==3]))
        assert_(np.all((input==output)[input!=3]))
        assert_equal(count,2)

class TestConvexHull:
    def test_00_00_zeros(self):
        """Make sure convex_hull can handle an empty array"""
        result,counts = morph.convex_hull(np.zeros((10,10),int), [])
        assert_equal(np.product(result.shape),0)
        assert_equal(np.product(counts.shape),0)
    
    def test_01_01_zeros(self):
        """Make sure convex_hull can work if a label has no points"""
        result,counts = morph.convex_hull(np.zeros((10,10),int), [1])
        assert_equal(np.product(result.shape),0)
        assert_equal(np.product(counts.shape),1)
        assert_equal(counts[0],0)
    
    def test_01_02_point(self):
        """Make sure convex_hull can handle the degenerate case of one point"""
        labels = np.zeros((10,10),int)
        labels[4,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        assert_equal(result.shape,(1,3))
        assert_equal(result[0,0],1)
        assert_equal(result[0,1],4)
        assert_equal(result[0,2],5)
        assert_equal(counts[0],1)
    
    def test_01_030_line(self):
        """Make sure convex_hull can handle the degenerate case of a line"""
        labels = np.zeros((10,10),int)
        labels[2:8,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        assert_equal(counts[0],2)
        assert_equal(result.shape,(2,3))
        assert_(np.all(result[:,0]==1))
        assert_(result[0,1] in (2,7))
        assert_(result[1,1] in (2,7))
        assert_(np.all(result[:,2]==5))
    
    def test_01_031_odd_line(self):
        """Make sure convex_hull can handle the degenerate case of a line with odd length
        
        This is a regression test: the line has a point in the center if
        it's odd and the sign of the difference of that point is zero
        which causes it to be included in the hull.
        """
        labels = np.zeros((10,10),int)
        labels[2:7,5] = 1
        result,counts = morph.convex_hull(labels,[1])
        assert_equal(counts[0],2)
        assert_equal(result.shape,(2,3))
        assert_(np.all(result[:,0]==1))
        assert_(result[0,1] in (2,6))
        assert_(result[1,1] in (2,6))
        assert_(np.all(result[:,2]==5))
    
    def test_01_04_square(self):
        """Make sure convex_hull can handle a square which is not degenerate"""
        labels = np.zeros((10,10),int)
        labels[2:7,3:8] = 1
        result,counts = morph.convex_hull(labels,[1])
        assert_equal(counts[0],4)
        order = np.lexsort((result[:,2], result[:,1]))
        result = result[order,:]
        expected = np.array([[1,2,3],
                                [1,2,7],
                                [1,6,3],
                                [1,6,7]])
        assert_(np.all(result==expected))
    
    def test_02_01_out_of_order(self):
        """Make sure convex_hull can handle out of order indices"""
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[5,6] = 2
        result,counts = morph.convex_hull(labels,[2,1])
        assert_equal(counts.shape[0],2)
        assert_(np.all(counts==1))
        
        expected = np.array([[2,5,6],[1,2,3]])
        assert_(np.all(result == expected))
    
    def test_02_02_out_of_order(self):
        """Make sure convex_hull can handle out of order indices
        that require different #s of loop iterations"""
        
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[1:7,4:8] = 2
        result,counts = morph.convex_hull(labels, [2,1])
        assert_equal(counts.shape[0],2)
        assert_(np.all(counts==(4,1)))
        assert_equal(result.shape,(5,3))
        order = np.lexsort((result[:,2],result[:,1],
                               np.array([0,2,1])[result[:,0]]))
        result = result[order,:]
        expected = np.array([[2,1,4],
                                [2,1,7],
                                [2,6,4],
                                [2,6,7],
                                [1,2,3]])
        assert_(np.all(result==expected))
    
    def test_02_03_two_squares(self):
        """Make sure convex_hull can handle two complex shapes"""
        labels = np.zeros((10,10),int)
        labels[1:5,3:7] = 1
        labels[6:10,1:7] = 2
        result,counts = morph.convex_hull(labels, [1,2])
        assert_equal(counts.shape[0],2)
        assert_(np.all(counts==(4,4)))
        order = np.lexsort((result[:,2],result[:,1],result[:,0]))
        result = result[order,:]
        expected = np.array([[1,1,3],[1,1,6],[1,4,3],[1,4,6],
                                [2,6,1],[2,6,6],[2,9,1],[2,9,6]])
        assert_(np.all(result==expected))
        
    def test_03_01_concave(self):
        """Make sure convex_hull handles a square with a concavity"""
        labels = np.zeros((10,10),int)
        labels[2:8,3:9] = 1
        labels[3:7,3] = 0
        labels[2:6,4] = 0
        labels[4:5,5] = 0
        result,counts = morph.convex_hull(labels,[1])
        assert_equal(counts[0],4)
        order = np.lexsort((result[:,2],result[:,1],result[:,0]))
        result = result[order,:]
        expected = np.array([[1,2,3],
                                [1,2,8],
                                [1,7,3],
                                [1,7,8]])
        assert_(np.all(result==expected))
        
    def test_04_01_regression(self):
        """The set of points given in this case yielded one in the interior"""
        np.random.seed(0)
        s = 10 # divide each image into this many mini-squares with a shape in each
        side = 250
        mini_side = side / s
        ct = 20
        labels = np.zeros((side,side),int)
        pts = np.zeros((s*s*ct,2),int)
        index = np.array(range(pts.shape[0])).astype(float)/float(ct)
        index = index.astype(int)
        idx = 0
        for i in range(0,side,mini_side):
            for j in range(0,side,mini_side):
                idx = idx+1
                # get ct+1 unique points
                p = np.random.uniform(low=0,high=mini_side,
                                         size=(ct+1,2)).astype(int)
                while True:
                    pu = np.unique(p[:,0]+p[:,1]*mini_side)
                    if pu.shape[0] == ct+1:
                        break
                    p[:pu.shape[0],0] = np.mod(pu,mini_side).astype(int)
                    p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                    p_size = (ct+1-pu.shape[0],2)
                    p[pu.shape[0],:] = np.random.uniform(low=0,
                                                            high=mini_side,
                                                            size=p_size)
                # Use the last point as the "center" and order
                # all of the other points according to their angles
                # to this "center"
                center = p[ct,:]
                v = p[:ct,:]-center
                angle = np.arctan2(v[:,0],v[:,1])
                order = np.lexsort((angle,))
                p = p[:ct][order]
                p[:,0] = p[:,0]+i
                p[:,1] = p[:,1]+j
                pts[(idx-1)*ct:idx*ct,:]=p
                #
                # draw lines on the labels
                #
                for k in range(ct):
                    r, c = line(p[k, 0], p[k, 1], p[(k+1)%ct, 0], p[(k+1)%ct, 1])
                    labels[r, c] = idx

        assert_(labels[5,106]==5)
        result,counts = morph.convex_hull(labels,np.array(range(100))+1)
        assert_equal(np.any(np.logical_and(result[:,1]==5,
                                           result[:,2]==106)), False)
    
    def test_05_01_missing_labels(self):
        """Ensure that there's an entry if a label has no corresponding points"""
        labels = np.zeros((10,10),int)
        labels[3:6,2:8] = 2
        result, counts = morph.convex_hull(labels, np.arange(2)+1)
        assert_equal(counts.shape[0], 2)
        assert_equal(counts[0], 0)
        assert_equal(counts[1], 4)
        
    def test_06_01_regression_373(self):
        '''Regression test of IMG-374'''
        labels = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
        result, counts = morph.convex_hull(labels, np.array([1]))
        assert_equal(counts[0], 2)
        
    def test_06_02_same_point_twice(self):
        '''Regression test of convex_hull_ijv - same point twice in list'''
        
        ii = [79, 11, 65, 73, 42, 26, 46, 48, 14, 53, 73, 42, 59, 12, 59, 65,
              7, 66, 84, 70]
        
        jj = [47, 97, 98,  0, 91, 49, 42, 85, 63, 19,  0,  9, 71, 15, 50, 98,
              14, 46, 89, 47]

        h, c = morph.convex_hull_ijv(
            np.column_stack((ii, jj, np.ones(len(ii)))), [1])
        assert_(np.any((h[:,1] == 73) & (h[:,2] == 0)))

        
class TestMinimumEnclosingCircle:
    def test_00_00_zeros(self):
        """Make sure minimum_enclosing_circle can handle an empty array"""
        center,radius = morph.minimum_enclosing_circle(np.zeros((10,10), int),
                                                       [])
        assert_equal(np.product(center.shape),0)
        assert_equal(np.product(radius.shape),0)
    
    def test_01_01_01_zeros(self):
        """Make sure minimum_enclosing_circle can work if a label has
        no points"""
        
        center,radius = morph.minimum_enclosing_circle(np.zeros((10,10),int),
                                                       [1])
        assert_equal(center.shape,(1,2))
        assert_equal(np.product(radius.shape),1)
        assert_equal(radius[0],0)
    
    def test_01_01_02_zeros(self):
        """Make sure minimum_enclosing_circle can work if one of two labels has
        no points
        
        This is a regression test of a bug
        """
        labels = np.zeros((10,10), int)
        labels[2,2:5] = 3
        labels[2,6:9] = 4
        hull_and_point_count = morph.convex_hull(labels)
        center,radius = morph.minimum_enclosing_circle(
            labels,
            hull_and_point_count=hull_and_point_count)
        assert_equal(center.shape,(2,2))
        assert_equal(np.product(radius.shape),2)
    
    def test_01_02_point(self):
        """Make sure minimum_enclosing_circle can handle the
        degenerate case of one point"""
        
        labels = np.zeros((10,10),int)
        labels[4,5] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        assert_equal(center.shape,(1,2))
        assert_equal(radius.shape,(1,))
        assert_(np.all(center==np.array([(4,5)])))
        assert_equal(radius[0],0)
    
    def test_01_03_line(self):
        """Make sure minimum_enclosing_circle can handle the
        degenerate case of a line"""
        
        labels = np.zeros((10,10),int)
        labels[2:7,5] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        assert_(np.all(center==np.array([(4,5)])))
        assert_equal(radius[0],2)
    
    def test_01_04_square(self):
        """Make sure minimum_enclosing_circle can handle a square
        which is not degenerate"""
        
        labels = np.zeros((10,10),int)
        labels[2:7,3:8] = 1
        center,radius = morph.minimum_enclosing_circle(labels,[1])
        assert_(np.all(center==np.array([(4,5)])))
        assert_almost_equal(radius[0],np.sqrt(8))
    
    def test_02_01_out_of_order(self):
        """Make sure minimum_enclosing_circle can handle out of order
        indices"""
        
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[5,6] = 2
        center,radius = morph.minimum_enclosing_circle(labels,[2,1])
        assert_equal(center.shape,(2,2))
        
        expected_center = np.array(((5,6),(2,3)))
        assert_(np.all(center == expected_center))
    
    def test_02_02_out_of_order(self):
        """Make sure minimum_enclosing_circle can handle out of order indices
        that require different #s of loop iterations"""
        
        labels = np.zeros((10,10),int)
        labels[2,3] = 1
        labels[1:6,4:9] = 2
        center,result = morph.minimum_enclosing_circle(labels, [2,1])
        expected_center = np.array(((3,6),(2,3)))
        assert_(np.all(center == expected_center))
    
    def test_03_01_random_polygons(self):
        """Test minimum_enclosing_circle on 250 random dodecagons"""
        np.random.seed(0)
        s = 10 # divide each image into this many mini-squares
               # with a shape in each
        side = 250
        mini_side = side / s
        ct = 20
        #
        # We keep going until we get at least 10 multi-edge cases -
        # polygons where the minimum enclosing circle intersects 3+ vertices
        #
        n_multi_edge = 0
        while n_multi_edge < 10:
            labels = np.zeros((side,side),int)
            pts = np.zeros((s*s*ct,2),int)
            index = np.array(range(pts.shape[0])).astype(float)/float(ct)
            index = index.astype(int)
            idx = 0
            for i in range(0,side,mini_side):
                for j in range(0,side,mini_side):
                    idx = idx+1
                    # get ct+1 unique points
                    p = np.random.uniform(low=0,high=mini_side,
                                             size=(ct+1,2)).astype(int)
                    while True:
                        pu = np.unique(p[:,0]+p[:,1]*mini_side)
                        if pu.shape[0] == ct+1:
                            break
                        p[:pu.shape[0],0] = np.mod(pu,mini_side).astype(int)
                        p[:pu.shape[0],1] = (pu / mini_side).astype(int)
                        p_size = (ct+1-pu.shape[0],2)
                        p[pu.shape[0],:] = np.random.uniform(low=0,
                                                                high=mini_side,
                                                                size=p_size)
                    # Use the last point as the "center" and order
                    # all of the other points according to their angles
                    # to this "center"
                    center = p[ct,:]
                    v = p[:ct,:]-center
                    angle = np.arctan2(v[:,0],v[:,1])
                    order = np.lexsort((angle,))
                    p = p[:ct][order]
                    p[:,0] = p[:,0]+i
                    p[:,1] = p[:,1]+j
                    pts[(idx-1)*ct:idx*ct,:]=p
                    #
                    # draw lines on the labels
                    #
                    for k in range(ct):
                        r, c = line(p[k, 0], p[k, 1], p[(k+1)%ct, 0], p[(k+1)%ct, 1])
                        labels[r, c] = idx

            center,radius = morph.minimum_enclosing_circle(\
                labels, np.array(range(s**2))+1)
            epsilon = .000001
            center_per_pt = center[index]
            radius_per_pt = radius[index]
            distance_from_center = np.sqrt(np.sum((pts.astype(float)-
                                                         center_per_pt)**2,1))
            #
            # All points must be within the enclosing circle
            #
            assert_(np.all(
                (distance_from_center - epsilon) < radius_per_pt)
                            )
            pt_on_edge = np.abs(distance_from_center - radius_per_pt)<epsilon
            count_pt_on_edge = ndi.sum(pt_on_edge, index,
                                       np.array(range(s**2), dtype=np.int32))
            count_pt_on_edge = np.array(count_pt_on_edge)
            #
            # Every dodecagon must have at least 2 points on the edge.
            #
            assert_(np.all(count_pt_on_edge>=2))
            #
            # Count the multi_edge cases
            #
            n_multi_edge += np.sum(count_pt_on_edge>=3)

class TestEllipseFromSecondMoments:
    def assertWithinFraction(self, actual, expected, 
                             fraction=.001, message=None):
        """Assert that a 'correlation' of the actual value to the expected is
        within the fraction
        
        actual - the value as calculated
        expected - the expected value of the variable
        fraction - the fractional difference of the two
        message - message to print on failure
        
        We divide the absolute difference by 1/2 of the sum of the variables
        to get our measurement.
        """
        measurement = abs(actual-expected)/(2*(actual+expected))
        assert_(measurement < fraction,
                        "%(actual)f != %(expected)f by the measure, "
                        "abs(%(actual)f-%(expected)f)) / 2(%(actual)f "
                        "+ %(expected)f)" % (locals()))
        
    def test_00_00_zeros(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.zeros((10,10)),
                                              np.zeros((10,10),int),
                                              [])
        assert_equal(centers.shape,(0,2))
        assert_equal(eccentricity.shape[0],0)
        assert_equal(major_axis_length.shape[0],0)
        assert_equal(minor_axis_length.shape[0],0)
    
    def test_00_01_zeros(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.zeros((10,10)),
                                              np.zeros((10,10),int),
                                              [1])
        assert_equal(centers.shape,(1,2))
        assert_equal(eccentricity.shape[0],1)
        assert_equal(major_axis_length.shape[0],1)
        assert_equal(minor_axis_length.shape[0],1)
    
    def test_01_01_rectangle(self):
        centers,eccentricity,major_axis_length,minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones((10,20)),
                                              np.ones((10,20),int),
                                              [1])
        assert_equal(centers.shape,(1,2))
        assert_equal(eccentricity.shape[0],1)
        assert_equal(major_axis_length.shape[0],1)
        assert_equal(minor_axis_length.shape[0],1)
        assert_almost_equal(eccentricity[0],.866,2)
        assert_almost_equal(centers[0,0],4.5)
        assert_almost_equal(centers[0,1],9.5)
        assert_almost_equal(major_axis_length[0], 23.0940, decimal=1)
        assert_almost_equal(minor_axis_length[0],11.5470, decimal=1)
        assert_almost_equal(theta[0],0)
    
    def test_01_02_circle(self):
        img = np.zeros((101,101),int)
        y,x = np.mgrid[-50:51,-50:51]
        img[x*x+y*y<=2500] = 1
        centers,eccentricity,major_axis_length, minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones((101,101)),img,[1])
        assert_almost_equal(eccentricity[0],0)
        assert_almost_equal(major_axis_length[0],100, decimal=2)
        assert_almost_equal(minor_axis_length[0],100, decimal=2)
    
    def test_01_03_blob(self):
        '''Regression test a blob against Matlab measurements'''
        blob = np.array(
            [[0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0],
             [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0],
             [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
             [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0]])
        centers,eccentricity,major_axis_length, minor_axis_length,theta =\
            morph.ellipse_from_second_moments(np.ones(blob.shape),blob,[1])
        assert_almost_equal(major_axis_length[0],37.55,1)
        assert_almost_equal(minor_axis_length[0],18.99,1)
        assert_almost_equal(eccentricity[0],0.8627,2)
        assert_almost_equal(centers[0,1],14.1689,2)
        assert_almost_equal(centers[0,0],14.8691,2)
        
    def test_02_01_compactness_square(self):
        image = np.zeros((9,9), int)
        image[1:8,1:8] = 1
        compactness = morph.ellipse_from_second_moments(
            np.ones(image.shape), image, [1], True)[-1]
        i,j = np.mgrid[0:9, 0:9]
        v_i = np.var(i[image > 0])
        v_j = np.var(j[image > 0])
        v = v_i + v_j
        area = np.sum(image > 0)
        expected = 2 * np.pi * v / area
        assert_almost_equal(compactness, expected)
        

class TestCalculateExtents:
    def test_00_00_zeros(self):
        """Make sure calculate_extents doesn't throw an exception if no image"""
        extents = morph.calculate_extents(np.zeros((10,10),int), [1])
    
    def test_01_01_square(self):
        """A square should have an extent of 1"""
        labels = np.zeros((10,10),int)
        labels[1:8,2:9]=1
        extents = morph.calculate_extents(labels,[1])
        assert_almost_equal(extents,1)
    
    def test_01_02_circle(self):
        """A circle should have an extent of pi/4"""
        labels = np.zeros((1001,1001),int)
        y,x = np.mgrid[-500:501,-500:501]
        labels[x*x+y*y<=250000] = 1
        extents = morph.calculate_extents(labels,[1])
        assert_almost_equal(extents,np.pi/4,2)
        
    def test_01_03_two_objects(self):
        """Make sure that calculate_extents works with more than one object
        
        Regression test of a bug: was computing area like this:
        ndimage.sum(labels, labels, indexes)
        which works for the object that's labeled '1', but is 2x for 2, 3x
        for 3, etc... oops.
        """
        labels = np.zeros((10,20), int)
        labels[3:7, 2:5] = 1
        labels[3:5, 5:8] = 1
        labels[2:8, 13:17] = 2
        extents = morph.calculate_extents(labels, [1,2])
        assert_equal(len(extents), 2)
        assert_almost_equal(extents[0], .75)
        assert_almost_equal(extents[1], 1)
         

class TestCalculatePerimeters:
    def test_00_00_zeros(self):
        """The perimeters of a zeros matrix should be all zero"""
        perimeters = morph.calculate_perimeters(np.zeros((10,10),int),[1])
        assert_equal(perimeters,0)
    
    def test_01_01_square(self):
        """The perimeter of a square should be the sum of the sides"""
        
        labels = np.zeros((10,10),int)
        labels[1:9,1:9] = 1
        perimeter = morph.calculate_perimeters(labels, [1])
        assert_equal(perimeter, 4*8)
        
    def test_01_02_circle(self):
        """The perimeter of a circle should be pi * diameter"""
        labels = np.zeros((101,101),int)
        y,x = np.mgrid[-50:51,-50:51]
        labels[x*x+y*y<=2500] = 1
        perimeter = morph.calculate_perimeters(labels, [1])
        epsilon = 20
        assert_(perimeter-np.pi*101<epsilon)
        
    def test_01_03_on_edge(self):
        """Check the perimeter of objects touching edges of matrix"""
        labels = np.zeros((10,20), int)
        labels[:4,:4] = 1 # 4x4 square = 16 pixel perimeter
        labels[-4:,-2:] = 2 # 4x2 square = 2+2+4+4 = 12
        expected = [ 16, 12]
        perimeter = morph.calculate_perimeters(labels, [1,2])
        assert_equal(len(perimeter), 2)
        assert_equal(perimeter[0], expected[0])
        assert_equal(perimeter[1], expected[1])

class TestCalculateConvexArea:
    def test_00_00_degenerate_zero(self):
        """The convex area of an empty labels matrix should be zero"""
        labels = np.zeros((10,10),int)
        result = morph.calculate_convex_hull_areas(labels, [1])
        assert_equal(result.shape[0],1)
        assert_equal(result[0],0)
    
    def test_00_01_degenerate_point(self):
        """The convex area of a point should be 1"""
        labels = np.zeros((10,10),int)
        labels[4,4] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        assert_equal(result.shape[0],1)
        assert_equal(result[0],1)

    def test_00_02_degenerate_line(self):
        """The convex area of a line should be its length"""
        labels = np.zeros((10,10),int)
        labels[1:9,4] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        assert_equal(result.shape[0],1)
        assert_equal(result[0],8)
    
    def test_01_01_square(self):
        """The convex area of a square should be its area"""
        labels = np.zeros((10,10),int)
        labels[1:9,1:9] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        assert_equal(result.shape[0],1)
        assert_almost_equal(result[0],64)
    
    def test_01_02_cross(self):
        """The convex area of a cross should be the area of the
        enclosing diamond
        
        The area of a diamond is 1/2 of the area of the enclosing bounding box
        """
        labels = np.zeros((10,10),int)
        labels[1:9,4] = 1
        labels[4,1:9] = 1
        result = morph.calculate_convex_hull_areas(labels, [1])
        assert_equal(result.shape[0],1)
        assert_almost_equal(result[0],32)
    
    def test_02_01_degenerate_point_and_line(self):
        """Test a degenerate point and line in the same image, out of
        order"""
        
        labels = np.zeros((10,10),int)
        labels[1,1] = 1
        labels[1:9,4] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        assert_equal(result.shape[0],2)
        assert_equal(result[0],8)
        assert_equal(result[1],1)
    
    def test_02_02_degenerate_point_and_square(self):
        """Test a degenerate point and a square in the same image"""
        labels = np.zeros((10,10),int)
        labels[1,1] = 1
        labels[3:8,4:9] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        assert_equal(result.shape[0],2)
        assert_equal(result[1],1)
        assert_almost_equal(result[0],25)
    
    def test_02_03_square_and_cross(self):
        """Test two non-degenerate figures"""
        labels = np.zeros((20,10),int)
        labels[1:9,1:9] = 1
        labels[11:19,4] = 2
        labels[14,1:9] = 2
        result = morph.calculate_convex_hull_areas(labels, [2,1])
        assert_equal(result.shape[0],2)
        assert_almost_equal(result[0],32)
        assert_almost_equal(result[1],64)

class TestEulerNumber:
    def test_00_00_even_zeros(self):
        labels = np.zeros((10,12),int)
        result = morph.euler_number(labels, [1])
        assert_equal(len(result),1)
        assert_equal(result[0],0)
    
    def test_00_01_odd_zeros(self):
        labels = np.zeros((11,13),int)
        result = morph.euler_number(labels, [1])
        assert_equal(len(result),1)
        assert_equal(result[0],0)
    
    def test_01_00_square(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        result = morph.euler_number(labels, [1])
        assert_equal(len(result),1)
        assert_equal(result[0],1)
        
    def test_01_01_square_with_hole(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        labels[3:6,3:6] = 0
        result = morph.euler_number(labels, [1])
        assert_equal(len(result),1)
        assert_equal(result[0],0)
    
    def test_01_02_square_with_two_holes(self):
        labels = np.zeros((10,12),int)
        labels[1:9,1:9] = 1
        labels[2:4,2:8] = 0
        labels[6:8,2:8] = 0
        result = morph.euler_number(labels, [1])
        assert_equal(len(result),1)
        assert_equal(result[0],-1)
    
    def test_02_01_square_touches_border(self):
        labels = np.ones((10,10),int)
        result = morph.euler_number(labels, [1])
        assert_equal(len(result),1)
        assert_equal(result[0],1)
    
    def test_03_01_two_objects(self):
        labels = np.zeros((10,10), int)
        # First object has a hole - Euler # is zero
        labels[1:4,1:4] = 1
        labels[2,2] = 0
        # Second object has no hole - Euler # is 1
        labels[5:8,5:8] = 2
        result = morph.euler_number(labels, [1,2])
        assert_equal(result[0], 0)
        assert_equal(result[1], 1)

    
class TestBranchpoints:
    def test_00_00_zeros(self):
        '''Test branchpoints on an array of all zeros'''
        result = morph.branchpoints(np.zeros((9,11), bool))
        assert_(np.all(result == False))
        
    def test_00_01_zeros_masked(self):
        '''Test branchpoints on an array that is completely masked'''
        result = morph.branchpoints(np.zeros((10,10),bool),
                                    np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_branchpoints_positive(self):
        '''Test branchpoints on positive cases'''
        image = np.array([[1,0,0,1,0,1,0,1,0,1,0,0,1],
                          [0,1,0,1,0,0,1,0,1,0,1,1,0],
                          [1,0,1,0,1,1,0,1,1,1,0,0,1]],bool)
        result = morph.branchpoints(image)
        assert_(np.all(image[1,:] == result[1,:]))
    
    def test_01_02_branchpoints_negative(self):
        '''Test branchpoints on negative cases'''
        image = np.array([[1,0,0,0,1,0,0,0,1,0,1,0,1],
                          [0,1,0,0,1,0,1,1,1,0,0,1,0],
                          [0,0,1,0,1,0,0,0,0,0,0,0,0]],bool)
        result = morph.branchpoints(image)
        assert_(np.all(result==False))
        
    def test_02_01_branchpoints_masked(self):
        '''Test that masking defeats branchpoints'''
        image = np.array([[1,0,0,1,0,1,0,1,1,1,0,0,1],
                          [0,1,0,1,0,0,1,0,1,0,1,1,0],
                          [1,0,1,1,0,1,0,1,1,1,0,0,1]],bool)
        mask  = np.array([[0,1,1,1,1,1,1,0,0,0,1,1,0],
                          [1,1,1,1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,0,1,0,1,1,0,0,1,1,1]],bool)
        result = morph.branchpoints(image, mask)
        assert_(np.all(result[mask]==False))
        
class TestBridge:
    def test_00_00_zeros(self):
        '''Test bridge on an array of all zeros'''
        result = morph.bridge(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test bridge on an array that is completely masked'''
        result = morph.bridge(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_bridge_positive(self):
        '''Test some typical positive cases of bridging'''
        image = np.array([[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1]],bool)
        expected = np.array([[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0],
                             [0,1,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1],
                             [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1]],bool)
        result = morph.bridge(image)
        assert_(np.all(result==expected))
    
    def test_01_02_bridge_negative(self):
        '''Test some typical negative cases of bridging'''
        image = np.array([[1,1,0,0,0,1,1,0,0,0,0,0,0,0,1,1,1,0,0,1,1,0],
                          [0,0,1,0,0,1,0,0,0,1,0,1,0,0,1,0,1,0,0,1,0,0],
                          [0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1]],bool)

        expected = np.array([[1,1,0,0,0,1,1,0,0,0,1,0,0,0,1,1,1,0,0,1,1,0],
                             [0,0,1,0,0,1,0,1,0,1,0,1,0,0,1,0,1,0,0,1,0,1],
                             [0,0,1,0,0,1,1,0,0,1,1,1,0,0,1,1,1,0,0,1,1,1]],bool)
        result = morph.bridge(image)
        assert_(np.all(result==expected))

    def test_02_01_bridge_mask(self):
        '''Test that a masked pixel does not cause a bridge'''
        image = np.array([[1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,1,0,0,1,0,1,0,0,0,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,1,1]],bool)
        mask = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                         [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                         [1,1,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],bool)
        expected = np.array([[1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,1,1,0],
                             [0,0,0,0,0,1,1,1,0,1,1,1,0,0,1,1,1,0,0,1,1,1],
                             [0,0,1,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0,0,0,1,1]],bool)
        result = morph.bridge(image,mask)
        assert_(np.all(result[mask]==expected[mask]))

class TestClean:
    def test_00_00_zeros(self):
        '''Test clean on an array of all zeros'''
        result = morph.clean(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test clean on an array that is completely masked'''
        result = morph.clean(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_clean_positive(self):
        '''Test removal of a pixel using clean'''
        image = np.array([[0,0,0],[0,1,0],[0,0,0]],bool)
        assert_(np.all(morph.clean(image) == False))
    
    def test_01_02_clean_negative(self):
        '''Test patterns that should not clean'''
        image = np.array([[1,0,0,0,1,0,1,0,0,0,1,0,0,0,0,0,1,0,0,1,1,0],
                          [0,1,0,0,1,0,1,0,1,1,0,1,0,0,1,0,1,0,0,1,0,0],
                          [0,0,1,0,0,0,1,0,0,0,0,0,1,0,1,0,0,0,0,0,1,1]],bool)
        assert_(np.all(image == morph.clean(image)))
    
    def test_02_01_clean_edge(self):
        '''Test that clean removes isolated pixels on the edge of an image'''
        
        image = np.array([[1,0,1,0,1],
                          [0,0,0,0,0],
                          [1,0,0,0,1],
                          [0,0,0,0,0],
                          [1,0,1,0,1]],bool)
        assert_(np.all(morph.clean(image) == False))
        
    def test_02_02_clean_mask(self):
        '''Test that clean removes pixels adjoining a mask'''
        image = np.array([[0,0,0],[1,1,0],[0,0,0]],bool)
        mask  = np.array([[1,1,1],[0,1,1],[1,1,1]],bool)
        result= morph.clean(image,mask)
        assert_equal(result[1,1], False)
    
    def test_03_01_clean_labels(self):
        '''Test clean on a labels matrix where two single-pixel objects touch'''
        
        image = np.zeros((10,10), int)
        image[2,2] = 1
        image[2,3] = 2
        image[5:8,5:8] = 3
        result = morph.clean(image)
        assert_(np.all(result[image != 3] == 0))
        assert_(np.all(result[image==3] == 3))

class TestDiag:
    def test_00_00_zeros(self):
        '''Test diag on an array of all zeros'''
        result = morph.diag(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test diag on an array that is completely masked'''
        result = morph.diag(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_diag_positive(self):
        '''Test all cases of diag filling in a pixel'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,1,0,0,0,0,1,0,0],
                          [0,1,0,0,0,0,1,0,0,1,0,1,0],
                          [0,0,0,0,0,0,0,0,0,0,1,0,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,1,0,0,1,1,0,0,1,1,1,0],
                             [0,1,1,0,0,1,1,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.diag(image)
        assert_(np.all(result == expected))
    
    def test_01_02_diag_negative(self):
        '''Test patterns that should not diag'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,0,0,0,1,0,0,1,1,0,1,0,1,0],
                          [0,0,1,0,1,1,0,1,1,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        assert_(np.all(image == morph.diag(image)))
    
    def test_02_01_diag_edge(self):
        '''Test that diag works on edges'''
        
        image = np.array([[1,0,0,0,1],
                          [0,1,0,1,0],
                          [0,0,0,0,0],
                          [0,1,0,1,0],
                          [1,0,0,0,1]],bool)
        expected = np.array([[1,1,0,1,1],
                             [1,1,0,1,1],
                             [0,0,0,0,0],
                             [1,1,0,1,1],
                             [1,1,0,1,1]],bool)
        assert_(np.all(morph.diag(image) == expected))
        image = np.array([[0,1,0,1,0],
                          [1,0,0,0,1],
                          [0,0,0,0,0],
                          [1,0,0,0,1],
                          [0,1,0,1,0]],bool)
        assert_(np.all(morph.diag(image) == expected))
        
        
    def test_02_02_diag_mask(self):
        '''Test that diag connects if one of the pixels is masked'''
        image = np.array([[0,0,0],
                          [1,0,0],
                          [1,1,0]],bool)
        mask  = np.array([[1,1,1],
                          [1,1,1],
                          [0,1,1]],bool)
        result= morph.diag(image,mask)
        assert_equal(result[1,1], True)
        
class TestEndpoints:
    def test_00_00_zeros(self):
        '''Test endpoints on an array of all zeros'''
        result = morph.endpoints(np.zeros((9,11), bool))
        assert_(np.all(result == False))
        
    def test_00_01_zeros_masked(self):
        '''Test endpoints on an array that is completely masked'''
        result = morph.endpoints(np.zeros((10,10),bool),
                                 np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_positive(self):
        '''Test positive endpoint cases'''
        image = np.array([[0,0,0,1,0,1,0,0,0,0,0],
                          [0,1,0,1,0,0,1,0,1,0,1],
                          [1,0,0,0,0,0,0,0,0,1,0]],bool)
        result = morph.endpoints(image)
        assert_(np.all(image[1,:] == result[1,:]))
    
    def test_01_02_negative(self):
        '''Test negative endpoint cases'''
        image = np.array([[0,0,1,0,0,1,0,0,0,0,0,1],
                          [0,1,0,1,0,1,0,0,1,1,0,1],
                          [1,0,0,0,1,0,0,1,0,0,1,0]],bool)
        result = morph.endpoints(image)
        assert_(np.all(result[1,:] == False))
        
    def test_02_02_mask(self):
        """Test that masked positive pixels don't change the endpoint
        determination"""
        image = np.array([[0,0,1,1,0,1,0,1,0,1,0],
                          [0,1,0,1,0,0,1,0,1,0,1],
                          [1,0,0,0,1,0,0,0,0,1,0]],bool)
        mask  = np.array([[1,1,0,1,1,1,1,0,1,0,1],
                          [1,1,1,1,1,1,1,1,1,1,1],
                          [1,1,1,1,0,1,1,1,1,1,1]],bool)
        result = morph.endpoints(image, mask)
        assert_(np.all(image[1,:] == result[1,:]))
    
class TestFill:
    def test_00_00_zeros(self):
        '''Test fill on an array of all zeros'''
        result = morph.fill(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test fill on an array that is completely masked'''
        result = morph.fill(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_fill_positive(self):
        '''Test addition of a pixel using fill'''
        image = np.array([[1,1,1],[1,0,1],[1,1,1]],bool)
        assert_(np.all(morph.fill(image)))
    
    def test_01_02_fill_negative(self):
        '''Test patterns that should not fill'''
        image = np.array([[0,1,1,1,0,1,0,1,1,1,0,1,1,1,1,1,0,1,1,0,0,1],
                          [1,0,1,1,0,1,0,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1],
                          [1,1,0,1,1,1,0,1,1,1,1,1,0,1,0,1,1,1,1,1,0,0]],bool)
        assert_(np.all(image == morph.fill(image)))
    
    def test_02_01_fill_edge(self):
        '''Test that fill fills isolated pixels on an edge'''
        
        image = np.array([[0,1,0,1,0],
                          [1,1,1,1,1],
                          [0,1,1,1,0],
                          [1,1,1,1,1],
                          [0,1,0,1,0]],bool)
        assert_(np.all(morph.fill(image) == True))
        
    def test_02_02_fill_mask(self):
        '''Test that fill adds pixels if a neighbor is masked'''
        image = np.array([[1,1,1],
                          [0,0,1],
                          [1,1,1]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.fill(image,mask)
        assert_equal(result[1,1], True)

class TestHBreak:
    def test_00_00_zeros(self):
        '''Test hbreak on an array of all zeros'''
        result = morph.hbreak(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test hbreak on an array that is completely masked'''
        result = morph.hbreak(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_hbreak_positive(self):
        '''Test break of a horizontal line'''
        image = np.array([[1,1,1],
                          [0,1,0],
                          [1,1,1]],bool)
        expected = np.array([[1,1,1],
                             [0,0,0],
                             [1,1,1]],bool)
        assert_(np.all(morph.hbreak(image)==expected))
    
    def test_01_02_hbreak_negative(self):
        '''Test patterns that should not hbreak'''
        image = np.array([[0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,0],
                          [0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0],
                          [0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,0]],bool)
        assert_(np.all(image == morph.hbreak(image)))
    
class TestVBreak:
    def test_00_00_zeros(self):
        '''Test vbreak on an array of all zeros'''
        result = morph.vbreak(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test vbreak on an array that is completely masked'''
        result = morph.vbreak(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_vbreak_positive(self):
        '''Test break of a vertical line'''
        image = np.array([[1,0,1],
                          [1,1,1],
                          [1,0,1]],bool)
        expected = np.array([[1,0,1],
                             [1,0,1],
                             [1,0,1]],bool)
        assert_(np.all(morph.vbreak(image)==expected))
    
    def test_01_02_vbreak_negative(self):
        '''Test patterns that should not vbreak'''
        # stolen from hbreak
        image = np.array([[0,1,1,0,0,1,0,1,1,0,0,1,1,1,1,1,0,1,1,0,0,1,0],
                          [0,0,1,0,0,1,1,1,0,0,1,0,1,1,0,1,0,1,1,0,1,1,0],
                          [0,1,1,1,0,1,0,1,1,0,1,1,0,1,0,1,1,1,1,1,0,0,0]],bool)
        image = image.transpose()
        assert_(np.all(image == morph.vbreak(image)))
    
class TestMajority:
    def test_00_00_zeros(self):
        '''Test majority on an array of all zeros'''
        result = morph.majority(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test majority on an array that is completely masked'''
        result = morph.majority(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_majority(self):
        '''Test majority on a random field'''
        np.random.seed(0)
        image = np.random.uniform(size=(10,10)) > .5

        expected = ndi.convolve(image.astype(int), np.ones((3,3)), 
                                mode='constant', cval=0) > 4.5

        result = morph.majority(image)
        assert_(np.all(result==expected))
                                        
class TestRemove:
    def test_00_00_zeros(self):
        '''Test remove on an array of all zeros'''
        result = morph.remove(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test remove on an array that is completely masked'''
        result = morph.remove(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_remove_positive(self):
        '''Test removing a pixel'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,0,1,0,0,0,1,1,0,1,1,1,0],
                          [0,1,1,1,0,1,1,1,0,1,1,1,0],
                          [0,0,1,0,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,1,0,0,0,1,1,0,1,1,1,0],
                             [0,1,0,1,0,1,0,1,0,1,0,1,0],
                             [0,0,1,0,0,0,1,0,0,1,1,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.remove(image)
        assert_(np.all(result == expected))
    
    def test_01_02_remove_negative(self):
        '''Test patterns that should not diag'''
        image = np.array([[0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,1,0,0,1,0,0,1,1,0,1,0,1,0],
                          [0,0,1,1,1,1,0,1,1,0,0,1,0,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0]],bool)
        assert_(np.all(image == morph.remove(image)))
    
    def test_02_01_remove_edge(self):
        '''Test that remove does nothing'''
        
        image = np.array([[1,1,1,1,1],
                          [1,1,0,1,1],
                          [1,0,0,0,1],
                          [1,1,0,1,1],
                          [1,1,1,1,1]],bool)
        assert_(np.all(morph.remove(image) == image))
        
    def test_02_02_remove_mask(self):
        '''Test that a masked pixel does not cause a remove'''
        image = np.array([[1,1,1],
                          [1,1,1],
                          [1,1,1]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.remove(image,mask)
        assert_equal(result[1,1], True)

class TestSkeleton:
    def test_00_00_zeros(self):
        '''Test skeletonize on an array of all zeros'''
        result = morph.skeletonize(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test skeletonize on an array that is completely masked'''
        result = morph.skeletonize(np.zeros((10,10),bool),
                                   np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_rectangle(self):
        '''Test skeletonize on a rectangle'''
        image = np.zeros((9,15),bool)
        image[1:-1,1:-1] = True
        #
        # The result should be four diagonals from the
        # corners, meeting in a horizontal line
        #
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                             [0,0,0,0,1,1,1,1,1,1,1,0,0,0,0],
                             [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.skeletonize(image)
        assert_(np.all(result == expected))
    
    def test_01_02_hole(self):
        '''Test skeletonize on a rectangle with a hole in the middle'''
        image = np.zeros((9,15),bool)
        image[1:-1,1:-1] = True
        image[4,4:-4] = False
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,0,0,0,0,0,0,0,0,0,1,0,0],
                             [0,0,1,1,1,1,1,1,1,1,1,1,1,0,0],
                             [0,1,0,0,0,0,0,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.skeletonize(image)
        assert_(np.all(result == expected))
         
class TestSpur:
    def test_00_00_zeros(self):
        '''Test spur on an array of all zeros'''
        result = morph.spur(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test spur on an array that is completely masked'''
        result = morph.spur(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_spur_positive(self):
        '''Test removing a spur pixel'''
        image    = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,1,0,0,0,1,0,1,0,0,0],
                             [0,1,1,1,0,1,0,0,1,0,0,0,1,0,0],
                             [0,0,0,0,0,1,0,1,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,1,0,0,1,0,0,1,0,0,0,1,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        assert_(np.all(result == expected))
    
    def test_01_02_spur_negative(self):
        '''Test patterns that should not spur'''
        image = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                          [0,1,1,0,1,0,0,0,1,0,0,1,0,0,0,1,0,0],
                          [0,0,0,0,1,0,0,1,0,0,0,0,1,0,1,1,1,0],
                          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        l,count = ndi.label(result,ndi.generate_binary_structure(2, 2))
        assert_equal(count, 5)
        a = np.array(ndi.sum(result,l,np.arange(4,dtype=np.int32)+1))
        assert_(np.all((a==1) | (a==4)))
    
    def test_02_01_spur_edge(self):
        '''Test that spurs on edges go away'''
        
        image = np.array([[1,0,0,1,0,0,1],
                          [0,1,0,1,0,1,0],
                          [0,0,1,1,1,0,0],
                          [1,1,1,1,1,1,1],
                          [0,0,1,1,1,0,0],
                          [0,1,0,1,0,1,0],
                          [1,0,0,1,0,0,1]],bool)
        expected = np.array([[0,0,0,0,0,0,0],
                             [0,1,0,1,0,1,0],
                             [0,0,1,1,1,0,0],
                             [0,1,1,1,1,1,0],
                             [0,0,1,1,1,0,0],
                             [0,1,0,1,0,1,0],
                             [0,0,0,0,0,0,0]],bool)
        result = morph.spur(image)
        assert_(np.all(result == expected))
        
    def test_02_02_spur_mask(self):
        '''Test that a masked pixel does not prevent a spur remove'''
        image = np.array([[1,0,0],
                          [1,1,0],
                          [0,0,0]],bool)
        mask  = np.array([[1,1,1],
                          [0,1,1],
                          [1,1,1]],bool)
        result= morph.spur(image,mask)
        assert_equal(result[1,1], False)

class TestThicken:
    def test_00_00_zeros(self):
        '''Test thicken on an array of all zeros'''
        result = morph.thicken(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test thicken on an array that is completely masked'''
        result = morph.thicken(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_thicken_positive(self):
        '''Test thickening positive cases'''
        image    = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                             [0,0,0,0,0,0,0,0,1,0,0,1,0,0,0],
                             [0,1,1,1,0,0,0,1,0,0,0,0,1,0,0],
                             [0,0,0,0,0,0,1,0,0,0,0,0,0,1,0],
                             [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]],bool)
        expected = np.array([[0,0,0,0,0,0,0,1,1,1,1,1,1,0,0],
                             [1,1,1,1,1,0,1,1,1,1,1,1,1,1,0],
                             [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
                             [1,1,1,1,1,1,1,1,1,0,0,1,1,1,1],
                             [0,0,0,0,0,1,1,1,0,0,0,0,1,1,1]],bool)
        result = morph.thicken(image)
        assert_(np.all(result == expected))
    
    def test_01_02_thicken_negative(self):
        '''Test patterns that should not thicken'''
        image = np.array([[1,1,0,1],
                          [0,0,0,0],
                          [1,1,1,1],
                          [0,0,0,0],
                          [1,1,0,1]],bool)
        result = morph.thicken(image)
        assert_(np.all(result==image))
    
    def test_02_01_thicken_edge(self):
        '''Test thickening to the edge'''
        
        image = np.zeros((5,5),bool)
        image[1:-1,1:-1] = True
        result = morph.thicken(image)
        assert_(np.all(result))
        
class TestThin:
    def test_00_00_zeros(self):
        '''Test thin on an array of all zeros'''
        result = morph.thin(np.zeros((10,10),bool))
        assert_(np.all(result==False))
        
    def test_00_01_zeros_masked(self):
        '''Test thin on an array that is completely masked'''
        result = morph.thin(np.zeros((10,10),bool),np.zeros((10,10),bool))
        assert_(np.all(result==False))
    
    def test_01_01_bar(self):
        '''Test thin on a bar of width 3'''
        image = np.zeros((10,10), bool)
        image[3:6,2:8] = True
        expected = np.zeros((10,10), bool)
        expected[4,3:7] = True
        result = morph.thin(expected,iterations = None)
        assert_(np.all(result==expected))
    
    def test_02_01_random(self):
        '''A random image should preserve its Euler number'''
        np.random.seed(0)
        for i in range(20):
            image = np.random.uniform(size=(100,100)) < .1+float(i)/30.
            expected_euler_number = morph.euler_number(image)
            result = morph.thin(image)
            euler_number = morph.euler_number(result)
            assert_(expected_euler_number == euler_number)
    
    def test_03_01_labels(self):
        '''Thin a labeled image'''
        image = np.zeros((10,10), int)
        #
        # This is two touching bars
        #
        image[3:6,2:8] = 1
        image[6:9,2:8] = 2
        expected = np.zeros((10,10),int)
        expected[4,3:7] = 1
        expected[7,3:7] = 2
        result = morph.thin(expected,iterations = None)
        assert_(np.all(result==expected))

class TestTableLookup:
    def test_01_01_all_centers(self):
        '''Test table lookup at pixels off of the edge'''
        image = np.zeros((512*3+2,5),bool)
        for i in range(512):
            pattern = morph._pattern_of(i)
            image[i*3+1:i*3+4,1:4] = pattern
        table = np.arange(512)
        table[511] = 0 # do this to force using the normal mechanism
        index = morph._table_lookup(image, table, False, 1)
        assert_(np.all(index[2::3,2] == table))
    
    def test_01_02_all_corners(self):
        '''Test table lookup at the corners of the image'''
        np.random.seed(0)
        for iteration in range(100):
            table = np.random.uniform(size=512) > .5
            for p00 in (False,True):
                for p01 in (False, True):
                    for p10 in (False, True):
                        for p11 in (False,True):
                            image = np.array([[False,False,False,False,False,False],
                                              [False,p00,  p01,  p00,  p01,  False],
                                              [False,p10,  p11,  p10,  p11,  False],
                                              [False,p00,  p01,  p00,  p01,  False],
                                              [False,p10,  p11,  p10,  p11,  False],
                                              [False,False,False,False,False,False]])
                            expected = morph._table_lookup(image,table,False,1)[1:-1,1:-1]
                            result = morph._table_lookup(image[1:-1,1:-1],table,False,1)
                            assert_(np.all(result==expected),
                                            "Failure case:\n%7s,%s\n%7s,%s"%
                                            (p00,p01,p10,p11))
    
    def test_01_03_all_edges(self):
        '''Test table lookup along the edges of the image'''
        image = np.zeros((32*3+2,6),bool)
        np.random.seed(0)
        for iteration in range(100):
            table = np.random.uniform(size=512) > .5
            for i in range(32):
                pattern = morph._pattern_of(i)
                image[i*3+1:i*3+4,1:3] = pattern[:,:2]
                image[i*3+1:i*3+4,3:5] = pattern[:,:2]
            for im in (image,image.transpose()):
                expected = morph._table_lookup(im,table,False, 1)[1:-1,1:-1]
                result = morph._table_lookup(im[1:-1,1:-1],table,False,1)
                assert_(np.all(result==expected))


class TestNeighbors:
    def test_00_00_zeros(self):
        labels = np.zeros((10,10),int)
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        assert_equal(len(v_counts), 0)
        assert_equal(len(v_indexes), 0)
        assert_equal(len(v_neighbors), 0)
    
    def test_01_01_no_touch(self):
        labels = np.zeros((10,10),int)
        labels[2,2] = 1
        labels[7,7] = 2
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        assert_equal(len(v_counts), 2)
        assert_equal(v_counts[0], 0)
        assert_equal(v_counts[1], 0)
    
    def test_01_02_touch(self):
        labels = np.zeros((10,10),int)
        labels[2,2:5] = 1
        labels[3,2:5] = 2
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        assert_equal(len(v_counts), 2)
        assert_equal(v_counts[0], 1)
        assert_equal(v_neighbors[v_indexes[0]], 2)
        assert_equal(v_counts[1], 1)
        assert_equal(v_neighbors[v_indexes[1]], 1)
    
    def test_01_03_complex(self):
        labels = np.array([[1,1,2,2],
                           [2,2,2,3],
                           [4,3,3,3],
                           [5,6,3,3],
                           [0,7,8,9]])
        v_counts, v_indexes, v_neighbors = morph.find_neighbors(labels)
        assert_equal(len(v_counts), 9)
        for i, neighbors in ((1,[2]),
                             (2,[1,3,4]),
                             (3,[2,4,5,6,7,8,9]),
                             (4,[2,3,5,6]),
                             (5,[3,4,6,7]),
                             (6,[3,4,5,7,8]),
                             (7,[3,5,6,8]),
                             (8,[3,6,7,9]),
                             (9,[3,8])):
            i_neighbors = v_neighbors[v_indexes[i-1]:v_indexes[i-1]+v_counts[i-1]]
            assert_(np.all(i_neighbors == np.array(neighbors)))

class TestColor:
    def test_01_01_color_zeros(self):
        '''Color a labels matrix of all zeros'''
        labels = np.zeros((10,10), int)
        colors = morph.color_labels(labels)
        assert_(np.all(colors==0))
    
    def test_01_02_color_ones(self):
        '''color a labels matrix of all ones'''
        labels = np.ones((10,10), int)
        colors = morph.color_labels(labels)
        assert_(np.all(colors==1))

    def test_01_03_color_complex(self):
        '''Create a bunch of shapes using Voroni cells and color them'''
        np.random.seed(0)
        mask = np.random.uniform(size=(100,100)) < .1
        labels,count = ndi.label(mask, np.ones((3,3),bool))
        distances,(i,j) = ndi.distance_transform_edt(~mask, 
                                                       return_indices = True)
        labels = labels[i,j]
        colors = morph.color_labels(labels)
        l00 = labels[1:-2,1:-2]
        c00 = colors[1:-2,1:-2]
        for i,j in ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)):
            lij = labels[1+i:i-2,1+j:j-2]
            cij = colors[1+i:i-2,1+j:j-2]
            assert_(np.all((l00 == lij) | (c00 != cij)))
            
    def test_02_01_color_127(self):
        '''Color 127 labels stored in a int8 array
        
        Regression test of img-1099
        '''
        # Create 127 labels
        labels = np.zeros((32,16), np.int8)
        i,j = np.mgrid[0:32, 0:16]
        mask = (i % 2 > 0) & (j % 2 > 0)
        labels[mask] = np.arange(np.sum(mask))
        colors = morph.color_labels(labels)
        assert_(np.all(colors[labels==0] == 0))
        assert_(np.all(colors[labels!=0] == 1))
            
class TestSkeletonizeLabels:
    def test_01_01_skeletonize_complex(self):
        '''Skeletonize a complex field of shapes and check each individually'''
        np.random.seed(0)
        mask = np.random.uniform(size=(100,100)) < .1
        labels,count = ndi.label(mask, np.ones((3,3),bool))
        distances,(i,j) = ndi.distance_transform_edt(~mask, 
                                                       return_indices = True)
        labels = labels[i,j]
        skel = morph.skeletonize_labels(labels)
        for i in range(1,count+1,10):
            mask = labels == i
            skel_test = morph.skeletonize(mask)
            assert_(np.all(skel[skel_test] == i))
            assert_(np.all(skel[~skel_test] != i))

class TestAssociateByDistance:
    def test_01_01_zeros(self):
        '''Test two label matrices with nothing in them'''
        result = morph.associate_by_distance(np.zeros((10,10),int),
                                             np.zeros((10,10),int), 0)
        assert_equal(result.shape[0], 0)
    
    def test_01_02_one_zero(self):
        '''Test a labels matrix with objects against one without'''
        result = morph.associate_by_distance(np.ones((10,10),int),
                                             np.zeros((10,10),int), 0)
        assert_equal(result.shape[0], 0)
    
    def test_02_01_point_in_square(self):
        '''Test a single point in a square'''
        #
        # Point is a special case - only one point in its convex hull
        #
        l1 = np.zeros((10,10),int)
        l1[1:5,1:5] = 1
        l1[5:9,5:9] = 2
        l2 = np.zeros((10,10),int)
        l2[2,3] = 3
        l2[2,9] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        assert_equal(result.shape[0], 1)
        assert_equal(result[0,0],1)
        assert_equal(result[0,1],3)
    
    def test_02_02_line_in_square(self):
        '''Test a line in a square'''
        l1 = np.zeros((10,10),int)
        l1[1:5,1:5] = 1
        l1[5:9,5:9] = 2
        l2 = np.zeros((10,10),int)
        l2[2,2:5] = 3
        l2[2,6:9] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        assert_equal(result.shape[0], 1)
        assert_equal(result[0,0],1)
        assert_equal(result[0,1],3)
    
    def test_03_01_overlap(self):
        '''Test a square overlapped by four other squares'''
        
        l1 = np.zeros((20,20),int)
        l1[5:16,5:16] = 1
        l2 = np.zeros((20,20),int)
        l2[1:6,1:6] = 1
        l2[1:6,14:19] = 2
        l2[14:19,1:6] = 3
        l2[14:19,14:19] = 4
        result = morph.associate_by_distance(l1, l2, 0)
        assert_equal(result.shape[0],4)
        assert_(np.all(result[:,0]==1))
        assert_(all([x in result[:,1] for x in range(1,5)]))
    
    def test_03_02_touching(self):
        '''Test two objects touching at one point'''
        l1 = np.zeros((10,10), int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,10), int)
        l2[5:9,5:9] = 1
        result = morph.associate_by_distance(l1, l2, 0)
        assert_equal(result.shape[0], 1)
        assert_equal(result[0,0],1)
        assert_equal(result[0,1],1)
    
    def test_04_01_distance_square(self):
        '''Test two squares separated by a distance'''
        l1 = np.zeros((10,20),int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,20),int)
        l2[3:6,10:16] = 1
        result = morph.associate_by_distance(l1,l2, 4)
        assert_equal(result.shape[0],0)
        result = morph.associate_by_distance(l1,l2, 5)
        assert_equal(result.shape[0],1)
    
    def test_04_02_distance_triangle(self):
        '''Test a triangle and a square (edge to point)'''
        l1 = np.zeros((10,20),int)
        l1[3:6,3:6] = 1
        l2 = np.zeros((10,20),int)
        l2[4,10] = 1
        l2[3:6,11] = 1
        l2[2:7,12] = 1
        result = morph.associate_by_distance(l1,l2, 4)
        assert_equal(result.shape[0],0)
        result = morph.associate_by_distance(l1,l2, 5)
        assert_equal(result.shape[0],1)

class TestDistanceToEdge:
    '''Test distance_to_edge'''
    def test_01_01_zeros(self):
        '''Test distance_to_edge with a matrix of zeros'''
        result = morph.distance_to_edge(np.zeros((10,10),int))
        assert_(np.all(result == 0))
    
    def test_01_02_square(self):
        '''Test distance_to_edge with a 3x3 square'''
        labels = np.zeros((10,10), int)
        labels[3:6,3:6] = 1
        expected = np.zeros((10,10))
        expected[3:6,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        result = morph.distance_to_edge(labels)
        assert_(np.all(result == expected))
    
    def test_01_03_touching(self):
        '''Test distance_to_edge when two objects touch each other'''
        labels = np.zeros((10,10), int)
        labels[3:6,3:6] = 1
        labels[6:9,3:6] = 2
        expected = np.zeros((10,10))
        expected[3:6,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        expected[6:9,3:6] = np.array([[1,1,1],[1,2,1],[1,1,1]])
        result = morph.distance_to_edge(labels)
        assert_(np.all(result == expected))

       
class TestAllConnectedComponents:
    def test_01_01_no_edges(self):
        result = morph._cp_all_connected_components(np.array([], int), np.array([], int))
        assert_equal(len(result), 0)
        
    def test_01_02_one_component(self):
        result = morph._cp_all_connected_components(np.array([0]), np.array([0]))
        assert_equal(len(result),1)
        assert_equal(result[0], 0)
        
    def test_01_03_two_components(self):
        result = morph._cp_all_connected_components(np.array([0,1]), 
                                                    np.array([0,1]))
        assert_equal(len(result),2)
        assert_equal(result[0], 0)
        assert_equal(result[1], 1)
        
    def test_01_04_one_connection(self):
        result = morph._cp_all_connected_components(np.array([0,1,2]),
                                                    np.array([0,2,1]))
        assert_equal(len(result),3)
        assert_(np.all(result == np.array([0,1,1])))
        
    def test_01_05_components_can_label(self):
        #
        # all_connected_components can be used to label a matrix
        #
        np.random.seed(0)
        for d in ((10,12),(100,102)):
            mask = np.random.uniform(size=d) < .2
            mask[-1,-1] = True
            #
            # Just do 4-connectivity
            #
            labels, count = ndi.label(mask)
            i,j = np.mgrid[0:d[0],0:d[1]]
            connected_top = (i > 0) & mask[i,j] & mask[i-1,j]
            idx = np.arange(np.prod(d))
            idx.shape = d
            connected_top_j = idx[connected_top] - d[1]
            
            connected_bottom = (i < d[0]-1) & mask[i,j] & mask[(i+1) % d[0],j]
            connected_bottom_j = idx[connected_bottom] + d[1]
            
            connected_left = (j > 0) & mask[i,j] & mask[i,j-1]
            connected_left_j = idx[connected_left] - 1
            
            connected_right = (j < d[1]-1) & mask[i,j] & mask[i,(j+1) % d[1]]
            connected_right_j = idx[connected_right] + 1
            
            i = np.hstack((idx[mask],
                           idx[connected_top],
                           idx[connected_bottom],
                           idx[connected_left],
                           idx[connected_right]))
            j = np.hstack((idx[mask], connected_top_j, connected_bottom_j,
                           connected_left_j, connected_right_j))
            result = morph._cp_all_connected_components(i,j)
            assert_equal(len(result), np.prod(d))
            result.shape = d
            result[mask] += 1
            result[~mask] = 0
            #
            # Correlate the labels with the result
            #
            import scipy.sparse
            coo = scipy.sparse.coo_matrix((np.ones(np.prod(d)),
                                           (labels.flatten(),
                                            result.flatten())))
            corr = coo.toarray()
            #
            # Make sure there's either no or one hit per label association
            #
            assert_(np.all(np.sum(corr != 0,0) <= 1))
            assert_(np.all(np.sum(corr != 0,1) <= 1))


class TestBranchings:
    def test_00_00_zeros(self):
        assert_(np.all(morph.branchings(np.zeros((10,11), bool)) == 0))
        
    def test_01_01_endpoint(self):
        image = np.zeros((10,11), bool)
        image[5,5:] = True
        assert_equal(morph.branchings(image)[5,5], 1)
        
    def test_01_02_line(self):
        image = np.zeros((10,11), bool)
        image[1:9, 5] = True
        assert_(np.all(morph.branchings(image)[2:8,5] == 2))
        
    def test_01_03_vee(self):
        image = np.zeros((11,11), bool)
        i,j = np.mgrid[-5:6,-5:6]
        image[-i == abs(j)] = True
        image[(j==0) & (i > 0)] = True
        assert_(morph.branchings(image)[5,5] == 3)
        
    def test_01_04_quadrabranch(self):
        image = np.zeros((11,11), bool)
        i,j = np.mgrid[-5:6,-5:6]
        image[abs(i) == abs(j)] = True
        assert_(morph.branchings(image)[5,5] == 4)


class TestLabelSkeleton:
    def test_00_00_zeros(self):
        '''Label a skeleton containing nothing'''
        skeleton = np.zeros((20,10), bool)
        result, count = morph.label_skeleton(skeleton)
        assert_equal(count, 0)
        assert_(np.all(result == 0))
        
    def test_01_01_point(self):
        '''Label a skeleton consisting of a single point'''
        skeleton = np.zeros((20,10), bool)
        skeleton[5,5] = True
        expected = np.zeros((20,10), int)
        expected[5,5] = 1
        result, count = morph.label_skeleton(skeleton)
        assert_equal(count, 1)
        assert_(np.all(result == expected))
        
    def test_01_02_line(self):
        """Label a skeleton that's a line"""
        skeleton = np.zeros((20,10), bool)
        skeleton[5:15, 5] = True
        result, count = morph.label_skeleton(skeleton)
        assert_equal(count, 1)
        assert_(np.all(result[skeleton] == 1))
        assert_(np.all(result[~skeleton] == 0))
        
    def test_01_03_branch(self):
        '''Label a skeleton that has a branchpoint'''
        skeleton = np.zeros((21,11), bool)
        i,j = np.mgrid[-10:11,-5:6]
        #
        # Looks like this:
        #  .   .
        #   . .
        #    .
        #    .
        skeleton[(i < 0) & (np.abs(i) == np.abs(j))] = True
        skeleton[(i >= 0) & (j == 0)] = True
        result, count = morph.label_skeleton(skeleton)
        assert_equal(count, 4)
        assert_(np.all(result[~skeleton] == 0))
        assert_(np.all(result[skeleton] > 0))
        assert_equal(result[10,5], 1)
        v1 = result[5,0]
        v2 = result[5,-1]
        v3 = result[-1, 5]
        assert_equal(len(np.unique((v1, v2, v3))), 3)
        assert_(np.all(result[(i < 0) & (i==j)] == v1))
        assert_(np.all(result[(i < 0) & (i==-j)] == v2))
        assert_(np.all(result[(i > 0) & (j == 0)] == v3))
        
    def test_02_01_branch_and_edge(self):
        '''A branchpoint meeting an edge at two points'''
        
        expected = np.array(((2,0,0,0,0,1),
                             (0,2,0,0,1,0),
                             (0,0,3,1,0,0),
                             (0,0,4,0,0,0),
                             (0,4,0,0,0,0),
                             (4,0,0,0,0,0)))
        skeleton = expected > 0
        result, count = morph.label_skeleton(skeleton)
        assert_equal(count, 4)
        assert_(np.all(result[~skeleton] == 0))
        assert_equal(len(np.unique(result)), 5)
        assert_equal(np.max(result), 4)
        assert_equal(np.min(result), 0)
        for i in range(1,5):
            assert_equal(len(np.unique(result[expected == i])), 1)

    def test_02_02_four_edges_meet(self):
        """Check the odd case of four edges meeting at a square
        
        The shape is something like this:
        
        .    .
         .  .
          ..
          ..
         .  .
        .    .
        None of the points above are branchpoints - they're sort of
        half-branchpoints.
        """
        i,j = np.mgrid[-10:10,-10:10]
        i[i<0] += 1
        j[j<0] += 1
        skeleton=np.abs(i) == np.abs(j)
        result, count = morph.label_skeleton(skeleton)
        assert_equal(count, 4)
        assert_(np.all(result[~skeleton]==0))
        assert_equal(np.max(result), 4)
        assert_equal(np.min(result), 0)
        assert_equal(len(np.unique(result)), 5)
        for im in (-1, 1):
            for jm in (-1, 1):
                assert_equal(len(np.unique(result[(i*im == j*jm) & 
                                                      (i*im > 0) &
                                                      (j*jm > 0)])), 1)

 
class TestIsLocalMaximum:
    def test_00_00_empty(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        assert_(np.all(~ result))
        
    def test_01_01_one_point(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        labels[5,5] = 1
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        assert_(np.all(result == (labels == 1)))
        
    def test_01_02_adjacent_and_same(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5:6] = 1
        labels[5,5:6] = 1
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        assert_(np.all(result == (labels == 1)))
        
    def test_01_03_adjacent_and_different(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,6] = .5
        labels[5,5:6] = 1
        expected = (image == 1)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        assert_(np.all(result == expected))
        
    def test_01_04_not_adjacent_and_different(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,8] = .5
        labels[image > 0] = 1
        expected = (labels == 1)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        assert_(np.all(result == expected))
        
    def test_01_05_two_objects(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,15] = .5
        labels[5,5] = 1
        labels[5,15] = 2
        expected = (labels > 0)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        assert_(np.all(result == expected))

    def test_01_06_adjacent_different_objects(self):
        image = np.zeros((10,20))
        labels = np.zeros((10,20), int)
        image[5,5] = 1
        image[5,6] = .5
        labels[5,5] = 1
        labels[5,6] = 2
        expected = (labels > 0)
        result = morph.is_local_maximum(image, labels, np.ones((3,3), bool))
        assert_(np.all(result == expected))
        
    def test_02_01_four_quadrants(self):
        np.random.seed(21)
        image = np.random.uniform(size=(40,60))
        i,j = np.mgrid[0:40,0:60]
        labels = 1 + (i >= 20) + (j >= 30) * 2
        i,j = np.mgrid[-3:4,-3:4]
        footprint = (i*i + j*j <=9)
        expected = np.zeros(image.shape, float)
        for imin, imax in ((0, 20), (20, 40)):
            for jmin, jmax in ((0, 30), (30, 60)):
                expected[imin:imax,jmin:jmax] = ndi.maximum_filter(
                    image[imin:imax, jmin:jmax], footprint = footprint)
        expected = (expected == image)
        result = morph.is_local_maximum(image, labels, footprint)
        assert_(np.all(result == expected))
        
    def test_03_01_disk_1(self):
        '''regression test of img-1194, footprint = [1]
        
        Test is_local_maximum when every point is a local maximum
        '''
        np.random.seed(31)
        image = np.random.uniform(size=(10,20))
        footprint = morph.strel_disk(.5)
        assert_equal(np.prod(footprint.shape), 1)
        assert_equal(footprint[0,0], 1)
        result = morph.is_local_maximum(image, np.ones((10,20)), footprint)
        assert_(np.all(result))
        

class TestAngularDistribution:
    def test_00_00_angular_dist(self):
        np.random.seed(0)
        # random labels from 0 to 9
        labels = (np.random.uniform(0, 0.95, (1000, 1000)) * 10).astype(np.int)
        # filled square of 11 (NB: skipped 10)
        labels[200:300, 600:900] = 11
        angdist = morph.angular_distribution(labels)
        # 10 is an empty label
        assert np.all(angdist[9, :] == 0.0)
        # check approximation to chord ratio of filled rectangle (roughly 3.16)
        resolution = angdist.shape[1]
        angdist2 = angdist[-1, :resolution/2] + angdist[-1, resolution/2:]
        assert np.abs(3.16 - np.sqrt(angdist2.max() / angdist2.min())) < 0.05

class TestFeretDiameter:
    def test_00_00_none(self):
        result = morph.feret_diameter(np.zeros((0,3)), np.zeros(0, int), [])
        assert_equal(len(result), 0)
        
    def test_00_01_point(self):
        min_result, max_result = morph.feret_diameter(
            np.array([[1, 0, 0]]),
            np.ones(1, int), [1])
        assert_equal(len(min_result), 1)
        assert_equal(min_result[0], 0)
        assert_equal(len(max_result), 1)
        assert_equal(max_result[0], 0)
        
    def test_01_02_line(self):
        min_result, max_result = morph.feret_diameter(
            np.array([[1, 0, 0], [1, 1, 1]]),
            np.array([2], int), [1])
        assert_equal(len(min_result), 1)
        assert_equal(min_result[0], 0)
        assert_equal(len(max_result), 1)
        assert_equal(max_result[0], np.sqrt(2))
        
    def test_01_03_single(self):
        r = np.random.RandomState()
        r.seed(204)
        niterations = 100
        iii = r.randint(0, 100, size=(20 * niterations))
        jjj = r.randint(0, 100, size=(20 * niterations))
        for iteration in range(100):
            ii = iii[(iteration * 20):((iteration + 1) * 20)]
            jj = jjj[(iteration * 20):((iteration + 1) * 20)]
            chulls, counts = morph.convex_hull_ijv(
                np.column_stack((ii, jj, np.ones(20, int))), [1])
            min_result, max_result = morph.feret_diameter(chulls, counts, [1])
            assert_equal(len(min_result), 1)
            distances = np.sqrt(
                ((ii[:,np.newaxis] - ii[np.newaxis,:]) ** 2 +
                 (jj[:,np.newaxis] - jj[np.newaxis,:]) ** 2).astype(float))
            expected = np.max(distances)
            if abs(max_result - expected) > .000001:
                a0,a1 = np.argwhere(distances == expected)[0]
                assert_almost_equal(
                    max_result[0], expected,
                    msg = "Expected %f, got %f, antipodes are %d,%d and %d,%d" %
                (expected, result, ii[a0], jj[a0], ii[a1], jj[a1]))
            #
            # Do a 180 degree sweep, measuring
            # the Feret diameter at each angle. Stupid but an independent test.
            #
            # Draw a line segment from the origin to a point at the given
            # angle from the horizontal axis
            #
            angles = np.pi * np.arange(20).astype(float) / 20.0
            i = -np.sin(angles)
            j = np.cos(angles)
            chull_idx, angle_idx = np.mgrid[0:counts[0],0:20]
            #
            # Compose a list of all vertices on the convex hull and all lines
            #
            v = chulls[chull_idx.ravel(),1:]
            pt1 = np.zeros((20 * counts[0], 2))
            pt2 = np.column_stack([i[angle_idx.ravel()], j[angle_idx.ravel()]])
            #
            # For angles from 90 to 180, the parallel line has to be sort of
            # at negative infinity instead of zero to keep all points on
            # the same side
            #
            pt1[angle_idx.ravel() < 10,1] -= 200
            pt2[angle_idx.ravel() < 10,1] -= 200
            pt1[angle_idx.ravel() >= 10,0] += 200
            pt2[angle_idx.ravel() >= 10,0] += 200
            distances = np.sqrt(morph.distance2_to_line(v, pt1, pt2))
            distances.shape = (counts[0], 20)
            dmin = np.min(distances, 0)
            dmax = np.max(distances, 0)
            expected_min = np.min(dmax - dmin)
            assert_(min_result[0] <= expected_min)
            
    def test_02_01_multiple_objects(self):
        r = np.random.RandomState()
        r.seed(204)
        niterations = 100
        ii = r.randint(0, 100, size=(20 * niterations))
        jj = r.randint(0, 100, size=(20 * niterations))
        vv = np.hstack([np.ones(20) * i for i in range(1,niterations+1)])
        indexes = np.arange(1, niterations+1)
        chulls, counts = morph.convex_hull_ijv(
            np.column_stack((ii, jj, vv)), indexes)
        min_result, max_result = morph.feret_diameter(chulls, counts, indexes)
        assert_equal(len(max_result), niterations)
        for i in range(niterations):
            #
            # Make sure values are same as single (validated) case.
            #
            iii = ii[(20*i):(20*(i+1))]
            jjj = jj[(20*i):(20*(i+1))]
            chulls, counts = morph.convex_hull_ijv(
                np.column_stack((iii, jjj, np.ones(len(iii), int))), [1])
            expected_min, expected_max = morph.feret_diameter(chulls, counts, [1])
            assert_almost_equal(expected_min[0], min_result[i])
            assert_almost_equal(expected_max[0], max_result[i])

if __name__ == "__main__":
    import numpy.testing
    numpy.testing.run_module_suite()
