"""test_watershed.py - tests the watershed function

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
#Portions of this test were taken from scipy's watershed test in test_ndimage.py
#
# Copyright (C) 2003-2005 Peter J. Verveer
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above
#    copyright notice, this list of conditions and the following
#    disclaimer in the documentation and/or other materials provided
#    with the distribution.
#
# 3. The name of the author may not be used to endorse or promote
#    products derived from this software without specific prior
#    written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS
# OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
# GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

__version__ = "$Revision$"

import math
import time
import unittest

import numpy as np
import scipy.ndimage

from scikits.image.morphology.watershed import watershed,fast_watershed, _slow_watershed

eps = 1e-12

def diff(a, b):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    if not isinstance(b, np.ndarray):
        b = np.asarray(b)
    if (0 in a.shape) and (0 in b.shape):
        return 0.0
    b[a==0]=0
    if (a.dtype in [np.complex64, np.complex128] or
        b.dtype in [np.complex64, np.complex128]):
        a = np.asarray(a, np.complex128)
        b = np.asarray(b, np.complex128)
        t = ((a.real - b.real)**2).sum() + ((a.imag - b.imag)**2).sum()
    else:
        a = np.asarray(a)
        a = a.astype(np.float64)
        b = np.asarray(b)
        b = b.astype(np.float64)
        t = ((a - b)**2).sum()
    return math.sqrt(t)

class TestFastWatershed(unittest.TestCase):
    eight = np.ones((3,3),bool)
    def test_watershed01(self):
        "watershed 1"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[ -1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 1, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0]],
                                 np.int8)
        out = fast_watershed(data, markers,self.eight)
        expected = np.array([[-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]])
        error = diff(expected, out)
        assert error < eps
        out = _slow_watershed(data, markers, 8)
        error = diff(expected, out)
        assert error < eps

    def test_watershed02(self):
        "watershed 2"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[ -1, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 1, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0]],
                                 np.int8)
        out = fast_watershed(data, markers)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1,  1,  1,  1, -1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1,  1,  1,  1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed03(self):
        "watershed 3"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 2, 0, 3, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, -1]],
                                 np.int8)
        out = fast_watershed(data, markers)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  0,  2,  0,  3,  0, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  0,  2,  0,  3,  0, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed04(self):
        "watershed 4"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 2, 0, 3, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, -1]],
                                 np.int8)
        out = fast_watershed(data, markers,self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1,  2,  2,  0,  3,  3, -1],
                      [-1, -1, -1, -1, -1, -1, -1],                      
                      [-1, -1, -1, -1, -1, -1, -1],                      
                      [-1, -1, -1, -1, -1, -1, -1],                      
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed05(self):
        "watershed 5"
        data = np.array([[0, 0, 0, 0, 0, 0, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 0, 1, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 3, 0, 2, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, 0],
                                  [ 0, 0, 0, 0, 0, 0, -1]],
                                 np.int8)
        out = fast_watershed(data, markers,self.eight)
        error = diff([[-1, -1, -1, -1, -1, -1, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1,  3,  3,  0,  2,  2, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed06(self):
        "watershed 6"
        data = np.array([[0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 0, 0, 0, 1, 0],
                               [0, 1, 1, 1, 1, 1, 0],
                               [0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 0]], np.uint8)
        markers = np.array([[ 0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 1, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  0, 0, 0, 0, 0, 0, 0],
                                  [  -1, 0, 0, 0, 0, 0, 0]],
                                 np.int8)
        out = fast_watershed(data, markers,self.eight)
        error = diff([[-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1,  1,  1,  1,  1,  1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1],
                      [-1, -1, -1, -1, -1, -1, -1]], out)
        self.failUnless(error < eps)

    def test_watershed07(self):
        "A regression test of a competitive case that failed"
        data = np.array([[255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,204,204,204,204,204,204,255,255,255,255,255],
                            [255,255,255,204,204,183,153,153,153,153,183,204,204,255,255,255],
                            [255,255,204,183,153,141,111,103,103,111,141,153,183,204,255,255],
                            [255,255,204,153,111, 94, 72, 52, 52, 72, 94,111,153,204,255,255],
                            [255,255,204,153,111, 72, 39,  1, 1, 39, 72,111,153,204,255,255],
                            [255,255,204,183,141,111, 72, 39, 39, 72,111,141,183,204,255,255],
                            [255,255,255,204,183,141,111, 72, 72,111,141,183,204,255,255,255],
                            [255,255,255,255,204,183,141, 94, 94,141,183,204,255,255,255,255],
                            [255,255,255,255,255,204,153,103,103,153,204,255,255,255,255,255],
                            [255,255,255,255,204,183,141, 94, 94,141,183,204,255,255,255,255],
                            [255,255,255,204,183,141,111, 72, 72,111,141,183,204,255,255,255],
                            [255,255,204,183,141,111, 72, 39, 39, 72,111,141,183,204,255,255],
                            [255,255,204,153,111, 72, 39,  1,  1, 39, 72,111,153,204,255,255],
                            [255,255,204,153,111, 94, 72, 52, 52, 72, 94,111,153,204,255,255],
                            [255,255,204,183,153,141,111,103,103,111,141,153,183,204,255,255],
                            [255,255,255,204,204,183,153,153,153,153,183,204,204,255,255,255],
                            [255,255,255,255,255,204,204,204,204,204,204,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]])
        mask = (data!=255)
        markers = np.zeros(data.shape,int)
        markers[6,7] = 1
        markers[14,7] = 2
        out = fast_watershed(data, markers,self.eight,mask=mask)
        #
        # The two objects should be the same size, except possibly for the
        # border region
        #
        size1 = np.sum(out==1)
        size2 = np.sum(out==2)
        self.assertTrue(abs(size1-size2) <=6)
    
    def test_watershed08(self):
        "The border pixels + an edge are all the same value"
        data = np.array([[255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,204,204,204,204,204,204,255,255,255,255,255],
                            [255,255,255,204,204,183,153,153,153,153,183,204,204,255,255,255],
                            [255,255,204,183,153,141,111,103,103,111,141,153,183,204,255,255],
                            [255,255,204,153,111, 94, 72, 52, 52, 72, 94,111,153,204,255,255],
                            [255,255,204,153,111, 72, 39,  1, 1, 39, 72,111,153,204,255,255],
                            [255,255,204,183,141,111, 72, 39, 39, 72,111,141,183,204,255,255],
                            [255,255,255,204,183,141,111, 72, 72,111,141,183,204,255,255,255],
                            [255,255,255,255,204,183,141, 94, 94,141,183,204,255,255,255,255],
                            [255,255,255,255,255,204,153,141,141,153,204,255,255,255,255,255],
                            [255,255,255,255,204,183,141, 94, 94,141,183,204,255,255,255,255],
                            [255,255,255,204,183,141,111, 72, 72,111,141,183,204,255,255,255],
                            [255,255,204,183,141,111, 72, 39, 39, 72,111,141,183,204,255,255],
                            [255,255,204,153,111, 72, 39,  1,  1, 39, 72,111,153,204,255,255],
                            [255,255,204,153,111, 94, 72, 52, 52, 72, 94,111,153,204,255,255],
                            [255,255,204,183,153,141,111,103,103,111,141,153,183,204,255,255],
                            [255,255,255,204,204,183,153,153,153,153,183,204,204,255,255,255],
                            [255,255,255,255,255,204,204,204,204,204,204,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255],
                            [255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255]])
        mask = (data!=255)
        markers = np.zeros(data.shape,int)
        markers[6,7] = 1
        markers[14,7] = 2
        out = fast_watershed(data, markers,self.eight,mask=mask)
        #
        # The two objects should be the same size, except possibly for the
        # border region
        #
        size1 = np.sum(out==1)
        size2 = np.sum(out==2)
        self.assertTrue(abs(size1-size2) <=6)
    
    def test_watershed09(self):
        """Test on an image of reasonable size
        
        This is here both for timing (does it take forever?) and to
        ensure that the memory constraints are reasonable
        """
        image = np.zeros((1000,1000))
        coords = np.random.uniform(0,1000,(100,2)).astype(int)
        markers = np.zeros((1000,1000),int)
        idx = 1
        for x,y in coords:
            image[x,y] = 1
            markers[x,y] = idx
            idx += 1
        
        image = scipy.ndimage.gaussian_filter(image, 4)
        before = time.clock() 
        out = fast_watershed(image,markers,self.eight)
        elapsed = time.clock()-before
        print "Fast watershed ran a megapixel image in %f seconds"%(elapsed)
        before = time.clock()
        out = scipy.ndimage.watershed_ift(image.astype(np.uint16), markers, self.eight)
        elapsed = time.clock()-before
        print "Scipy watershed ran a megapixel image in %f seconds"%(elapsed)

