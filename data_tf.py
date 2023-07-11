# Copyright 2023 Franco Aquistapace
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

# ----- Here we'll define the data related functions of the project ------
# We'll use a functional programming approach for now, we can develop classes 
# later if we need them.

# Import modules
import tensorflow as tf
import ovito
from tools_tf import *


# ------ I/O functions --------

# Define function to read files given a path
def read_file(path):
    '''
    Params:
        path : str
            Path of an ovito readable file containing
            at least a set of atomic positions.
    Output:
        Returns an ovito Pipeline object built from 
        the file located at path. This is a wrapper 
        around ovito's io.import_file() function.
    '''
    return ovito.io.import_file(path)


# ------ Data functions --------

# Define a function that returns the neighbour list
# for a given path
def neighbors_from_file(path, N, deltas=True):
    '''
    Params:
        path : str
            Path of an ovito readable file containing
            at least a set of atomic positions.
        N : int
            Number of neighbors to find for each atom.
        deltas : bool (optional)
            When True, returns an array of shape 
            (M, N, 3) with the delta vectors of each 
            atom's neighbors, where M is the amount of 
            atoms in the system. If False, returns an 
            array of shape (M, N) containing the indices
            of the neighbors of each atom.
    Output:
        Returns an array containing information about the
        N nearest neighbors of each atom in the given path.
    '''
    pipeline = read_file(path)
    computed = pipeline.compute()
    finder = ovito.data.NearestNeighborFinder(N, computed)
    neighbors = finder.find_all()
    if deltas:
        return neighbors[1]
    return neighbors[0]


# Define a function to generate a random affine and non-affine
# transformation
def gen_transformation(C, A, b, epsilon, d_row, d, p, 
                       seed=None):
    '''
    Params:
        C : tf.Tensor
            Array containing atomic positions in 3 dimensions.
            In every case, C must be of shape (N, 3), where N 
            is the number of atoms in the configuration.
        A : tf.Tensor
            Array of shape (3,3) that describes a generalized
            affine transformation except for displacements.
        b : tf.Tensor
            Array of shape (1,3) that describes an affine 
            displacement.
        epsilon : float
            Maximum magnitude of each element of the non-affine
            transformation matrix.
        d_row : int
            Row index to apply the non-affine displacement, must
            be between 0 and N-1.
        d : float
            Maximum magnitude of each element of the non-affine 
            displacement vector.
        p : float
            Probability of applying the transformation. Must be 
            between 0 and 1.
        seed : int (optional)
            Seed to use for the random generator of the shuffle.
    Output:
        Returns a new configuration C_new defined as:
            C_new = S[O_F[O_T[(A * C^T)^T + b]]]
        Where ^T is the transposition operation, O_T is the thermal
        transformation, O_F is the Frenkel transformation and S is 
        a shuffling transformation.
    '''
    C_new = shuffle_transformation(
            frenkel_transformation(
            thermal_transformation(
            affine_transformation(C, A, b), 
            epsilon), 
            d_row, d, p), 
            seed=seed)
    return C_new


# Define function to generate dataset of affine transformation
# matrices
def gen_affine_A(num, min_val, max_val):
    '''
    Params:
        num : int
            Number of matrices to generate.
        min_val : float
            Minimu value for any element of the
            matrices.
        max_val : float
            Maximum value for any element of the
            matrices.
    Output:
        Returns a tensor of shape (num, 3, 3) with elements
        sampled from a uniform distribution in the range
        [min_val, max_val). This can be understood as an 
        array of (3, 3) shaped affine transformation matrices.
    '''
    A_set = (tf.random.uniform((num, 3, 3)) * (max_val - min_val)) + min_val
    return A_set


# Define function to generate dataset of affine displacement
# vectors
def gen_affine_b(num, min_val, max_val):
    '''
    Params:
        num : int
            Number of vectors to generate.
        min_val : float
            Minimum value for any element of the
            vectors.
        max_val : float
            Maximum value for any element of the 
            vectors.
    Output:
        Returns a tensor of shape (num, 1, 3) with elements 
        sampled from a uniform distribution in the range
        [min_val, max_val). This can be understood as an 
        array of (1, 3) shaped affine displacement vectors.
    '''
    b_set = (tf.random.uniform((num, 1, 3)) * (max_val - min_val)) + min_val
    return b_set