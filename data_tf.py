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
import random
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


# Define function to write LAMMPS dump file from
# a given configuration
def write_config(C, path):
    '''
    Params:
        C : tf.Tensor
            Array containing atomic positions in 3 dimensions.
            In every case, C must be of shape (N, 3), where N 
            is the number of atoms in the configuration.
        path : str
            Path of the file in which to store the given 
            configuration.
    Output:
        Saves the configuration C as a LAMMPS dump file at the
        given path.
    '''
    # Get np array out of C tensor
    C_arr = C.numpy()
    # Define boundaries for the configuration
    x_min = float(tf.reduce_min(C_arr[:,0]).numpy()) - 0.01
    x_max = float(tf.reduce_max(C_arr[:,0]).numpy()) + 0.01
    y_min = float(tf.reduce_min(C_arr[:,1]).numpy()) - 0.01
    y_max = float(tf.reduce_max(C_arr[:,1]).numpy()) + 0.01
    z_min = float(tf.reduce_min(C_arr[:,2]).numpy()) - 0.01
    z_max = float(tf.reduce_max(C_arr[:,2]).numpy()) + 0.01
    # Design header
    header = 'ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n'
    header += str(C_arr.shape[0]) + '\n'
    header += 'ITEM: BOX BOUNDS pp pp pp\n'
    header += str(x_min) + ' ' + str(x_max) + '\n'
    header += str(y_min) + ' ' + str(y_max) + '\n'
    header += str(z_min) + ' ' + str(z_max) + '\n'
    header += 'ITEM: ATOMS id type x y z\n'
    # Define body of the file
    body = '' + header
    idx = 1
    atom_type = '1'
    for i in range(C_arr.shape[0]):
        body += str(idx) + ' ' + atom_type
        for j in range(3):
            body += ' ' + str(C_arr[i,j])
        body += '\n'
        idx += 1
    # Open file in write mode
    f = open(path, 'w')
    f.write(body)
    f.close()
    return None


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
def gen_affine_A(num, min_val, max_val, mode='normal'):
    '''
    Params:
        num : int
            Number of matrices to generate.
        min_val : float
            Minimum value for any element of the
            matrices.
        max_val : float
            Maximum value for any element of the
            matrices.
        mode : str (optional)
            Either "normal" (default) or "double". If "normal",
            then the elements of the matrices are sampled 
            from a uniform distribution with range 
            [min_val, max_val). If "double", then the elements
            of the matrices are sampled in the same proportion
            from two distributions, given respectively by the
            following ranges:
                [min_val, max_val)
                (-max_val, -min_val]
    Output:
        Returns a tensor of shape (num, 3, 3) with elements
        sampled from the selected distribution:
        - normal -> [min_val, max_val) 
        - double -> (-max_val, -min_val] ^ [min_val, max_val)
        This can be understood as an array of (3, 3) shaped 
        affine transformation matrices.
    '''
    if mode == 'normal':
        A_set = (tf.random.uniform((num, 3, 3)) * (max_val - min_val)) + min_val
    elif mode == 'double':
        # Generate two distributions
        n = num // 2
        m = num // 2
        if num % 2 == 1:
            m += 1
        A_set_1 = (tf.random.uniform((n * 3 * 3, 1)) * (max_val - min_val)) + min_val
        A_set_2 = (tf.random.uniform((m * 3 * 3, 1)) * (-min_val + max_val)) - max_val
        A_set = tf.random.shuffle(tf.concat([A_set_1, A_set_2], 0), seed=None)
        A_set = tf.reshape(A_set, (num, 3, 3))
    else:
        print('Error: mode must be "normal" or "double".')
        return None
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


# Define a function that can generate a random transformation
def random_transformation(C, A, b, epsilon, d, p):
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
        d : float
            Maximum magnitude of each element of the non-affine 
            displacement vector.
        p : float
            Probability of applying the transformation. Must be 
            between 0 and 1.
    Output:
        Returns a new random configuration C_new that is built
        by applying the gen_transformation function to a given 
        configuration C with a random row index d_row for the 
        Frenkel transformation, and without establishing a seed 
        for the shuffle transformation. 
    '''
    # Generate random row
    d_row = random.randint(0,C.shape[0])
    return gen_transformation(C, A, b, epsilon, d_row, d, p)


# Define a function that can generate a dataset from a neighbor list
def dataset_from_neighbors(neighbors, A_min, A_max, 
                            b_min, b_max, epsilon, d, p,
                            seed=42, A_mode='normal'):
    '''
    Params:
        neighbors : array
            Array of shape (M, N, 3) containing the N nearest neighbors
            delta vectors for each of M atoms.
        A_min : float
            Minimu value for any element of the
            affine transformation matrices.
        A_max : float
            Maximum value for any element of the
            affine transformation matrices.
        b_min : float
            Minimum value for any element of the
            affine transformation vectors.
        b_max : float
            Maximum value for any element of the 
            affine transformation vectors.
        epsilon : float
            Maximum magnitude of each element of the non-affine
            temperature transformation matrix.
        d : float
            Maximum magnitude of each element of the non-affine 
            Frenkel displacement vector.
        p : float
            Probability of applying the Frenkel transformation. 
            Must be between 0 and 1.
        seed : int (optional)
            Seed to use for the random operations of the function.
        A_mode : str (optional)
            Sampling mode that is passed to gen_affine_A, default 
            is "normal".
    Output:
        Returns a tf.data.Dataset object containing 2 zipped 
        datasets, input data and output data:
            Input data:
            1. Original neighbor configurations, shape (M, N, 3)
            2. Transformed neighbor configurations, shape 
            (M, N, 3)
            Output data:
            1. Affine transformation matrices used, shape (M, 3, 3)
            2. Affine transformation vectors used, shape (M, 1, 3)
        This dataset can be used to train a model to predict the 
        affine transformation matrix (A) and vector (b) need to go
        from an original configuration C to a transformed 
        configuration C'. The transformed configurations C' are 
        generated for each original configuration C as:
            C' = S[O_F[O_T[(A * C^T)^T + b]]]
        Where ^T is the transposition operation, O_T is the thermal
        transformation, O_F is the Frenkel transformation and S is 
        a shuffling transformation.
    '''
    # Initialize random seed
    tf.random.set_seed(seed)
    # Get original configurations dataset
    C_data = tf.data.Dataset.from_tensor_slices(neighbors).map(
                            lambda x: tf.cast(x,tf.float32))
    # Generate A and b datasets
    A_set = gen_affine_A(neighbors.shape[0], A_min, A_max,
                         mode=A_mode)
    b_set = gen_affine_b(neighbors.shape[0], b_min, b_max)
    A_data = tf.data.Dataset.from_tensor_slices(A_set)
    b_data = tf.data.Dataset.from_tensor_slices(b_set)

    # Define aux dataset
    tri_data = tf.data.Dataset.zip((C_data, A_data, b_data))
    # Define lambda expression that can generate a transformed
    # configuration
    f = lambda C, A, b : random_transformation(C, A, b,
                                epsilon=epsilon, d=d, p=p)
    # Get new data
    C_new_data = tri_data.map(f)

    # Define input and output datasets
    input_data = tf.data.Dataset.zip((C_data, C_new_data))
    output_data = tf.data.Dataset.zip((A_data, b_data))

    # Return total dataset
    return tf.data.Dataset.zip((input_data, output_data))