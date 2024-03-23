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
import random

# We need some logic to check if either ovito or atomman are available
try:
    import ovito
    OVITO_IMPORT = True
except:
    OVITO_IMPORT = False

try:
    import atomman as am 
    AM_IMPORT = True
except:
    AM_IMPORT = False

# Warn the user if both ovito and atomman are unavailable
if not (AM_IMPORT or OVITO_IMPORT):
    print('Warning: both ovito and atomman modules were not found.')
    print('Running the data_tf module without ovito or atomman may cause errors.')

from tools_tf import *


# ------ I/O functions --------

# Define function to read files given a path
def read_file(path):
    '''
    Params:
        path : str
            Path of an ovito/atomman readable file 
            containing at least a set of atomic 
            positions.
    Output:
        With ovito module (preferred): Returns an 
        ovito Pipeline object built from the file 
        located at path. This is a wrapper around 
        ovito's io.import_file() function.

        With atomman: Returns an atomman System
        object built from the file located at path.
        This is a wrapper around atommans's load()
        function.
    '''
    if OVITO_IMPORT:
        return ovito.io.import_file(path)
    else:
        return am.load('atom_dump',path)


# Define a function to generate a pandas DataFrame 
# from a dump file
def df_from_file(path):
    '''
    Params:
        path : str
            Path of the file to be opened.
    Output:
        Returns a tuple (df, header), where df is a
        pandas DataFrame object containing all of the
        data found in the file. Additionally, header
        is a list of the header lines of the file, 
        which can be used to write a new dump file
        after analysing the data.
    '''
    # Import pandas
    import pandas as pd
    # Open file and scan for header
    file = open(path,'r')
    found_cols = False
    header_lines = []
    line_count = 0
    while not found_cols:
        line = file.readline()
        line_count += 1
        if 'ITEM: ATOMS' in line:
            columns = line.split()
            columns.pop(0) # Pop 'ITEM:'
            columns.pop(0) # Pop 'ATOMS'
            found_cols = True
        header_lines.append(line)
    file.close()
    # Read csv data
    df = pd.read_csv(path,sep=' ', 
                     skiprows=line_count, names=columns)
    return df, header_lines


# Define a function to write LAMMPS dump file from 
# a pandas DataFrame and a list of header lines
def write_dump_from_df(df, header, path, new_cols=None,
                        replace_cols=False):
    '''
    Params:
        df : pandas DataFrame
            DataFrame containing all of the atomic
            data that is going to be written.
        header : list
            List of strings that are going to be
            used as header lines of the dump file.
            This header can usually be obtained when
            reading a LAMMPS dump with the 
            df_from_file function. The information in
            header must correspond to that found in df,
            i.e. correct boundaries and atom properties.
        path : str
            Path for the new file.
        new_cols : list (optional)
            List containing the names of the new columns
            in df, which are not specified in header. If
            not given, it is assumed that no new columns
            have been added to df with respect to header.
            The ordering of new_cols must correspond to
            the ordering of the respective columns in df.
        replace_cols : bool (optional)
            Whether to replace header columns contained 
            in new_cols or not. Default is False.
    Output:
        Writes a new file at path, containing the 
        information provided by df, below the lines
        stated in header. 
    '''
    new_file = open(path, 'w')
    
    # Write the header of the file
    feat_line = ''
    for col in new_cols:
        feat_line += ' ' + col
    feat_line += '\n'
    for i in range(len(header)):
        line = header[i]
        if i == len(header) - 1:
            if replace_cols:
                split_line = line.split()
                new_line = ''
                for col in split_line:
                    if not col in new_cols and col == '\n':
                        new_line += col 
                    elif not col in new_cols and col != '\n':
                        new_line += col + ' '
                line = new_line
            line = line.replace('\n', feat_line)
        new_file.write(line)
    
    # Now let's write the new data
    final_string = df.to_csv(index=False, sep=' ', 
                             float_format='%s', header=False)
    new_file.write(final_string)
    # Close the file and return
    new_file.close()
    return None


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
        If deltas=False, the indices of the neighbors of 
        each atom are returned (not recommended when running
        with atomman). If deltas=True, an array of distance
        vectors between each atom and its neighbors is 
        returned. Availability of the ovito module is highly
        recommended for this function, as it is more 
        efficient than the atomman implementation (which 
        additionally relies on pandas and scikit-learn).
    '''
    # Preferring ovito if available
    if OVITO_IMPORT:
        pipeline = read_file(path)
        computed = pipeline.compute()
        finder = ovito.data.NearestNeighborFinder(N, computed)
        neighbors = finder.find_all()
        if deltas:
            return neighbors[1]
        return neighbors[0]

    # Else, assuming atomman is available
    else: 
        # Try importing NearestNeighbors algorithm, it is needed
        # for this function to be computationally feasible
        try:
            from sklearn.neighbors import NearestNeighbors
        except:
            mssg = "sklearn.neighbors.NearestNeighbors not found"
            raise Exception("Error: " + mssg)
        # Read system from path
        system = read_file(path) 
        # Get array of original positions
        og_pos = tf.constant(system.atoms.pos, dtype='float32')
        # Define an array that will contain original indices
        og_idx = tf.range(og_pos.shape[0], dtype='int32')
        # Now we need to build a new system with the periodic 
        # boundary conditions, and then use the NearestNeighbors
        # model on it:
        # 1. Get pbc conditions
        og_pbc = system.pbc
        # 2. Build supersize conditions based on pbc
        scale_vals = [(-1,2) if c == True else (0,1) for c in og_pbc]
        # 3. Build supersized system
        super_sys_pos = tf.constant(
            system.supersize(scale_vals[0], 
                             scale_vals[1], 
                             scale_vals[2]).atoms.pos,
            dtype='float32')
        # 4. Train NearestNeighbors on the super system
        nn_model = NearestNeighbors(n_neighbors=N+1, n_jobs=-1)
        nn_model.fit(super_sys_pos)
        # 5. Get neighbors for original system and drop self 
        # connections
        ordered_pos = tf.constant(df_from_file(path)[0][['x','y','z']],
                                  dtype='float32')
        neigh_dist, neigh_idx = nn_model.kneighbors(ordered_pos)
        neigh_idx = neigh_idx[:,1:]
        if deltas:
            # Get neighbor vectors
            neigh_vecs = tf.gather(super_sys_pos, neigh_idx)
            # Broadcast central atom position
            expanded_og_pos = tf.broadcast_to(
                            tf.expand_dims(ordered_pos, axis=1),
                            neigh_vecs.shape)             
            # Get delta vectors     
            neigh_deltas = neigh_vecs - expanded_og_pos
            return neigh_deltas
        else:
            return neigh_idx % og_pos.shape[0]


# Define a function that returns the partial neighbor
# configurations, only considering a given particle type
def neighbor_types_from_file(path, N):
    '''
    Params:
        path : str
            Path of an ovito readable file containing
            at least a set of atomic positions.
        N : int
            Number of neighbors to find for each atom.
    Output:
        
    '''
    # Preferring ovito if available
    if OVITO_IMPORT:
        pipeline = read_file(path)
        computed = pipeline.compute()
        finder = ovito.data.NearestNeighborFinder(N, computed)
        neighbors = finder.find_all()
        # Get particle types
        p_type_arr = tf.constant(list(computed.particles.particle_type))
        # Use neighbor indexes to gather particle types of neighbors
        return tf.gather(p_type_arr,neighbors[0])
    else:
        raise('Error: function unavailable without ovito import.')


# Define a function to read a dump and output the ids as 
# a list
def ids_from_file(path):
    '''
    Params:
        path : str
            Path of the file.
    Output:
        Returns a list of each particle's identifier
        ordered as they appear in the file.
    '''
    # Open file 
    file = open(path, 'r')
    # Initialize empty list
    ids = []
    # Skip lines until finding ids
    found_cols = False
    while not found_cols:
        line = file.readline()
        if 'ITEM: ATOMS' in line:
            found_cols = True
    # Star registering ids
    for line in file.readlines():
        ids.append(int(line.split()[0]))
    # Close file
    file.close()

    return ids


# Define a function that outputs the neighbour configuration
# of an atom, given its id
def get_config_from_id(neighbors, id_list, atom_id, 
                       dtype='float32'):
    '''
    Params:
        neighbors : array
            Array of shape (M, N, 3) containing the N nearest
            neighbors delta vectors for each of M atoms.
        id_list : list
            List containing the M atom ids corresponding to 
            each configuration in neighbors.
        atom_id : int
            Identifier number of the desired atom.
        dtype : str or tf.dtype
            Data type for the resulting tensor, default is
            float32.
    Output:
        Returns a tf.Tensor that contains the neighbour
        configuration of the selected atom.
    '''
    # Get index of given id
    index = id_list.index(atom_id)
    # Generate config
    C = tf.constant(neighbors[index], dtype=dtype)
    return C

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


# Define function to output raw configuration databases
def raw_dataset_from_neighbors(neighbors, A_min, A_max, 
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
        Returns four tf.data.Dataset objects containing the
        following datasets:
            1. Original neighbor configurations, shape (M, N, 3)
            2. Transformed neighbor configurations, shape 
            (M, N, 3)
            3. Affine transformation matrices used, shape (M, 3, 3)
            4. Affine transformation vectors used, shape (M, 1, 3)
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

    # Return datasets
    return C_data, C_new_data, A_data, b_data

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
    # Get raw datasets
    C_data, C_new_data, A_data, b_data = raw_dataset_from_neighbors(
                            neighbors, A_min, A_max, 
                            b_min, b_max, epsilon, d, p,
                            seed=seed, A_mode=A_mode)

    # Define input and output datasets
    input_data = tf.data.Dataset.zip((C_data, C_new_data))
    output_data = tf.data.Dataset.zip((A_data, b_data))

    # Return total dataset
    return tf.data.Dataset.zip((input_data, output_data))